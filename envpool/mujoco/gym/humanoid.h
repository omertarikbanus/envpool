#ifndef ENVPOOL_MUJOCO_GYM_HUMANOID_H_
#define ENVPOOL_MUJOCO_GYM_HUMANOID_H_

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>  // For std::sqrt, std::abs
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"

#include <Eigen/Dense>

#include "control/ControlMessages.hh"
#include "control/LocomotionCtrlData.hh"
#include "controllers/LegController.h"
#include "estimators/StateEstimatorTypes.hh"
#include "hardware/IMUHW.hh"
#include "hardware/MotorHW.hh"
#include "hardware/SimHW.hh"
#include "modules/MdlControlParams.hh"
#include "modules/MdlCtGaitScheduler.hh"
#include "modules/MdlFootstepPlanner.hh"
#include "modules/MdlLegController.hh"
#include "modules/MdlRLLocomotionState.hh"
#include "modules/MdlStateEstimator.hh"
#include "modules/MdlWBIC.hh"
#include "rtcore/ClockHW.hh"
#include "rtcore/ConfigTable.hh"
#include "rtcore/ModuleManager.hh"
#include "types/cppTypes.h"
#include "MdlSimDriver.hh"

namespace mujoco_gym {

struct RLConstants {
  static constexpr int kNumLegs = 4;
  static constexpr int kActionDim = 24;
  static constexpr int kPhaseDeltaIdx = kActionDim - 1;
  static constexpr int kObservationDim = 46;
};

struct RewardWeights {
  mjtNum base_xvel{static_cast<mjtNum>(0.25)};
  mjtNum base_zvel{static_cast<mjtNum>(0.10)};
  mjtNum base_zpos{static_cast<mjtNum>(0.20)};
  mjtNum base_orientation{static_cast<mjtNum>(0.15)};
  mjtNum base_straight{static_cast<mjtNum>(0.15)};
  mjtNum base_linear_accel{static_cast<mjtNum>(0.025)};
  mjtNum base_angular_vel{static_cast<mjtNum>(0.025)};
  mjtNum action_smooth{static_cast<mjtNum>(0.10)};
  mjtNum phase_delta{static_cast<mjtNum>(0.05)};
};

struct ReferenceTargets {
  mjtNum xdot_ref{static_cast<mjtNum>(0.8)};
  mjtNum zdot_ref{static_cast<mjtNum>(0.0)};
  mjtNum z_ref{static_cast<mjtNum>(0.35)};
};

struct RewardConfig {
  RewardWeights weights{};
  ReferenceTargets refs{};
};

// Computes a Cassie-inspired locomotion reward from instantaneous penalties.
class LocomotionReward {
 public:
  static constexpr int kNumTerms = 9;
  enum TermIndex {
    kBaseXVel = 0,
    kBaseZVel,
    kBaseZPos,
    kBaseOrientation,
    kBaseStraight,
    kBaseLinearAccel,
    kBaseAngularVel,
    kActionSmooth,
    kPhaseDelta
  };

  struct Result {
    mjtNum total{0};
    std::array<mjtNum, kNumTerms> penalties{};
    std::array<mjtNum, kNumTerms> smoothed_penalties{};
    std::array<mjtNum, kNumTerms> rewards{};
  };

  explicit LocomotionReward(const RewardConfig& config) : config_(config) {}

  void Reset() {}

  void SetReferences(const ReferenceTargets& refs) { config_.refs = refs; }

  Result Compute(const Eigen::Matrix<mjtNum, 3, 1>& base_lin_vel,
                 const Eigen::Matrix<mjtNum, 3, 1>& base_pos,
                 const Eigen::Quaternion<mjtNum>& base_quat,
                 const Eigen::Matrix<mjtNum, 3, 1>& base_lin_acc,
                 const Eigen::Matrix<mjtNum, 3, 1>& base_ang_vel,
                 const std::vector<mjtNum>& action,
                 const std::vector<mjtNum>& prev_action) {
    Result result;
    constexpr mjtNum kVelPenaltyScale = static_cast<mjtNum>(3.0);

    result.penalties[kBaseXVel] =
        kVelPenaltyScale * std::abs(base_lin_vel[0] - config_.refs.xdot_ref);
    result.penalties[kBaseZVel] =
        kVelPenaltyScale * std::abs(base_lin_vel[2] - config_.refs.zdot_ref);
    result.penalties[kBaseZPos] =
        kVelPenaltyScale * std::abs(base_pos[2] - config_.refs.z_ref);

    Eigen::Quaternion<mjtNum> q = base_quat;
    if (std::abs(static_cast<double>(q.norm() - 1.0)) > 1e-6) {
      q.normalize();
    }
    const Eigen::Quaternion<mjtNum> q_neutral(static_cast<mjtNum>(1.0),
                                               static_cast<mjtNum>(0.0),
                                               static_cast<mjtNum>(0.0),
                                               static_cast<mjtNum>(0.0));
    mjtNum dot_q = q.dot(q_neutral);
    dot_q = std::clamp(dot_q, static_cast<mjtNum>(-1.0), static_cast<mjtNum>(1.0));
    result.penalties[kBaseOrientation] =
        static_cast<mjtNum>(50.0) * (static_cast<mjtNum>(1.0) - dot_q);

    result.penalties[kBaseStraight] =
        static_cast<mjtNum>(5.0) * std::abs(base_pos[1]) +
        static_cast<mjtNum>(3.0) * std::abs(base_lin_vel[1]);

    result.penalties[kBaseLinearAccel] = base_lin_acc.squaredNorm();
    result.penalties[kBaseAngularVel] = base_ang_vel.squaredNorm();
    result.penalties[kActionSmooth] =
        ComputeActionSmoothPenalty(action, prev_action);
    result.penalties[kPhaseDelta] = ComputePhaseDeltaPenalty(action);

    result.smoothed_penalties = result.penalties;  // Legacy field, now identical to raw penalties.
    for (int idx = 0; idx < kNumTerms; ++idx) {
      result.rewards[idx] = std::exp(-result.penalties[idx]);
    }

    const auto& w = config_.weights;
    result.total = w.base_xvel * result.rewards[kBaseXVel] +
                   w.base_zvel * result.rewards[kBaseZVel] +
                   w.base_zpos * result.rewards[kBaseZPos] +
                   w.base_orientation * result.rewards[kBaseOrientation] +
                   w.base_straight * result.rewards[kBaseStraight] +
                   w.base_linear_accel * result.rewards[kBaseLinearAccel] +
                   w.base_angular_vel * result.rewards[kBaseAngularVel] +
                   w.action_smooth * result.rewards[kActionSmooth] +
                   w.phase_delta * result.rewards[kPhaseDelta];
    return result;
  }

 private:
  mjtNum ComputeActionSmoothPenalty(const std::vector<mjtNum>& action,
                                    const std::vector<mjtNum>& prev_action) const {
    if (action.empty() || prev_action.empty()) {
      return static_cast<mjtNum>(0.0);
    }
    const std::size_t n = std::min(action.size(), prev_action.size());
    mjtNum diff_sq_sum = 0;
    for (std::size_t i = 0; i < n; ++i) {
      const mjtNum diff = action[i] - prev_action[i];
      diff_sq_sum += diff * diff;
    }
    return static_cast<mjtNum>(3.0) * diff_sq_sum;
  }

  mjtNum ComputePhaseDeltaPenalty(const std::vector<mjtNum>& action) const {
    if (action.size() <= RLConstants::kPhaseDeltaIdx) {
      return static_cast<mjtNum>(0.0);
    }
    // Linear penalty: zero cost when delta_theta <= -1, increasing linearly above that.
    const mjtNum delta_theta = action[RLConstants::kPhaseDeltaIdx];
    if (delta_theta <= static_cast<mjtNum>(-1.0)) {
      return static_cast<mjtNum>(0.0);
    }
    return delta_theta + static_cast<mjtNum>(1.0);
  }

  RewardConfig config_;
};

class HumanoidEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict(
        "frame_skip"_.Bind(5), "post_constraint"_.Bind(true),
        "use_contact_force"_.Bind(false), "forward_reward_weight"_.Bind(1),
        "terminate_when_unhealthy"_.Bind(true),
        "render_mode"_.Bind(false),
        "video_record"_.Bind(-1),
        "video_path"_.Bind(std::string("")),
        "video_width"_.Bind(0),
        "video_height"_.Bind(0),
        "video_fps"_.Bind(0.0),
        "csv_logging_enabled"_.Bind(false), 
        "random_force_enabled"_.Bind(false),
        "random_force_min"_.Bind(0.0),
        "random_force_max"_.Bind(30.0),
        "random_force_hold_steps"_.Bind(20),
        "render_env_id"_.Bind(0),
        "model_xml_path"_.Bind(std::string("/app/quadcontrol/models/demir_1/scene.xml")),
        "urdf_path"_.Bind(std::string("/app/quadcontrol/models/demir_1/demir_1.urdf")),
        "sim_config_path"_.Bind(std::string("/app/quadcontrol/config/robots/sim/rl_sim.toml")),
        "robot_namespace"_.Bind(std::string("go2")),
        "exclude_current_positions_from_observation"_.Bind(true),
        "ctrl_cost_weight"_.Bind(2e-4), "healthy_reward"_.Bind(1.0),
        "healthy_z_min"_.Bind(0.20), "healthy_z_max"_.Bind(0.75),
        "contact_cost_weight"_.Bind(5e-7), "contact_cost_max"_.Bind(10.0),
        "velocity_tracking_weight"_.Bind(0.5),
        "yaw_tracking_weight"_.Bind(0.2),
        "orientation_penalty_weight"_.Bind(0.1),
        "height_penalty_weight"_.Bind(.9),
        "foot_slip_penalty_weight"_.Bind(0.1),
        "action_penalty_weight"_.Bind(5e-2),
        "cmd_profile_mode"_.Bind(std::string("random_episode")),
        "cmd_fixed_vx"_.Bind(0.5),
        "cmd_fixed_vy"_.Bind(0.0),
        "cmd_fixed_yaw"_.Bind(0.0),
        "cmd_rand_vx_min"_.Bind(-3.5),
        "cmd_rand_vx_max"_.Bind(3.5),
        "cmd_rand_vy_min"_.Bind(-0.5),
        "cmd_rand_vy_max"_.Bind(0.5),
        "cmd_rand_yaw_min"_.Bind(-0.4),
        "cmd_rand_yaw_max"_.Bind(0.4),
        "cmd_tracking_weight"_.Bind(1.0),
        "cmd_residual_linear_limit"_.Bind(0.5),
        "cmd_residual_yaw_limit"_.Bind(0.3),
        "reset_noise_scale"_.Bind(0));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    mjtNum inf = std::numeric_limits<mjtNum>::infinity();
    return MakeDict("obs"_.Bind(
                        Spec<mjtNum>({RLConstants::kObservationDim},
                                     {-inf, inf})),
#ifdef ENVPOOL_TEST
                    "info:qpos0"_.Bind(Spec<mjtNum>({24})),
                    "info:qvel0"_.Bind(Spec<mjtNum>({23})),
#endif
                    "info:reward_linvel"_.Bind(Spec<mjtNum>({-1})),
                    "info:reward_quadctrl"_.Bind(Spec<mjtNum>({-1})),
                    "info:reward_alive"_.Bind(Spec<mjtNum>({-1})),
                    "info:reward_impact"_.Bind(Spec<mjtNum>({-1})),
                    "info:x_position"_.Bind(Spec<mjtNum>({-1})),
                    "info:y_position"_.Bind(Spec<mjtNum>({-1})),
                    "info:distance_from_origin"_.Bind(Spec<mjtNum>({-1})),
                    "info:x_velocity"_.Bind(Spec<mjtNum>({-1})),
                    "info:y_velocity"_.Bind(Spec<mjtNum>({-1})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    // Action layout (24 dims total):
    // [0] vBody_des.x residual, [1] vBody_des.y residual, [2] yaw_rate residual,
    // Per-leg (4 legs), stride 5:
    //   [3 + 5*leg + 0..2] ground reaction force target (x, y, z) when in contact
    //   [3 + 5*leg + 3..4] foot placement residual (x, y) for swing target
    // [23] gait phase delta_theta (normalized, mapped to 0..kMaxPhaseDelta)
    return MakeDict("action"_.Bind(
        Spec<mjtNum>({-1, RLConstants::kActionDim}, {-1, 1})));
  }
};

using HumanoidEnvSpec = EnvSpec<HumanoidEnvFns>;

class HumanoidEnv : public Env<HumanoidEnvSpec> {
 protected:
  struct HardwareThreadGuard {
    HardwareThreadGuard(MotorHW* motor, IMUHW* imu, rtcore::ClockHW* clock)
        : prev_motor_(MotorHW::threadInstance()),
          prev_imu_(IMUHW::threadInstance()),
          prev_clock_(rtcore::ClockHW::threadInstance()) {
      MotorHW::setThreadInstance(motor);
      IMUHW::setThreadInstance(imu);
      rtcore::ClockHW::setThreadInstance(clock);
    }

    ~HardwareThreadGuard() {
      MotorHW::setThreadInstance(prev_motor_);
      IMUHW::setThreadInstance(prev_imu_);
      rtcore::ClockHW::setThreadInstance(prev_clock_);
    }

    MotorHW* prev_motor_;
    IMUHW* prev_imu_;
    rtcore::ClockHW* prev_clock_;
  };

  bool terminate_when_unhealthy_, no_pos_, use_contact_force_, render_mode_, csv_logging_enabled_;
  bool random_force_enabled_;
  mjtNum ctrl_cost_weight_, forward_reward_weight_, healthy_reward_;
  mjtNum healthy_z_min_, healthy_z_max_;
  mjtNum contact_cost_weight_, contact_cost_max_;
  std::uniform_real_distribution<> dist_;

  rtcore::ModuleManager module_manager_;

  MdlSimDriver* sim_driver_{nullptr};
  quadruped::MdlControlParams* control_params_{nullptr};
  quadruped::MdlStateEstimator* state_estimator_{nullptr};
  quadruped::MdlLegController* leg_controller_module_{nullptr};
  quadruped::MdlCtGaitScheduler* ct_gait_scheduler_{nullptr};
  quadruped::MdlFootstepPlanner* footstep_planner_{nullptr};
  quadruped::MdlRLLocomotionState* rl_state_module_{nullptr};
  quadruped::MdlWBIC* wbic_{nullptr};
  MotorHW* motor_hw_{nullptr};
  IMUHW* imu_hw_{nullptr};
  rtcore::ClockHW* clock_hw_{nullptr};
  SimHWContext* sim_hw_ctx_{nullptr};

  int frame_skip_{};
  int max_episode_steps_{};
  int elapsed_step_{0};
  bool done_{true};
  bool disabled_{false};

  quadruped::GaitSchedule last_gait_schedule_{};
  quadruped::StateEstimate<float> last_state_est_{};

  std::ofstream outputFile;
  float lastReward = 0;
  mjtNum desired_h;
  mjtNum velocity_tracking_weight_;
  mjtNum yaw_tracking_weight_;
  mjtNum orientation_penalty_weight_;
  mjtNum height_penalty_weight_;
  mjtNum foot_slip_penalty_weight_;
  mjtNum action_penalty_weight_;
  RewardConfig locomotion_reward_config_;
  LocomotionReward locomotion_reward_;
  std::array<mjtNum, LocomotionReward::kNumTerms> last_penalties_{};
  std::array<mjtNum, LocomotionReward::kNumTerms> last_smoothed_penalties_{};
  std::array<mjtNum, LocomotionReward::kNumTerms> last_term_rewards_{};
  std::string cmd_profile_mode_;
  std::array<mjtNum, 3> cmd_fixed_cmd_vel_;
  mjtNum cmd_rand_vx_min_;
  mjtNum cmd_rand_vx_max_;
  mjtNum cmd_rand_vy_min_;
  mjtNum cmd_rand_vy_max_;
  mjtNum cmd_rand_yaw_min_;
  mjtNum cmd_rand_yaw_max_;
  mjtNum cmd_tracking_weight_;
  mjtNum cmd_residual_linear_limit_;
  mjtNum cmd_residual_yaw_limit_;
  mjtNum random_force_min_;
  mjtNum random_force_max_;
  int random_force_hold_steps_;
  int random_force_steps_remaining_{0};
  Eigen::Matrix<mjtNum, 6, 1> random_force_cached_{
      Eigen::Matrix<mjtNum, 6, 1>::Zero()};
  Eigen::Matrix<mjtNum, 6, 1> filtered_random_force_{
      Eigen::Matrix<mjtNum, 6, 1>::Zero()};
  mjtNum random_force_filter_tau_{static_cast<mjtNum>(0.01)};  // seconds
  Eigen::Matrix<mjtNum, 6, 1> last_applied_wrench_{
      Eigen::Matrix<mjtNum, 6, 1>::Zero()};
  std::array<mjtNum, 3> cmd_vel_target_body_{{0.0, 0.0, 0.0}};
  std::mt19937 cmd_rng_;
  std::mt19937 reset_rng_;
  std::vector<mjtNum> last_action_vector_;
  std::vector<mjtNum> prev_action_vector_;
  std::vector<mjtNum> last_observation_;
  mjtNum last_ctrl_cost_{0.0};
  mjtNum last_contact_cost_{0.0};
  mjtNum last_healthy_reward_{0.0};
  mjtNum last_x_velocity_{0.0};
  mjtNum last_y_velocity_{0.0};
  mjtNum last_x_position_{0.0};
  mjtNum last_y_position_{0.0};
  bool last_is_healthy_{true};
  bool pending_reset_marker_{true};
  // Added: CSV logging switch (set to 1 manually to enable CSV writing)
  std::string csv_filename_;
  std::string last_done_reason_{"init"};
  std::string model_xml_path_;
  std::string urdf_path_;
  std::string sim_config_path_;
  std::string robot_namespace_;
  int render_env_id_{0};
  int video_record_;
  std::string video_path_;
  int video_width_;
  int video_height_;
  double video_fps_;

 public:
  HumanoidEnv(const Spec& spec, int env_id)
      : Env<HumanoidEnvSpec>(spec, env_id),
        terminate_when_unhealthy_(spec.config["terminate_when_unhealthy"_]),
        no_pos_(spec.config["exclude_current_positions_from_observation"_]),
        use_contact_force_(spec.config["use_contact_force"_]),
        render_mode_(spec.config["render_mode"_]),
        csv_logging_enabled_(spec.config["csv_logging_enabled"_]),
        random_force_enabled_(spec.config["random_force_enabled"_]),
        ctrl_cost_weight_(spec.config["ctrl_cost_weight"_]),
        forward_reward_weight_(spec.config["forward_reward_weight"_]),
        healthy_reward_(spec.config["healthy_reward"_]),
        healthy_z_min_(spec.config["healthy_z_min"_]),
        healthy_z_max_(spec.config["healthy_z_max"_]),
        contact_cost_weight_(spec.config["contact_cost_weight"_]),
        contact_cost_max_(spec.config["contact_cost_max"_]),
        dist_(-spec.config["reset_noise_scale"_],
              spec.config["reset_noise_scale"_]),
        frame_skip_(spec.config["frame_skip"_]),
        max_episode_steps_(spec.config["max_episode_steps"_]),
        velocity_tracking_weight_(spec.config["velocity_tracking_weight"_]),
        yaw_tracking_weight_(spec.config["yaw_tracking_weight"_]),
        orientation_penalty_weight_(spec.config["orientation_penalty_weight"_]),
        height_penalty_weight_(spec.config["height_penalty_weight"_]),
        foot_slip_penalty_weight_(spec.config["foot_slip_penalty_weight"_]),
        action_penalty_weight_(spec.config["action_penalty_weight"_]),
        locomotion_reward_config_(),
        locomotion_reward_(locomotion_reward_config_),
        cmd_profile_mode_(spec.config["cmd_profile_mode"_]),
        cmd_fixed_cmd_vel_{{spec.config["cmd_fixed_vx"_],
                            spec.config["cmd_fixed_vy"_],
                            spec.config["cmd_fixed_yaw"_]}},
        cmd_rand_vx_min_(spec.config["cmd_rand_vx_min"_]),
        cmd_rand_vx_max_(spec.config["cmd_rand_vx_max"_]),
        cmd_rand_vy_min_(spec.config["cmd_rand_vy_min"_]),
        cmd_rand_vy_max_(spec.config["cmd_rand_vy_max"_]),
        cmd_rand_yaw_min_(spec.config["cmd_rand_yaw_min"_]),
        cmd_rand_yaw_max_(spec.config["cmd_rand_yaw_max"_]),
        cmd_tracking_weight_(spec.config["cmd_tracking_weight"_]),
        cmd_residual_linear_limit_(spec.config["cmd_residual_linear_limit"_]),
        cmd_residual_yaw_limit_(spec.config["cmd_residual_yaw_limit"_]),
        random_force_min_(spec.config["random_force_min"_]),
        random_force_max_(spec.config["random_force_max"_]),
        random_force_hold_steps_(spec.config["random_force_hold_steps"_]),
        cmd_rng_(std::random_device{}() + env_id),
        reset_rng_(std::random_device{}() + env_id + 101),
        model_xml_path_(spec.config["model_xml_path"_]),
        urdf_path_(spec.config["urdf_path"_]),
        sim_config_path_(spec.config["sim_config_path"_]),
        robot_namespace_(spec.config["robot_namespace"_]),
        render_env_id_(spec.config["render_env_id"_]),
        video_record_(spec.config["video_record"_]),
        video_path_(spec.config["video_path"_]),
        video_width_(spec.config["video_width"_]),
        video_height_(spec.config["video_height"_]),
        video_fps_(spec.config["video_fps"_]) {
    if (env_id_ == 0 || env_id_ == 1) {
      csv_logging_enabled_ = true;
    }
    const bool allow_render_env = (env_id_ == render_env_id_);
    if (!allow_render_env) {
      // Only the primary environment may render or record video.
      render_mode_ = false;
      video_record_ = 0;
      video_path_.clear();
      video_width_ = 0;
      video_height_ = 0;
      video_fps_ = 0.0;
    }
    // Disable external force application for stability.
    random_force_enabled_ = false;

    random_force_hold_steps_ = std::max(1, random_force_hold_steps_);

    csv_filename_ = "/app/envpool/data/current/" + std::to_string(env_id_) + "_log.csv";
    ClearCsvLogs();

    {
      static std::mutex s_init_mutex;
      std::lock_guard<std::mutex> lock(s_init_mutex);
      InitializeQuadcontrol();
      desired_h = 0.35;  // Set the desired height reference for reward shaping
      ResetSimulation();
    }
    if (sim_driver_) {
      sim_driver_->clearExternalWrench();
    }
    done_ = false;
    elapsed_step_ = 0;
    ResampleCommandVelocity();
    locomotion_reward_.Reset();
    UpdateRewardReferences();
    if (!disabled_) {
      HardwareThreadGuard hw_guard(motor_hw_, imu_hw_, clock_hw_);
      module_manager_.stepOnce();
    }
    CacheModuleState();
    prev_action_vector_.clear();
    writeDataToCSV(2);
  }

  ~HumanoidEnv() override {
    if (disabled_) {
      return;
    }
    HardwareThreadGuard hw_guard(motor_hw_, imu_hw_, clock_hw_);
    module_manager_.shutdown();
    cleanupHardware(&module_manager_);
    delete control_params_;
    delete state_estimator_;
    delete leg_controller_module_;
    delete ct_gait_scheduler_;
    delete footstep_planner_;
    delete rl_state_module_;
    delete wbic_;
    control_params_ = nullptr;
    state_estimator_ = nullptr;
    leg_controller_module_ = nullptr;
    ct_gait_scheduler_ = nullptr;
    footstep_planner_ = nullptr;
    rl_state_module_ = nullptr;
    wbic_ = nullptr;
  }

  bool IsDone() override { return done_; }

  void Reset() override {
    if (disabled_) {
      done_ = true;
      return;
    }
    std::cout << "[HumanoidEnv] Reset env_id=" << env_id_
              << " (prev reason=" << last_done_reason_
              << ", elapsed_step=" << elapsed_step_ << ")\n";
    HardwareThreadGuard hw_guard(motor_hw_, imu_hw_, clock_hw_);
    writeDataToCSV(1);
    last_action_vector_.clear();
    prev_action_vector_.clear();
    last_observation_.clear();
    random_force_steps_remaining_ = 0;
    random_force_cached_.setZero();
    filtered_random_force_.setZero();
    last_applied_wrench_.setZero();
    last_penalties_.fill(static_cast<mjtNum>(0.0));
    last_smoothed_penalties_.fill(static_cast<mjtNum>(0.0));
    last_term_rewards_.fill(static_cast<mjtNum>(0.0));
    locomotion_reward_.Reset();
    if(render_mode_)
      std::cout << "Resetting the environment..." << std::endl;
    done_ = false;
    elapsed_step_ = 0;
    std::uniform_real_distribution<mjtNum> dis(0.32, 0.4);
    desired_h = dis(reset_rng_);
    ResetModules();
    ResetSimulation();
    if (sim_driver_) {
      sim_driver_->clearExternalWrench();
    }
    module_manager_.stepOnce();
    CacheModuleState();
    WriteState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    ResampleCommandVelocity();
    UpdateRewardReferences();
    if (csv_logging_enabled_) {
      pending_reset_marker_ = true;
    }

  }

  void ClearCsvLogs() {
    if (csv_logging_enabled_) {
      if (outputFile.is_open()) {
        outputFile.close();
      }
      if (!csv_filename_.empty()) {
        std::remove(csv_filename_.c_str());
      }
      if (!csv_filename_.empty()) {
        std::filesystem::path path(csv_filename_);
        std::error_code ec;
        std::filesystem::create_directories(path.parent_path(), ec);
      }
      outputFile.open(csv_filename_.c_str(), std::ios::out | std::ios::trunc);
      if (!outputFile.is_open()) {
        std::cerr << "Error: Unable to recreate CSV log file at " << csv_filename_ << std::endl;
      }
      pending_reset_marker_ = true;
    }

  }

  void Step(const Action& action) override {
    if (disabled_) {
      done_ = true;
      return;
    }
    HardwareThreadGuard hw_guard(motor_hw_, imu_hw_, clock_hw_);
    auto action_array = action["action"_];
    auto* act = static_cast<mjtNum*>(action_array.Data());
    std::size_t action_count = action_array.size;
    const std::vector<mjtNum> prev_action = prev_action_vector_;

    bool invalid_action = false;
    for (std::size_t idx = 0; idx < action_count; ++idx) {
      if (!std::isfinite(static_cast<double>(act[idx]))) {
        act[idx] = 0.0;
        invalid_action = true;
      }
    }
    if (invalid_action && env_id_ == 0) {
      std::cerr << "[HumanoidEnv] Non-finite action detected; replaced with zeros." << std::endl;
    }
    last_action_vector_.assign(act, act + action_count);
    ParseActionToModules(act, action_count);

    mjtNum ctrl_cost = 0.0;

    if (random_force_enabled_) {
      if (random_force_steps_remaining_ <= 0) {
        random_force_cached_ =
            SampleRandomForce(random_force_min_, random_force_max_);
        random_force_steps_remaining_ =
            std::max(1, random_force_hold_steps_);
      }
      const mjtNum dt =
          static_cast<mjtNum>(static_cast<double>(module_manager_.getStepPeriod()) / 1e6) *
          static_cast<mjtNum>(frame_skip_);
      const mjtNum tau =
          std::max(static_cast<mjtNum>(1e-4), random_force_filter_tau_);
      const mjtNum alpha =
          static_cast<mjtNum>(1.0) - std::exp(-dt / tau);
      filtered_random_force_ =
          filtered_random_force_ +
          alpha * (random_force_cached_ - filtered_random_force_);
      applyForce(filtered_random_force_);
      --random_force_steps_remaining_;
    } else if (sim_driver_) {
      sim_driver_->clearExternalWrench();
    }

    for (int i = 0; i < frame_skip_; ++i) {
      module_manager_.stepOnce();
      CacheModuleState();
      ctrl_cost += ComputeControlCost();
    }

    // Compute contact cost.
    mjtNum contact_cost = 0.0;
    if (use_contact_force_) {
      contact_cost = 0.0;
    }

    // if(elapsed_step_ % frame_skip_ * 15 == 0){
    //   static std::random_device rd;
    //   static std::mt19937 gen(rd());
    //   // normal distribution with 0 mean and 0.003 stddev
    //   std::normal_distribution<mjtNum> dis( 0.00, 0.003);
    //   desired_h += dis(gen);
    //   //clamp desired_h to [0.25, 0.4]
    //   desired_h = std::max(0.32, std::min(0.4, desired_h));
    // }
    UpdateRewardReferences();

    const Eigen::Matrix<mjtNum, 3, 1> base_lin_vel(
        static_cast<mjtNum>(last_state_est_.vWorld[0]),
        static_cast<mjtNum>(last_state_est_.vWorld[1]),
        static_cast<mjtNum>(last_state_est_.vWorld[2]));
    const Eigen::Matrix<mjtNum, 3, 1> base_pos(
        static_cast<mjtNum>(last_state_est_.position[0]),
        static_cast<mjtNum>(last_state_est_.position[1]),
        static_cast<mjtNum>(last_state_est_.position[2]));
    const Eigen::Quaternion<mjtNum> base_quat(
        static_cast<mjtNum>(last_state_est_.orientation[0]),
        static_cast<mjtNum>(last_state_est_.orientation[1]),
        static_cast<mjtNum>(last_state_est_.orientation[2]),
        static_cast<mjtNum>(last_state_est_.orientation[3]));
    const Eigen::Matrix<mjtNum, 3, 1> base_lin_acc(
        static_cast<mjtNum>(last_state_est_.aWorld[0]),
        static_cast<mjtNum>(last_state_est_.aWorld[1]),
        static_cast<mjtNum>(last_state_est_.aWorld[2]));
    const Eigen::Matrix<mjtNum, 3, 1> base_ang_vel(
        static_cast<mjtNum>(last_state_est_.omegaBody[0]),
        static_cast<mjtNum>(last_state_est_.omegaBody[1]),
        static_cast<mjtNum>(last_state_est_.omegaBody[2]));

    const auto reward_result = locomotion_reward_.Compute(
        base_lin_vel, base_pos, base_quat, base_lin_acc, base_ang_vel,
        last_action_vector_, prev_action);

    bool is_healthy = IsHealthy();
    mjtNum reward = is_healthy ? reward_result.total : static_cast<mjtNum>(-10.0);

    if (!std::isfinite(static_cast<double>(reward))) {
      if (env_id_ == 0) {
        std::cerr << "[HumanoidEnv] Non-finite reward detected; forcing termination." << std::endl;
      }
      done_ = true;
      last_done_reason_ = "nonfinite_reward";
      last_is_healthy_ = false;
      WriteState(-5.0, base_lin_vel[0], base_lin_vel[1], ctrl_cost,
                 contact_cost, base_pos[0], base_pos[1], 0.0);
      return;
    }

    last_penalties_ = reward_result.penalties;
    last_smoothed_penalties_ = reward_result.smoothed_penalties;
    last_term_rewards_ = reward_result.rewards;
    last_ctrl_cost_ = ctrl_cost;
    last_contact_cost_ = contact_cost;
    last_healthy_reward_ = is_healthy ? healthy_reward_ : 0.0;
    last_x_velocity_ = base_lin_vel[0];
    last_y_velocity_ = base_lin_vel[1];
    last_x_position_ = base_pos[0];
    last_y_position_ = base_pos[1];
    last_is_healthy_ = is_healthy;
    lastReward = static_cast<float>(reward);
    prev_action_vector_ = last_action_vector_;

    ++elapsed_step_;
    const bool hit_unhealthy = terminate_when_unhealthy_ ? !is_healthy : false;
    const bool hit_timeout = (elapsed_step_ >= max_episode_steps_);
    done_ = done_ || hit_unhealthy || hit_timeout;
    if (done_) {
      if (!std::isfinite(static_cast<double>(reward))) {
        last_done_reason_ = "nonfinite_reward";
      } else if (hit_unhealthy) {
        last_done_reason_ = "unhealthy";
      } else if (hit_timeout) {
        last_done_reason_ = "timeout";
      } else {
        last_done_reason_ = "done";
      }
    }
    WriteState(reward, base_lin_vel[0], base_lin_vel[1], ctrl_cost,
               contact_cost, base_pos[0], base_pos[1], last_healthy_reward_);
  }

  void applyForce(const Eigen::Matrix<mjtNum, 6, 1>& force_and_torque) {
    Eigen::Matrix<mjtNum, 6, 1> sanitized = force_and_torque;
    bool invalid_input = false;
    for (int i = 0; i < 6; ++i) {
      if (!std::isfinite(static_cast<double>(sanitized[i]))) {
        sanitized[i] = 0.0;
        invalid_input = true;
      }
    }
    if (invalid_input && env_id_ == 0) {
      std::cerr << "[HumanoidEnv] Non-finite wrench provided to applyForce; "
                   "replacing with zeros."
                << std::endl;
    }
    if (sim_driver_) {
      sim_driver_->setExternalWrench(sanitized.data());
    }
    last_applied_wrench_ = sanitized;
  }

  Eigen::Matrix<mjtNum, 6, 1> SampleRandomForce(
      mjtNum min_magnitude = static_cast<mjtNum>(20.0),
      mjtNum max_magnitude = static_cast<mjtNum>(120.0)) {
    if (max_magnitude < min_magnitude) {
      std::swap(max_magnitude, min_magnitude);
    }
    min_magnitude = std::max(static_cast<mjtNum>(0.0), min_magnitude);
    max_magnitude = std::max(static_cast<mjtNum>(0.0), max_magnitude);
    max_magnitude = std::max(max_magnitude, min_magnitude);

    // Uniform direction on the unit sphere using normalized Gaussian noise.
    std::normal_distribution<mjtNum> normal_dist(
        static_cast<mjtNum>(0.0), static_cast<mjtNum>(1.0));
    Eigen::Matrix<mjtNum, 3, 1> direction;
    for (int attempt = 0; attempt < 8; ++attempt) {
      direction << normal_dist(cmd_rng_), normal_dist(cmd_rng_),
          normal_dist(cmd_rng_);
      if (!std::isfinite(direction[0]) || !std::isfinite(direction[1]) ||
          !std::isfinite(direction[2])) {
        continue;
      }
      const mjtNum n = direction.norm();
      if (n > std::numeric_limits<mjtNum>::epsilon()) {
        direction /= n;
        break;
      }
      if (attempt == 7) {
        direction = Eigen::Matrix<mjtNum, 3, 1>::UnitX();
      }
    }

    std::uniform_real_distribution<mjtNum> magnitude_dist(min_magnitude,
                                                          max_magnitude);
    const mjtNum magnitude = magnitude_dist(cmd_rng_);
    Eigen::Matrix<mjtNum, 6, 1> wrench;
    wrench << direction * magnitude, Eigen::Matrix<mjtNum, 3, 1>::Zero();
    return wrench;
  }

  void applyRandomForce(
      mjtNum min_magnitude = static_cast<mjtNum>(20.0),
      mjtNum max_magnitude = static_cast<mjtNum>(120.0)) {
    random_force_cached_ = SampleRandomForce(min_magnitude, max_magnitude);
    applyForce(random_force_cached_);
  }

 private:
  void InitializeQuadcontrol() {
    if (disabled_) return;

    module_manager_.clearConfig();
    if (!sim_config_path_.empty()) {
      if (!module_manager_.appendConfigFile(sim_config_path_.c_str()) &&
          env_id_ == 0) {
        std::cerr << "[HumanoidEnv] Failed to load sim config: "
                  << sim_config_path_ << std::endl;
      }
    }

    {
      std::ostringstream cfg;
      cfg << "[simulation]\n";
      if (sim_config_path_.empty()) {
        if (!model_xml_path_.empty()) {
          cfg << "model_path = \"" << model_xml_path_ << "\"\n";
        }
        if (!urdf_path_.empty()) {
          cfg << "urdf_path = \"" << urdf_path_ << "\"\n";
        }
      }
      const bool allow_render = (env_id_ == render_env_id_);
      const bool headless =
          allow_render ? (!render_mode_ && video_record_ <= 0) : true;
      cfg << "headless = " << (headless ? "true" : "false") << "\n";
      cfg << "realtime = false\n";
      if (!allow_render) {
        cfg << "video_record = false\n";
        cfg << "video_path = \"\"\n";
        cfg << "video_width = 0\n";
        cfg << "video_height = 0\n";
        cfg << "video_fps = 0.0\n";
      } else {
        if (video_record_ >= 0) {
          cfg << "video_record = " << (video_record_ ? "true" : "false") << "\n";
        }
        if (!video_path_.empty()) {
          cfg << "video_path = \"" << video_path_ << "\"\n";
        }
        if (video_width_ > 0) {
          cfg << "video_width = " << video_width_ << "\n";
        }
        if (video_height_ > 0) {
          cfg << "video_height = " << video_height_ << "\n";
        }
        if (video_fps_ > 0.0) {
          cfg << "video_fps = " << video_fps_ << "\n";
        }
      }
      module_manager_.appendConfigString(cfg.str().c_str());
    }

    if (!module_manager_.finalizeConfig() && env_id_ == 0) {
      std::cerr << "[HumanoidEnv] Failed to finalize config." << std::endl;
    }

    initHardware(&module_manager_, false);
    sim_hw_ctx_ = getSimHWContext(&module_manager_);
    if (!sim_hw_ctx_ || !sim_hw_ctx_->simdriver) {
      std::cerr << "[HumanoidEnv] SimHW context not available." << std::endl;
      cleanupHardware(&module_manager_);
      disabled_ = true;
      done_ = true;
      return;
    }
    sim_driver_ = sim_hw_ctx_->simdriver;
    motor_hw_ = sim_hw_ctx_->motorhw;
    imu_hw_ = sim_hw_ctx_->imuhw;
    clock_hw_ = sim_hw_ctx_->clockhw;
    if (!motor_hw_ || !imu_hw_ || !clock_hw_) {
      std::cerr << "[HumanoidEnv] SimHW hardware instances missing." << std::endl;
      cleanupHardware(&module_manager_);
      disabled_ = true;
      done_ = true;
      return;
    }
    HardwareThreadGuard hw_guard(motor_hw_, imu_hw_, clock_hw_);

    const rtcore::CLOCK sim_period = sim_driver_->simStepPeriod();
    if (sim_period > 0) {
      module_manager_.setStepPeriod(sim_period);
    }
    sim_driver_->setPeriod(module_manager_.getStepPeriod());

    static constexpr int kOrderControlParams = 100;
    static constexpr int kOrderGaitScheduler = 200;
    static constexpr int kOrderFootstepPlanner = 210;
    static constexpr int kOrderRlState = 220;
    static constexpr int kOrderWBIC = 300;
    static constexpr int kOrderLegController = 400;
    static constexpr int kOrderStateEstimator = 32000;

    control_params_ = new quadruped::MdlControlParams();
    module_manager_.addModule(control_params_, 1, 0, kOrderControlParams);
    module_manager_.activateModule(control_params_);

    ct_gait_scheduler_ = new quadruped::MdlCtGaitScheduler();
    module_manager_.addModule(ct_gait_scheduler_, 1, 0, kOrderGaitScheduler);
    module_manager_.activateModule(ct_gait_scheduler_);

    footstep_planner_ = new quadruped::MdlFootstepPlanner();
    module_manager_.addModule(footstep_planner_, 1, 0, kOrderFootstepPlanner);
    module_manager_.activateModule(footstep_planner_);

    rl_state_module_ = new quadruped::MdlRLLocomotionState();
    module_manager_.addModule(rl_state_module_, 1, 0, kOrderRlState);
    module_manager_.activateModule(rl_state_module_);

    wbic_ = new quadruped::MdlWBIC();
    module_manager_.addModule(wbic_, 1, 0, kOrderWBIC);
    module_manager_.activateModule(wbic_);

    leg_controller_module_ = new quadruped::MdlLegController();
    module_manager_.addModule(leg_controller_module_, 1, 0,
                              kOrderLegController);
    module_manager_.activateModule(leg_controller_module_);

    state_estimator_ = new quadruped::MdlStateEstimator();
    module_manager_.addModule(state_estimator_, 1, 0, kOrderStateEstimator);
    module_manager_.activateModule(state_estimator_);
    if (control_params_) {
      control_params_->setControlMode(quadruped::ControlMode::kRlLocomotion);
    }
  }

  void ResetModules() {
    if (!control_params_ || !ct_gait_scheduler_ || !footstep_planner_ ||
        !rl_state_module_ || !wbic_ || !leg_controller_module_ ||
        !state_estimator_) {
      return;
    }
    HardwareThreadGuard hw_guard(motor_hw_, imu_hw_, clock_hw_);
    module_manager_.deactivateModule(control_params_);
    module_manager_.deactivateModule(ct_gait_scheduler_);
    module_manager_.deactivateModule(footstep_planner_);
    module_manager_.deactivateModule(rl_state_module_);
    module_manager_.deactivateModule(wbic_);
    module_manager_.deactivateModule(leg_controller_module_);
    module_manager_.deactivateModule(state_estimator_);

    module_manager_.activateModule(control_params_);
    module_manager_.activateModule(ct_gait_scheduler_);
    module_manager_.activateModule(footstep_planner_);
    module_manager_.activateModule(rl_state_module_);
    module_manager_.activateModule(wbic_);
    module_manager_.activateModule(leg_controller_module_);
    module_manager_.activateModule(state_estimator_);
    control_params_->setControlMode(quadruped::ControlMode::kRlLocomotion);
    rl_state_module_->resetState();
  }

  void ResetSimulation() {
    if (!sim_driver_) return;
    sim_driver_->resetSimulation();

    rtcore::ConfigTable simConfig;
    bool has_sim = module_manager_.getConfigTable("simulation", simConfig);

    double base_x = 0.0;
    double base_y = 0.0;
    double base_z = desired_h;
    if (has_sim) {
      rtcore::ConfigArray posArray;
      if (simConfig.getArray("initial_position", posArray) &&
          posArray.size() >= 3) {
        base_x = posArray.getDoubleAt(0, base_x);
        base_y = posArray.getDoubleAt(1, base_y);
        base_z = posArray.getDoubleAt(2, base_z);
      }
    }
    const double min_z = static_cast<double>(healthy_z_min_) + 1e-3;
    const double max_z = static_cast<double>(healthy_z_max_) - 1e-3;
    if (min_z < max_z) {
      base_z = std::clamp(base_z, min_z, max_z);
    }
    sim_driver_->setBasePosition(static_cast<mjtNum>(base_x),
                                 static_cast<mjtNum>(base_y),
                                 static_cast<mjtNum>(base_z));

    double abad = 10.0;
    double hip = -80.0;
    double knee = 130.0;
    if (has_sim) {
      rtcore::ConfigArray legArray;
      if (simConfig.getArray("initial_leg_joint_deg", legArray) &&
          legArray.size() >= 3) {
        abad = legArray.getDoubleAt(0, abad);
        hip = legArray.getDoubleAt(1, hip);
        knee = legArray.getDoubleAt(2, knee);
      }
    }
    sim_driver_->setLegJointAnglesDeg(abad, hip, knee);
  }

  void CacheModuleState() {
    if (ct_gait_scheduler_ && ct_gait_scheduler_->hasSchedule()) {
      last_gait_schedule_ = ct_gait_scheduler_->gaitSchedule();
    }

    if (state_estimator_) {
      last_state_est_ = state_estimator_->getCheetahStateEstimate();
    }
  }

  void ParseActionToModules(const mjtNum* act, std::size_t action_count) {

    auto map_to_range = [](mjtNum input, mjtNum in_min, mjtNum in_max,
                           mjtNum out_min, mjtNum out_max) -> mjtNum {
      return (input - in_min) * (out_max - out_min) / (in_max - in_min) +
             out_min;
    };

    mjtNum residual_vx = 0.0;
    mjtNum residual_vy = 0.0;
    mjtNum residual_yaw = 0.0;
    if (action_count >= 3) {
      residual_vx = map_to_range(act[0], -1.0, 1.0,
                                 -cmd_residual_linear_limit_,
                                 cmd_residual_linear_limit_);
      residual_vy = map_to_range(act[1], -1.0, 1.0,
                                 -cmd_residual_linear_limit_,
                                 cmd_residual_linear_limit_);
      residual_yaw = map_to_range(act[2], -1.0, 1.0,
                                  -cmd_residual_yaw_limit_,
                                  cmd_residual_yaw_limit_);
    }

    quadruped::RlDesiredVelocity desired{};
    desired.body_velocity[0] = static_cast<float>(cmd_vel_target_body_[0] +
                                                  residual_vx);
    desired.body_velocity[1] = static_cast<float>(cmd_vel_target_body_[1] +
                                                  residual_vy);
    desired.body_velocity[2] = static_cast<float>(cmd_vel_target_body_[2] +
                                                  residual_yaw);
    if (footstep_planner_) {
      footstep_planner_->setDesiredVelocity(desired);
    }
    if (rl_state_module_) {
      rl_state_module_->setDesiredVelocity(desired);
    }

    quadruped::FootForceTargets forces{};
    forces.reset();
    quadruped::FootstepResiduals residuals{};
    residuals.reset();
    for (int leg = 0; leg < RLConstants::kNumLegs; ++leg) {
      const int base = 3 + leg * 5;
      if (action_count > static_cast<std::size_t>(base + 2)) {
        forces.forces[leg][0] = static_cast<float>(
            map_to_range(act[base + 0], -1.0, 1.0, -50.0, 50.0));
        forces.forces[leg][1] = static_cast<float>(
            map_to_range(act[base + 1], -1.0, 1.0, -50.0, 50.0));
        forces.forces[leg][2] = static_cast<float>(
            map_to_range(act[base + 2], -1.0, 1.0, 0.0, 250.0));
      }
      if (action_count > static_cast<std::size_t>(base + 4)) {
        residuals.offsets[leg][0] = static_cast<float>(act[base + 3] * 0.50);
        residuals.offsets[leg][1] = static_cast<float>(act[base + 4] * 0.50);
        residuals.offsets[leg][2] = 0.0f;
      }
    }
    if (rl_state_module_) {
      rl_state_module_->setFootForces(forces);
    }
    if (footstep_planner_) {
      footstep_planner_->setFootstepResiduals(residuals);
    }

    mjtNum delta_theta = 0.0;
    if (action_count > RLConstants::kPhaseDeltaIdx) {
      delta_theta = map_to_range(act[RLConstants::kPhaseDeltaIdx], -1.0, 1.0,
                                 0.0,
                                 quadruped::MdlCtGaitScheduler::kMaxPhaseDelta);
    }
    if (ct_gait_scheduler_) {
      ct_gait_scheduler_->setPhaseDelta(static_cast<float>(delta_theta));
    }
    if (control_params_) {
      control_params_->setControlMode(quadruped::ControlMode::kRlLocomotion);
    }
  }

  mjtNum ComputeControlCost() {
    if (!motor_hw_) return static_cast<mjtNum>(0.0);
    mjtNum ctrl_cost = 0.0;
    MotorHW::cmd_t cmd{};
    for (int axis = 0; axis < 12; ++axis) {
      motor_hw_->getCommand(axis, cmd);
      const mjtNum tau = static_cast<mjtNum>(cmd.tau);
      if (std::isfinite(static_cast<double>(tau))) {
        ctrl_cost += ctrl_cost_weight_ * tau * tau;
      }
    }
    return ctrl_cost;
  }

  void FillObservation(mjtNum* obs, std::size_t obs_count) {
    if (!obs || obs_count == 0) {
      return;
    }

    mjtNum* ptr = obs;
    const mjtNum* obs_end = obs + obs_count;

    auto write_value = [&](mjtNum value) {
      if (ptr < obs_end) {
        *ptr++ = value;
      }
    };

    write_value(cmd_vel_target_body_[0]);

    for (int i = 0; i < 3; ++i) {
      write_value(static_cast<mjtNum>(last_state_est_.vBody[i]));
    }
    for (int i = 0; i < 3; ++i) {
      write_value(static_cast<mjtNum>(last_state_est_.omegaBody[i]));
    }

    for (int i = 0; i < 3; ++i) {
      write_value(static_cast<mjtNum>(last_state_est_.rpy[i]));
    }

    auto* leg_controller =
        leg_controller_module_ ? leg_controller_module_->legController()
                               : nullptr;
    if (leg_controller) {
      for (int leg = 0; leg < RLConstants::kNumLegs; ++leg) {
        const auto& data = leg_controller->datas[leg];
        for (int joint = 0; joint < 3; ++joint) {
          write_value(static_cast<mjtNum>(data.q[joint]));
        }
      }
      for (int leg = 0; leg < RLConstants::kNumLegs; ++leg) {
        const auto& data = leg_controller->datas[leg];
        for (int joint = 0; joint < 3; ++joint) {
          write_value(static_cast<mjtNum>(data.qd[joint]));
        }
      }
    } else {
      for (int i = 0; i < RLConstants::kNumLegs * 6; ++i) {
        write_value(static_cast<mjtNum>(0.0));
      }
    }
    for (int leg = 0; leg < RLConstants::kNumLegs; ++leg) {
      const mjtNum contact =
          (last_gait_schedule_.contact_state[leg] > 0.5f)
              ? static_cast<mjtNum>(1.0)
              : static_cast<mjtNum>(0.0);
      write_value(contact);
    }

    std::array<std::array<mjtNum, 3>, RLConstants::kNumLegs> foot_pos_body{};
    if (leg_controller) {
      for (int leg = 0; leg < RLConstants::kNumLegs; ++leg) {
        const auto& data = leg_controller->datas[leg];
        Eigen::Matrix<mjtNum, 3, 1> hip = Eigen::Matrix<mjtNum, 3, 1>::Zero();
        if (data.quadruped) {
          const auto hip_loc = data.quadruped->getHipLocation(leg);
          hip[0] = hip_loc[0];
          hip[1] = hip_loc[1];
          hip[2] = hip_loc[2];
        }
        foot_pos_body[leg][0] = hip[0] + static_cast<mjtNum>(data.p[0]);
        foot_pos_body[leg][1] = hip[1] + static_cast<mjtNum>(data.p[1]);
        foot_pos_body[leg][2] = hip[2] + static_cast<mjtNum>(data.p[2]);
      }
    } else {
      for (int leg = 0; leg < RLConstants::kNumLegs; ++leg) {
        foot_pos_body[leg].fill(static_cast<mjtNum>(0.0));
      }
    }
    for (int axis = 0; axis < 3; ++axis) {
      write_value(foot_pos_body[0][axis]);
    }
    for (int axis = 0; axis < 3; ++axis) {
      write_value(foot_pos_body[1][axis]);
    }

    mjtNum phi = static_cast<mjtNum>(0.0);
    for (int leg = 0; leg < RLConstants::kNumLegs; ++leg) {
      phi += static_cast<mjtNum>(last_gait_schedule_.swing_phase[leg]);
    }
    phi /= static_cast<mjtNum>(RLConstants::kNumLegs);
    phi = std::clamp(phi, static_cast<mjtNum>(0.0), static_cast<mjtNum>(1.0));
    constexpr mjtNum kTwoPi =
        static_cast<mjtNum>(6.28318530717958647692);
    write_value(std::sin(kTwoPi * phi));
    write_value(std::cos(kTwoPi * phi));
  }

  bool IsHealthy() {
    const mjtNum z = static_cast<mjtNum>(last_state_est_.position[2]);
    bool healthy = (healthy_z_min_ < z) && (z < healthy_z_max_);
    if (!healthy && render_mode_) {
      std::cout << "[IsHealthy] Unhealthy state detected: "
                << "z position = " << z
                << ", healthy_z_min = " << healthy_z_min_
                << ", healthy_z_max = " << healthy_z_max_ << std::endl;
    }
    return healthy;
  }

  mjtNum ComputeOrientationPenalty() const {
    Eigen::Quaternion<mjtNum> q(
        static_cast<mjtNum>(last_state_est_.orientation[0]),
        static_cast<mjtNum>(last_state_est_.orientation[1]),
        static_cast<mjtNum>(last_state_est_.orientation[2]),
        static_cast<mjtNum>(last_state_est_.orientation[3]));
    Eigen::Vector3<mjtNum> eul = q.toRotationMatrix().eulerAngles(0, 1, 2);
    mjtNum roll = eul[0];
    mjtNum pitch = eul[1];
    return orientation_penalty_weight_ * (roll * roll + pitch * pitch);
  }

  mjtNum ComputeHeightPenalty() const {
    const mjtNum height = static_cast<mjtNum>(last_state_est_.position[2]);
    const mjtNum desired = static_cast<mjtNum>(desired_h);
    const mjtNum err = height - desired;
    return height_penalty_weight_ * err * err;
  }

  mjtNum ComputeFootSlipPenalty() {
    mjtNum slip_cost = 0.0;
    auto* leg_ctrl =
        leg_controller_module_ ? leg_controller_module_->legController()
                               : nullptr;
    if (!leg_ctrl) {
      return 0.0;
    }
    for (int leg = 0; leg < 4; ++leg) {
      if (last_gait_schedule_.contact_state[leg] > 0.5f) {
        const auto& leg_data = leg_ctrl->datas[leg];
        const mjtNum vx = static_cast<mjtNum>(leg_data.v[0]);
        const mjtNum vy = static_cast<mjtNum>(leg_data.v[1]);
        const mjtNum vz = static_cast<mjtNum>(leg_data.v[2]);
        slip_cost += vx * vx + vy * vy + vz * vz;
      }
    }
    return foot_slip_penalty_weight_ * slip_cost;
  }

  mjtNum SampleCommandValue(mjtNum min_value, mjtNum max_value) {
    if (min_value > max_value) {
      std::swap(min_value, max_value);
    }
    std::uniform_real_distribution<mjtNum> dist(min_value, max_value);
    return dist(cmd_rng_);
  }

  void SetCommandVelocityTarget(const std::array<mjtNum, 3>& cmd) {
    cmd_vel_target_body_ = cmd;
  }

  void ResampleCommandVelocity() {
    std::array<mjtNum, 3> new_cmd{};
    if (cmd_profile_mode_ == "fixed") {
      new_cmd = cmd_fixed_cmd_vel_;
    } else {
      new_cmd[0] = SampleCommandValue(cmd_rand_vx_min_, cmd_rand_vx_max_);
      new_cmd[1] = SampleCommandValue(cmd_rand_vy_min_, cmd_rand_vy_max_);
      new_cmd[2] = SampleCommandValue(cmd_rand_yaw_min_, cmd_rand_yaw_max_);
    }
    SetCommandVelocityTarget(new_cmd);
  }

  void UpdateRewardReferences() {
    locomotion_reward_config_.refs.xdot_ref = cmd_vel_target_body_[0];
    locomotion_reward_config_.refs.zdot_ref = static_cast<mjtNum>(0.0);
    locomotion_reward_config_.refs.z_ref =
        static_cast<mjtNum>(desired_h);
    locomotion_reward_.SetReferences(locomotion_reward_config_.refs);
  }

  // ------------------------------------------------------------------------

  void WriteState(float reward, mjtNum xv, mjtNum yv, mjtNum ctrl_cost,
                  mjtNum contact_cost, mjtNum x_after, mjtNum y_after,
                  mjtNum healthy_reward) {
    State state = Allocate();
    state["reward"_] = reward;
    // Debug the state allocation - check if it's properly created
    // std::cout << "State obs buffer size: "
    // << state["obs"_].Shape()[0]  << state["obs"_].Shape()[1] << std::endl;
    auto obs_array = state["obs"_];
    auto* obs = static_cast<mjtNum*>(obs_array.Data());
    mjtNum* obs_start = obs;  // Save the starting pointer for debugging

    FillObservation(obs, obs_array.size);
    bool invalid_obs = false;
    for (std::size_t idx = 0; idx < obs_array.size; ++idx) {
      if (!std::isfinite(static_cast<double>(obs[idx]))) {
        obs[idx] = 0.0;
        invalid_obs = true;
      }
    }
    if (invalid_obs && env_id_ == 0) {
      std::cerr << "[HumanoidEnv] Non-finite observation detected; replaced with zeros." << std::endl;
    }
    if (obs_array.size > 0) {
      last_observation_.assign(obs_start, obs_start + obs_array.size);
    } else {
      last_observation_.clear();
    }
    obs_array.Assign(obs, obs_array.size);

    // Print confirmation
    // std::cout << "Filled " << 42 << " elements in observation array" << std::endl;
    // std::cout << "First few values: "
    //           << obs[0] << " "
    //           << obs[1] << " "
    //           << obs[2] << " "
    //           << std::endl;
    //   // Print the first 30 observation values using the saved starting pointer
    //   std::cout << "obs values: "; for(int i=0; i<30; i++) {
    //   std::cout << obs_start[i] << " ";
    //   }
    //   std::cout << std::endl;

    // Add this to WriteState() after filling all observations
    // std::cout << "Filled " << static_cast<int>(obs_end - obs_start) << " elements in observation
    // array" << std::endl;
    state["info:reward_linvel"_] = xv * forward_reward_weight_;
    state["info:reward_quadctrl"_] = -ctrl_cost;
    state["info:reward_alive"_] = healthy_reward;
    state["info:reward_impact"_] = -contact_cost;
    state["info:x_position"_] = x_after;
    state["info:y_position"_] = y_after;
    state["info:distance_from_origin"_] =
        std::sqrt(x_after * x_after + y_after * y_after);
    state["info:x_velocity"_] = xv;
    state["info:y_velocity"_] = yv;

    lastReward = reward;
    writeDataToCSV();

  }
  void writeDataToCSV(int mode = 0) {
    if (csv_logging_enabled_ == 0) {
      return;  // Skip writing if logging is disabled
    }
    if (!outputFile.is_open()) {
      std::cerr << "Error: Unable to open CSV file for writing." << std::endl;
      return;
    }

    if (mode == 1 || mode == 2) {
      pending_reset_marker_ = true;
      return;
    }
    if (mode != 0) {
      return;
    }

    if (last_observation_.empty() || last_action_vector_.empty()) {
      return;  // Skip until we have both action and observation data
    }

    if (outputFile.tellp() == 0) {
      outputFile << "elapsed_step,reset_marker,body_x,body_y,body_z,"
                 << "quat_w,quat_x,quat_y,quat_z,"
                 << "FR_abad,FR_hip,FR_knee,"
                 << "FL_abad,FL_hip,FL_knee,"
                 << "HR_abad,HR_hip,HR_knee,"
                 << "HL_abad,HL_hip,HL_knee,"
                 << "reward_total,healthy_reward,"
                 << "penalty_base_xvel,penalty_base_zvel,penalty_base_zpos,"
                 << "penalty_base_orientation,penalty_base_straight,"
                 << "penalty_base_linear_accel,penalty_base_angular_vel,penalty_action_smooth,penalty_phase_delta,"
                 << "ctrl_cost,contact_cost,is_healthy,"
                 << "x_position,y_position,x_velocity,y_velocity,"
                 << "pBody_des_z,desired_h,cmd_vel_x,cmd_vel_y,cmd_vel_yaw,"
                 << "applied_fx,applied_fy,applied_fz,applied_tx,applied_ty,applied_tz,"
                 << "reward_base_xvel,reward_base_zvel,reward_base_zpos,reward_base_orientation,"
                 << "reward_base_straight,reward_base_linear_accel,reward_base_angular_vel,reward_action_smooth,reward_phase_delta";
      for (std::size_t i = 0; i < last_action_vector_.size(); ++i) {
        outputFile << ",action_" << i;
      }
      for (std::size_t i = 0; i < last_observation_.size(); ++i) {
        outputFile << ",obs_" << i;
      }
      outputFile << std::endl;
    }

    const bool reset_marker = pending_reset_marker_;
    pending_reset_marker_ = false;
    mjtNum body_des_z = desired_h;
    if (wbic_ && wbic_->wbcData()) {
      body_des_z = static_cast<mjtNum>(wbic_->wbcData()->pBody_des[2]);
    }

    outputFile << elapsed_step_ << ',' << (reset_marker ? 1 : 0);
    outputFile << ',' << static_cast<double>(last_state_est_.position[0])
               << ',' << static_cast<double>(last_state_est_.position[1])
               << ',' << static_cast<double>(last_state_est_.position[2])
               << ',' << static_cast<double>(last_state_est_.orientation[0])
               << ',' << static_cast<double>(last_state_est_.orientation[1])
               << ',' << static_cast<double>(last_state_est_.orientation[2])
               << ',' << static_cast<double>(last_state_est_.orientation[3]);
    auto* leg_controller =
        leg_controller_module_ ? leg_controller_module_->legController()
                               : nullptr;
    for (int leg = 0; leg < 4; ++leg) {
      for (int joint = 0; joint < 3; ++joint) {
        double q = 0.0;
        if (leg_controller) {
          q = static_cast<double>(leg_controller->datas[leg].q[joint]);
        }
        outputFile << ',' << q;
      }
    }
    outputFile << ',' << static_cast<double>(lastReward)
               << ',' << static_cast<double>(last_healthy_reward_)
               << ',' << static_cast<double>(last_smoothed_penalties_[LocomotionReward::kBaseXVel])
               << ',' << static_cast<double>(last_smoothed_penalties_[LocomotionReward::kBaseZVel])
               << ',' << static_cast<double>(last_smoothed_penalties_[LocomotionReward::kBaseZPos])
               << ',' << static_cast<double>(last_smoothed_penalties_[LocomotionReward::kBaseOrientation])
               << ',' << static_cast<double>(last_smoothed_penalties_[LocomotionReward::kBaseStraight])
               << ',' << static_cast<double>(last_smoothed_penalties_[LocomotionReward::kBaseLinearAccel])
               << ',' << static_cast<double>(last_smoothed_penalties_[LocomotionReward::kBaseAngularVel])
               << ',' << static_cast<double>(last_smoothed_penalties_[LocomotionReward::kActionSmooth])
               << ',' << static_cast<double>(last_smoothed_penalties_[LocomotionReward::kPhaseDelta])
               << ',' << static_cast<double>(last_ctrl_cost_)
               << ',' << static_cast<double>(last_contact_cost_)
               << ',' << (last_is_healthy_ ? 1 : 0)
               << ',' << static_cast<double>(last_x_position_)
               << ',' << static_cast<double>(last_y_position_)
               << ',' << static_cast<double>(last_x_velocity_)
               << ',' << static_cast<double>(last_y_velocity_)
               << ','
               << static_cast<double>(body_des_z)
               << ',' << static_cast<double>(desired_h)
               << ',' << static_cast<double>(cmd_vel_target_body_[0])
               << ',' << static_cast<double>(cmd_vel_target_body_[1])
               << ',' << static_cast<double>(cmd_vel_target_body_[2])
               << ',' << static_cast<double>(last_applied_wrench_[0])
               << ',' << static_cast<double>(last_applied_wrench_[1])
               << ',' << static_cast<double>(last_applied_wrench_[2])
               << ',' << static_cast<double>(last_applied_wrench_[3])
               << ',' << static_cast<double>(last_applied_wrench_[4])
               << ',' << static_cast<double>(last_applied_wrench_[5]);

    for (int i = 0; i < LocomotionReward::kNumTerms; ++i) {
      outputFile << ',' << static_cast<double>(last_term_rewards_[i]);
    }

    for (std::size_t i = 0; i < last_action_vector_.size(); ++i) {
      outputFile << ',' << static_cast<double>(last_action_vector_[i]);
    }
    for (std::size_t i = 0; i < last_observation_.size(); ++i) {
      outputFile << ',' << static_cast<double>(last_observation_[i]);
    }
    outputFile << std::endl;
  }
};

using HumanoidEnvPool = AsyncEnvPool<HumanoidEnv>;

}  // namespace mujoco_gym

#endif  // ENVPOOL_MUJOCO_GYM_HUMANOID_H_
