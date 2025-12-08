#ifndef ENVPOOL_MUJOCO_GYM_HUMANOID_H_
#define ENVPOOL_MUJOCO_GYM_HUMANOID_H_

#include <algorithm>
#include <array>
#include <cmath>  // For std::sqrt, std::abs
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "cppTypes.h"
#include <Dynamics/Quadruped.h>
#include <Dynamics/FloatingBaseModel.h>
#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/mujoco/gym/mujoco_env.h"

#include <eigen3/Eigen/Dense>
#include <RobotRunner.h>
#include <RobotController.h>
#include <Controllers/EmbeddedController.hpp>
#include <Utilities/RobotCommands.h>
#include "mbc_interface.h"

namespace mujoco_gym {

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
    if (action.size() <= ModelBasedControllerInterface::kPhaseDeltaIdx) {
      return static_cast<mjtNum>(0.0);
    }
    // Linear penalty: zero cost when delta_theta <= -1, increasing linearly above that.
    const mjtNum delta_theta = action[ModelBasedControllerInterface::kPhaseDeltaIdx];
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
        "csv_logging_enabled"_.Bind(false), 
        "random_force_enabled"_.Bind(true),
        "random_force_min"_.Bind(0.0),
        "random_force_max"_.Bind(50.0),
        "random_force_hold_steps"_.Bind(20),
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
                        Spec<mjtNum>({ModelBasedControllerInterface::kObservationDim},
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
        Spec<mjtNum>({-1, ModelBasedControllerInterface::kActionDim}, {-1, 1})));
  }
};

using HumanoidEnvSpec = EnvSpec<HumanoidEnvFns>;

class HumanoidEnv : public Env<HumanoidEnvSpec>, public MujocoEnv {
 protected:
  bool terminate_when_unhealthy_, no_pos_, use_contact_force_, render_mode_, csv_logging_enabled_;
  bool random_force_enabled_;
  mjtNum ctrl_cost_weight_, forward_reward_weight_, healthy_reward_;
  mjtNum healthy_z_min_, healthy_z_max_;
  mjtNum contact_cost_weight_, contact_cost_max_;
  std::uniform_real_distribution<> dist_;
  ModelBasedControllerInterface mbc;
  // New: Backup of the controller

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
  Eigen::Matrix<mjtNum, 6, 1> last_applied_wrench_{
      Eigen::Matrix<mjtNum, 6, 1>::Zero()};
  std::array<mjtNum, 3> cmd_vel_target_body_{{0.0, 0.0, 0.0}};
  std::mt19937 cmd_rng_;
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
  // Added: Torque bound constant (example value; adjust as needed)
  const mjtNum torque_limit_ = 65.0;
  // Added: CSV logging switch (set to 1 manually to enable CSV writing)
  std::string csv_filename_;
  std::string wbc_csv_filename_;
  int base_body_id_{-1};  // MuJoCo body index for the floating base
  void UpdateMbcDebugMarkers();

 public:
  HumanoidEnv(const Spec& spec, int env_id)
      : Env<HumanoidEnvSpec>(spec, env_id),
        // Construct MujocoEnv using a fixed XML file path.
        MujocoEnv(std::string("/app/envpool/envpool/mujoco/legged-sim/resource/"
                              "demir_1/scene.xml"),
                  spec.config["frame_skip"_], spec.config["post_constraint"_],
                  spec.config["max_episode_steps"_]),
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
        random_force_min_(spec.config["random_force_min"_]),
        random_force_max_(spec.config["random_force_max"_]),
        random_force_hold_steps_(spec.config["random_force_hold_steps"_]),
        dist_(-spec.config["reset_noise_scale"_],
              spec.config["reset_noise_scale"_]),
        mbc(),
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
        cmd_rng_(std::random_device{}() + env_id) {
    mbc.setModel(model_);
    base_body_id_ = mj_name2id(model_, mjOBJ_BODY, "demir/base_link_inertia");
    if (base_body_id_ < 0 && env_id_ == 0) {
      std::cerr << "[HumanoidEnv] Failed to find base body 'demir/base_link_inertia'." << std::endl;
    }
    mbc.setCommandResidualLimits(static_cast<float>(cmd_residual_linear_limit_),
                                 static_cast<float>(cmd_residual_yaw_limit_));
    if (env_id_ == 0 ) {
      csv_logging_enabled_ = true;
    }

    if (render_mode_)
      EnableRender(true);

    random_force_hold_steps_ = std::max(1, random_force_hold_steps_);

  csv_filename_ = "/app/envpool/data/current/" + std::to_string(env_id_) + "_log.csv";
  wbc_csv_filename_ = "/app/envpool/data/current/" + std::to_string(env_id_) + "_wbc.csv";
  ClearCsvLogs();

    MujocoReset();
    mbc.setModeLocomotion();
    if (csv_logging_enabled_) {
      mbc.writeWBCLogCSV(wbc_csv_filename_, /*write_header_if_new=*/true);
    }

    desired_h = 0.35;  // Set the desired height to 0.35
    mbc.setBaseHeight(static_cast<float>(desired_h));
    ResampleCommandVelocity();
    locomotion_reward_.Reset();
    UpdateRewardReferences();
    prev_action_vector_.clear();

    // Save the initial controller configuration to backup
    // while (mbc.getMode() != 6) {
    //   mjtNum dummy_action[22];
    //   // set all values to zero;
    //   for (int i = 0; i < 22; i++) {
    //     dummy_action[i] = 0;
    //   }
    //   mjtNum* act = dummy_action;
    //   mbc.setFeedback(data_);
    //   mbc.setAction(act, action_count);
    //   mbc.run();

    //   mjtNum motor_commands[12];
    //   std::array<double, 12> mc = mbc.getMotorCommands();
    //   for (int i = 0; i < 12; ++i) {
    //     // Clamp motor commands to torque bounds.
    //     motor_commands[i] =
    //         std::max(std::min(static_cast<mjtNum>(mc[i]), torque_limit_),
    //                  -torque_limit_);
    //   }
    //   const auto& before = GetMassCenter();

    //   MujocoStep(motor_commands);

    //   writeDataToCSV(0);
    // }
    writeDataToCSV(2);
    // model_backup_ = mj_copyModel(nullptr, model_);
    // data_backup_ = mj_copyData(nullptr, model_, data_);
  }

  void MujocoResetModel() override {
    setIC();


#ifdef ENVPOOL_TEST
    std::memcpy(qpos0_, data_->qpos, sizeof(mjtNum) * model_->nq);
    std::memcpy(qvel0_, data_->qvel, sizeof(mjtNum) * model_->nv);
#endif
  }

  bool IsDone() override { return done_; }

  void Reset() override {
    writeDataToCSV(1);
    last_action_vector_.clear();
    prev_action_vector_.clear();
    last_observation_.clear();
    random_force_steps_remaining_ = 0;
    random_force_cached_.setZero();
    last_applied_wrench_.setZero();
    last_penalties_.fill(static_cast<mjtNum>(0.0));
    last_smoothed_penalties_.fill(static_cast<mjtNum>(0.0));
    last_term_rewards_.fill(static_cast<mjtNum>(0.0));
    locomotion_reward_.Reset();
    mbc.clearObservationHistory();
    if(render_mode_)
      std::cout << "Resetting the environment..." << std::endl;
    done_ = false;
    elapsed_step_ = 0;
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<mjtNum> dis( 0.32, 0.4);
    desired_h = dis(gen);
    mbc.setBaseHeight(static_cast<float>(desired_h));
    // MujocoReset();
    // mj_deleteData(data_);
    // mj_deleteModel(model_);
    // model_ = mj_copyModel(nullptr, model_backup_);
    // data_ = mj_copyData(nullptr, model_, data_backup_);
    mj_resetData(model_, data_);
    setIC();
    WriteState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    mbc.setFeedback(data_);
    mbc.setModeLocomotion();
    mbc.setDesiredBodyHeight(static_cast<float>(desired_h));
    ResampleCommandVelocity();
    UpdateRewardReferences();
    if (csv_logging_enabled_) {
      mbc.writeWBCLogCSV(wbc_csv_filename_, /*write_header_if_new=*/true);
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
      outputFile.open(csv_filename_.c_str(), std::ios::out | std::ios::trunc);
      if (!outputFile.is_open()) {
        std::cerr << "Error: Unable to recreate CSV log file at " << csv_filename_ << std::endl;
      }
      pending_reset_marker_ = true;
    }

    mbc.clearWBCLogFile(wbc_csv_filename_);
    if (csv_logging_enabled_) {
      mbc.writeWBCLogCSV(wbc_csv_filename_, /*write_header_if_new=*/true);
    }
  }

  void Step(const Action& action) override {
    // step
    auto action_array = action["action"_];
    auto* act = static_cast<mjtNum*>(action_array.Data());
    std::size_t action_count = action_array.size;
    const std::vector<mjtNum> prev_action = prev_action_vector_;
    mbc.setCommandVelocity(cmd_vel_target_body_[0], cmd_vel_target_body_[1],
                           cmd_vel_target_body_[2]);
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
    // repeat frameskip times
    mjtNum ctrl_cost = 0.0;

    if (random_force_enabled_) {
      if (random_force_steps_remaining_ <= 0) {
        random_force_cached_ =
            SampleRandomForce(random_force_min_, random_force_max_);
        // Ensure at least one step of hold.
        random_force_steps_remaining_ =
            std::max(1, random_force_hold_steps_);
      }
      applyForce(random_force_cached_);
      --random_force_steps_remaining_;
      if (render_mode_ && base_body_id_ >= 0 &&
          base_body_id_ < model_->nbody) {
        const Eigen::Matrix<mjtNum, 3, 1> f = random_force_cached_.head<3>();
        const mjtNum norm = f.norm();
        if (norm > std::numeric_limits<mjtNum>::epsilon()) {
          const mjtNum length_scale = static_cast<mjtNum>(0.01);
          std::array<mjtNum, 3> dir{f[0] / norm, f[1] / norm, f[2] / norm};
          std::array<mjtNum, 3> pos{
              data_->xpos[3 * base_body_id_ + 0],
              data_->xpos[3 * base_body_id_ + 1],
              data_->xpos[3 * base_body_id_ + 2]};
          const mjtNum length = norm * length_scale;
          addArrow(pos, dir, length);
        }
      }
    }

    for (int i = 0; i < frame_skip_; ++i) {
      mbc.setAction(act, action_count);
      mbc.setFeedback(data_);
      mbc.run();

      mjtNum motor_commands[12];
      std::array<double, 12> mc = mbc.getMotorCommands();
      bool invalid_motor_command = false;
      for (int j = 0; j < 12; ++j) {
        mjtNum candidate = static_cast<mjtNum>(mc[j]);
        if (!std::isfinite(static_cast<double>(candidate))) {
          candidate = 0.0;
          invalid_motor_command = true;
        }
        candidate = std::clamp(candidate, -torque_limit_, torque_limit_);
        motor_commands[j] = candidate;
        ctrl_cost += ctrl_cost_weight_ * motor_commands[j] * motor_commands[j];
      }
      if (invalid_motor_command && env_id_ == 0) {
        std::cerr << "[HumanoidEnv] Non-finite motor command detected; clamped to zero." << std::endl;
      }
      if (render_mode_) {
        UpdateMbcDebugMarkers();
      }
      MujocoStep(motor_commands);
    }

    if (csv_logging_enabled_) {
      mbc.writeWBCLogCSV(wbc_csv_filename_, /*write_header_if_new=*/true);
    }

    // Compute contact cost.
    mjtNum contact_cost = 0.0;
    if (use_contact_force_) {
      for (int i = 0; i < 6 * model_->nbody; ++i) {
        mjtNum x = data_->cfrc_ext[i];
        contact_cost += contact_cost_weight_ * x * x;
      }
      contact_cost = std::min(contact_cost, contact_cost_max_);
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
        data_->qvel[0], data_->qvel[1], data_->qvel[2]);
    const Eigen::Matrix<mjtNum, 3, 1> base_pos(data_->qpos[0], data_->qpos[1],
                                               data_->qpos[2]);
    const Eigen::Quaternion<mjtNum> base_quat(
        data_->qpos[3], data_->qpos[4], data_->qpos[5], data_->qpos[6]);
    const Eigen::Matrix<mjtNum, 3, 1> base_lin_acc(
        data_->qacc[0], data_->qacc[1], data_->qacc[2]);
    const Eigen::Matrix<mjtNum, 3, 1> base_ang_vel(
        data_->qvel[3], data_->qvel[4], data_->qvel[5]);

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

    // ++elapsed_step_;
    done_ = done_ || (terminate_when_unhealthy_ ? !is_healthy : false) ||
            (elapsed_step_ >= max_episode_steps_);
    WriteState(reward, base_lin_vel[0], base_lin_vel[1], ctrl_cost,
               contact_cost, base_pos[0], base_pos[1], last_healthy_reward_);
  }

  void applyForce(const Eigen::Matrix<mjtNum, 6, 1>& force_and_torque) {
    if (base_body_id_ < 0 || base_body_id_ >= model_->nbody) {
      if (env_id_ == 0) {
        std::cerr << "[HumanoidEnv] Base body id invalid; cannot apply force."
                  << std::endl;
      }
      return;
    }

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

    const int offset = 6 * base_body_id_;
    std::fill(data_->xfrc_applied + offset, data_->xfrc_applied + offset + 6,
              static_cast<mjtNum>(0.0));
    for (int i = 0; i < 6; ++i) {
      data_->xfrc_applied[offset + i] = sanitized[i];
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

    // Uniform direction on the unit sphere
    constexpr mjtNum kTwoPi =
        static_cast<mjtNum>(6.283185307179586476925286766559);
    std::uniform_real_distribution<mjtNum> azimuth_dist(
        static_cast<mjtNum>(0.0), kTwoPi);
    std::uniform_real_distribution<mjtNum> z_dist(
        static_cast<mjtNum>(-1.0), static_cast<mjtNum>(1.0));
    const mjtNum z = z_dist(cmd_rng_);
    const mjtNum azimuth = azimuth_dist(cmd_rng_);
    const mjtNum r_xy = std::sqrt(std::max(static_cast<mjtNum>(0.0),
                                           static_cast<mjtNum>(1.0 - z * z)));
    Eigen::Matrix<mjtNum, 3, 1> direction(
        r_xy * std::cos(azimuth), r_xy * std::sin(azimuth), z);

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
  bool IsHealthy() {
    if (mbc.controlMode() == 0) {
      std::cout << "[IsHealthy] Passive state detected. env_id: " << env_id_ << std::endl;
      return false;  // end if state is passive
    }

    bool healthy =
        (healthy_z_min_ < data_->qpos[2]) && (data_->qpos[2] < healthy_z_max_);
    if (!healthy && render_mode_) {
      std::cout << "[IsHealthy] Unhealthy state detected: "
                << "z position = " << data_->qpos[2]
                << ", healthy_z_min = " << healthy_z_min_
                << ", healthy_z_max = " << healthy_z_max_ << std::endl;
    }
    return healthy;
  }

  std::array<mjtNum, 2> GetMassCenter() {
    mjtNum mass_sum = 0.0;
    mjtNum mass_x = 0.0;
    mjtNum mass_y = 0.0;
    for (int i = 0; i < model_->nbody; ++i) {
      mjtNum mass = model_->body_mass[i];
      mass_sum += mass;
      mass_x += mass * data_->xipos[i * 3 + 0];
      mass_y += mass * data_->xipos[i * 3 + 1];
    }
    return {mass_x / mass_sum, mass_y / mass_sum};
  }

  mjtNum ComputeOrientationPenalty() const {
    Eigen::Quaternion<mjtNum> q(data_->qpos[3], data_->qpos[4], data_->qpos[5],
                                data_->qpos[6]);
    Eigen::Vector3<mjtNum> eul = q.toRotationMatrix().eulerAngles(0, 1, 2);
    mjtNum roll = eul[0];
    mjtNum pitch = eul[1];
    return orientation_penalty_weight_ * (roll * roll + pitch * pitch);
  }

  mjtNum ComputeHeightPenalty() const {
    const mjtNum height = data_->qpos[2];
    const mjtNum desired = static_cast<mjtNum>(mbc.desiredBodyHeight());
    const mjtNum err = height - desired;
    return height_penalty_weight_ * err * err;
  }

  mjtNum ComputeFootSlipPenalty() const {
    mjtNum slip_cost = 0.0;
    const auto* loco = mbc.locomotionData();
    const auto* leg_ctrl = mbc.legController();
    if (!loco || !leg_ctrl) {
      return 0.0;
    }
    for (int leg = 0; leg < 4; ++leg) {
      if (loco->contact_state[leg] > 0.5f) {
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
    mbc.setCommandVelocity(cmd[0], cmd[1], cmd[2]);
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
        static_cast<mjtNum>(mbc.desiredBodyHeight());
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

    mjtNum* obs_end = mbc.setObservation(obs);
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
    (void)obs_end;
    // std::cout << "Filled " << static_cast<int>(obs_end - obs_start) << " elements in observation
    // array" << std::endl;
    // state["info:reward_linvel"_] = xv * forward_reward_weight_;
    // state["info:reward_quadctrl"_] = -ctrl_cost;
    // state["info:reward_alive"_] = healthy_reward;
    // state["info:reward_impact"_] = -contact_cost;
    // state["info:x_position"_] = x_after;
    // state["info:y_position"_] = y_after;
    // state["info:distance_from_origin"_] =
    //     std::sqrt(x_after * x_after + y_after * y_after);
    // state["info:x_velocity"_] = xv;
    // state["info:y_velocity"_] = yv;

    lastReward = reward;
    writeDataToCSV();

#ifdef ENVPOOL_TEST
    state["info:qpos0"_].Assign(qpos0_, model_->nq);
    state["info:qvel0"_].Assign(qvel0_, model_->nv);
#endif
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

    outputFile << elapsed_step_ << ',' << (reset_marker ? 1 : 0);
    for (int i = 0; i < 19; ++i) {
      outputFile << ',' << static_cast<double>(data_->qpos[i]);
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
               << static_cast<double>(mbc.desiredBodyHeight())
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

inline void mujoco_gym::HumanoidEnv::UpdateMbcDebugMarkers() {
  if (!render_mode_) {
    return;
  }
  const VectorNavData* imu = mbc.imuData();
  if (!imu) {
    return;
  }

  std::array<mjtNum, 3> body_pos{static_cast<mjtNum>(imu->pos_x),
                                 static_cast<mjtNum>(imu->pos_y),
                                 static_cast<mjtNum>(imu->pos_z)};

  std::vector<std::array<mjtNum, 3>> joint_positions;
  std::vector<std::array<mjtNum, 3>> foot_positions;
  joint_positions.reserve(ModelBasedControllerInterface::kNumLegs * 3);
  foot_positions.reserve(ModelBasedControllerInterface::kNumLegs);

  const auto* leg_controller = mbc.legController();
  auto* robot_model = mbc.robotModel();
  const SpiData* spi = mbc.actuatorData();

  bool used_model = false;
  if (leg_controller && robot_model && spi) {
    FBModelState<float> viz_state;
    Quat<float> body_quat(static_cast<float>(imu->quat[0]),
                          static_cast<float>(imu->quat[1]),
                          static_cast<float>(imu->quat[2]),
                          static_cast<float>(imu->quat[3]));
    if (body_quat.norm() < 1e-6f) {
      body_quat = Quat<float>::Identity();
    } else {
      body_quat.normalize();
    }
    viz_state.bodyOrientation = body_quat;
    viz_state.bodyPosition = Vec3<float>(static_cast<float>(imu->pos_x),
                                         static_cast<float>(imu->pos_y),
                                         static_cast<float>(imu->pos_z));
    viz_state.bodyVelocity.setZero();
    viz_state.q = DVec<float>::Zero(cheetah::num_act_joint);
    viz_state.qd = DVec<float>::Zero(cheetah::num_act_joint);
    for (int leg = 0; leg < ModelBasedControllerInterface::kNumLegs; ++leg) {
      viz_state.q[leg * 3 + 0] = spi->q_abad[leg];
      viz_state.q[leg * 3 + 1] = spi->q_hip[leg];
      viz_state.q[leg * 3 + 2] = spi->q_knee[leg];
      viz_state.qd[leg * 3 + 0] = spi->qd_abad[leg];
      viz_state.qd[leg * 3 + 1] = spi->qd_hip[leg];
      viz_state.qd[leg * 3 + 2] = spi->qd_knee[leg];
    }
    robot_model->setState(viz_state);
    robot_model->contactJacobians();

    for (int leg = 0; leg < ModelBasedControllerInterface::kNumLegs; ++leg) {
      for (int axis = 0; axis < 3; ++axis) {
        const Vec3<float> joint =
            robot_model->getPosition(6 + leg * 3 + axis);
        joint_positions.push_back(
            {static_cast<mjtNum>(joint[0]), static_cast<mjtNum>(joint[1]),
             static_cast<mjtNum>(joint[2])});
      }
    }

    const int gc_indices[ModelBasedControllerInterface::kNumLegs] = {
        linkID::FR, linkID::FL, linkID::HR, linkID::HL};
    for (int leg = 0; leg < ModelBasedControllerInterface::kNumLegs; ++leg) {
      const Vec3<float>& foot = robot_model->_pGC[gc_indices[leg]];
      foot_positions.push_back({static_cast<mjtNum>(foot[0]),
                                static_cast<mjtNum>(foot[1]),
                                static_cast<mjtNum>(foot[2])});
    }
    used_model = true;
  }

  if (!used_model) {
    for (int leg = 0; leg < ModelBasedControllerInterface::kNumLegs; ++leg) {
      foot_positions.push_back(
          {static_cast<mjtNum>(imu->foot_pos[leg * 3 + 0]),
           static_cast<mjtNum>(imu->foot_pos[leg * 3 + 1]),
           static_cast<mjtNum>(imu->foot_pos[leg * 3 + 2])});
    }
  }

  addSpheres(body_pos, joint_positions, foot_positions);
}

#endif  // ENVPOOL_MUJOCO_GYM_HUMANOID_H_
