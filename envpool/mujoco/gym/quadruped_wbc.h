// Unitree Go2 quadruped driven by the quadcontrol WBIC stack.
//
// This file began life as EnvPool's Gym MuJoCo Humanoid environment and kept
// that name long after every line of its dynamics, observation, action and
// reward had been replaced. Nothing here relates to Gym's Humanoid any more.
//
// Two task ids share this class, differing only in observation width:
//
//   QuadrupedWBC-v0  54 obs  symmetric        (the "Gamma" line)
//   QuadrupedWBC-v1  61 obs  privileged tail  (the "Delta" line, asymmetric
//                                              actor-critic; the trailing 7
//                                              values go to the critic only)
//
// The privileged values are written last and FillObservation is bounded by the
// buffer it is handed, so v0 is exactly v1 with the tail withheld. Verified
// against commit 0fd79b7 (the pre-Delta 54-dim env): the first 54 writes are
// unchanged.

#ifndef ENVPOOL_MUJOCO_GYM_QUADRUPED_WBC_H_
#define ENVPOOL_MUJOCO_GYM_QUADRUPED_WBC_H_

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>  // For std::sqrt, std::abs
#include <random>
#include <stdexcept>
#include <cstdio>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>
#include <vector>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/mujoco/gym/quadruped_domain_rand.h"

#include <Eigen/Dense>

#include "control/ControlMessages.hh"
#include "control/LocomotionCtrlData.hh"
#include "controllers/LegController.h"
#include "estimators/StateEstimatorTypes.hh"
#include "supervisor/RLPipelineRuntime.hh"
#include "types/cppTypes.h"

namespace mujoco_gym {

struct RLConstants {
  static constexpr int kNumLegs = 4;
  static constexpr int kActionDim = 25;
  // Gamma1 appended the body-height residual at index 23, so phase delta moved
  // to 24. Deriving the index keeps it last for free.
  static constexpr int kPhaseDeltaIdx = kActionDim - 1;
  static constexpr int kActorObservationDim = 54;
  static constexpr int kPrivilegedObservationDim = 7;
  static constexpr int kObservationDim = 61;
};

struct RewardWeights {
  mjtNum base_xvel{static_cast<mjtNum>(0.25)};
  mjtNum base_zvel{static_cast<mjtNum>(0.10)};
  mjtNum base_zpos{static_cast<mjtNum>(0.35)};
  mjtNum base_orientation{static_cast<mjtNum>(0.15)};
  mjtNum base_straight{static_cast<mjtNum>(0.15)};
  mjtNum base_linear_accel{static_cast<mjtNum>(0.025)};
  mjtNum base_angular_vel{static_cast<mjtNum>(0.025)};
  mjtNum action_smooth{static_cast<mjtNum>(0.10)};
  mjtNum phase_delta{static_cast<mjtNum>(0.05)};
  mjtNum action_magnitude{static_cast<mjtNum>(0.05)};
};

struct ReferenceTargets {
  mjtNum xdot_ref{static_cast<mjtNum>(0.8)};
  mjtNum zdot_ref{static_cast<mjtNum>(0.0)};
  // Overwritten every step from the controller's user_parameters.body_height;
  // this default only covers the window before the runtime exists.
  mjtNum z_ref{static_cast<mjtNum>(0.32)};
};

struct RewardConfig {
  RewardWeights weights{};
  ReferenceTargets refs{};
};

// Computes a Cassie-inspired locomotion reward from instantaneous penalties.
class LocomotionReward {
 public:
  static constexpr int kNumTerms = 10;
  enum TermIndex {
    kBaseXVel = 0,
    kBaseZVel,
    kBaseZPos,
    kBaseOrientation,
    kBaseStraight,
    kBaseLinearAccel,
    kBaseAngularVel,
    kActionSmooth,
    kPhaseDelta,
    kActionMagnitude
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
    // Height is a POSITION error in metres against a 0.32 m stance: 3.0 prices a
    // 4.6 cm sag at 13 % of one term (3.9 % of the total), which is why Beta3
    // rode at 0.274 m. 20.0 prices the same sag at 60 % of the term and still
    // saturates rather than exploding on a push transient.
    constexpr mjtNum kPosPenaltyScale = static_cast<mjtNum>(20.0);

    result.penalties[kBaseXVel] =
        kVelPenaltyScale * std::abs(base_lin_vel[0] - config_.refs.xdot_ref);
    result.penalties[kBaseZVel] =
        kVelPenaltyScale * std::abs(base_lin_vel[2] - config_.refs.zdot_ref);
    result.penalties[kBaseZPos] =
        kPosPenaltyScale * std::abs(base_pos[2] - config_.refs.z_ref);

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
    result.penalties[kActionMagnitude] = ComputeActionMagnitudePenalty(action);

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
                   w.phase_delta * result.rewards[kPhaseDelta] +
                   w.action_magnitude * result.rewards[kActionMagnitude];
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

  // Prices how LARGE an action is; kActionSmooth prices only how fast it
  // changes, so before Gamma1 nothing held a residual near zero in steady state.
  //
  // Covers the residual channels: the velocity residuals, per-leg Fx/Fy, the
  // footstep xy residuals, and the height residual. Two channels are excluded,
  // for different reasons.
  //
  // Per-leg Fz (5, 8, 11, 14): no neutral value to pull toward. It maps
  // [-1, 1] onto [0, force_z_max], so -1 commands 0 N, which fights body
  // support, while 0 commands force_z_max / 2 = 125 N against a 147 N robot --
  // already enough to ride high. Penalising |a| would nominate 125 N as "free"
  // for a reason unconnected to the dynamics.
  //
  // Phase delta (kPhaseDeltaIdx): a slow gait is WANTED -- it lowers cost of
  // transport and relaxes the actuators -- and kPhaseDelta already buys it with
  // a linear penalty whose minimum is a = -1. A quadratic term centred on
  // a = 0 (the nominal 1/200 rate, 0.4 s period) pulls the other way and very
  // nearly cancels it: moving a from 0 to -0.5 gains 0.0119 of reward from
  // kPhaseDelta and loses up to 0.0111 to this term. Worse, because the penalty
  // is exp(-sum a^2), that marginal cost depends on how active every OTHER
  // channel is, so the gait-rate incentive would become a function of unrelated
  // residuals. One dedicated, uncoupled term for gait rate is the point.
  mjtNum ComputeActionMagnitudePenalty(const std::vector<mjtNum>& action) const {
    mjtNum sq_sum = 0;
    const int n = static_cast<int>(action.size());
    for (int i = 0; i < n; ++i) {
      if (i == RLConstants::kPhaseDeltaIdx) continue;
      if (i >= 3 && i < 15 && ((i - 3) % 3) == 2) continue;  // per-leg Fz
      sq_sum += action[i] * action[i];
    }
    return sq_sum;
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

class QuadrupedWBCEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict(
        "frame_skip"_.Bind(5), "post_constraint"_.Bind(true),
        "use_contact_force"_.Bind(false), "forward_reward_weight"_.Bind(1),
        "terminate_when_unhealthy"_.Bind(true),
        // Ground-friction randomisation, matching the Rudin arm. Default
        // false: evaluation must run on the MJCF plant, and only training
        // configs opt in. See quadruped_domain_rand.h.
        "wbc_randomize_friction"_.Bind(false),
        // Legacy base-height termination, superseded by common_fall_v1.
        "terminate_on_height_band"_.Bind(false),
        "sim_config_path"_.Bind(
            std::string("/app/quadcontrol/config/robots/sim/envpool.toml")),
        // TOML text layered over sim_config_path. Lets a caller override force
        // settings without writing a temp config file whose fixed name made
        // concurrent evaluation processes clobber each other.
        "sim_config_overlay"_.Bind(std::string("")),
        "render_mode"_.Bind(false),
        "exclude_current_positions_from_observation"_.Bind(true),
        // false -> 54-dim actor observation (QuadrupedWBC-v0).
        // true  -> 61-dim, appending the 7 simulator-truth values
        // the asymmetric critic consumes (QuadrupedWBC-v1).
        "privileged_observations"_.Bind(true),
        "ctrl_cost_weight"_.Bind(2e-4), "healthy_reward"_.Bind(1.0),
        "healthy_z_min"_.Bind(0.20), "healthy_z_max"_.Bind(0.75),
        "contact_cost_weight"_.Bind(5e-7), "contact_cost_max"_.Bind(10.0),
        "velocity_tracking_weight"_.Bind(0.5),
        "yaw_tracking_weight"_.Bind(0.2),
        "orientation_penalty_weight"_.Bind(0.1),
        "height_penalty_weight"_.Bind(.9),
        "foot_slip_penalty_weight"_.Bind(0.1),
        "action_penalty_weight"_.Bind(5e-2));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    mjtNum inf = std::numeric_limits<mjtNum>::infinity();
    const int obs_n = conf["privileged_observations"_]
                          ? RLConstants::kObservationDim
                          : RLConstants::kActorObservationDim;
    return MakeDict("obs"_.Bind(Spec<mjtNum>({obs_n}, {-inf, inf})),
#ifdef ENVPOOL_TEST
                    "info:qpos0"_.Bind(Spec<mjtNum>({24})),
                    "info:qvel0"_.Bind(Spec<mjtNum>({23})),
#endif
                    "info:reward_linvel"_.Bind(Spec<mjtNum>({-1})),
                    "info:reward_quadctrl"_.Bind(Spec<mjtNum>({-1})),
                    "info:reward_alive"_.Bind(Spec<mjtNum>({-1})),
                    "info:sim_time"_.Bind(Spec<mjtNum>({-1})),
                    "info:force_onset"_.Bind(Spec<mjtNum>({-1})),
                    "info:force_phase"_.Bind(Spec<mjtNum>({-1})),
                    "info:force_applied"_.Bind(Spec<mjtNum>({3})),
                    "info:force_direction"_.Bind(Spec<int>({-1})),
                    "info:force_requested_impulse"_.Bind(Spec<mjtNum>({-1})),
                    "info:body_height"_.Bind(Spec<mjtNum>({-1})),
                    // legged_gym's time_out_buf. EnvPool's own `trunc` counts
                    // steps since reset, which the recipe's randomised first
                    // episode length makes wrong, so the flag is explicit.
                    "info:time_out"_.Bind(Spec<int>({-1})),
                    // Controller-independent evaluator rule from simulator
                    // truth: base contact, excessive roll/pitch, or non-finite
                    // rigid-body state. The joint-PD task emits the same key.
                    "info:fall"_.Bind(Spec<int>({-1})),
                    "info:reward_terms"_.Bind(
                        Spec<mjtNum>({LocomotionReward::kNumTerms})),
                    "info:reward_impact"_.Bind(Spec<mjtNum>({-1})),
                    "info:x_position"_.Bind(Spec<mjtNum>({-1})),
                    "info:y_position"_.Bind(Spec<mjtNum>({-1})),
                    "info:distance_from_origin"_.Bind(Spec<mjtNum>({-1})),
                    "info:x_velocity"_.Bind(Spec<mjtNum>({-1})),
                    "info:y_velocity"_.Bind(Spec<mjtNum>({-1})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    // Action layout (25 dims total):
    // [0] vBody_des.x residual, [1] vBody_des.y residual, [2] yaw_rate residual
    // [3..14] per-leg ground reaction force targets (4 legs x 3): [fx, fy, fz]
    // [15..22] per-leg swing foot residuals (4 legs x 2): [x, y]
    // [23] body-height residual (mapped to +-height_residual_limit, added to
    //      the commanded stance height in MdlRLLocomotionState)
    // [24] gait phase delta_theta (normalized, mapped to 0..kMaxPhaseDelta)
    return MakeDict("action"_.Bind(
        Spec<mjtNum>({-1, RLConstants::kActionDim}, {-1.0, 1.0})));
  }
};

using QuadrupedWBCEnvSpec = EnvSpec<QuadrupedWBCEnvFns>;

class QuadrupedWBCEnv : public Env<QuadrupedWBCEnvSpec> {
 protected:
  // common_fall_v1 thresholds; identical to metrics.py and quadruped_pd.h.
  static constexpr double kFallContactForceN = 1.0;
  static constexpr double kFallRollRad = 0.8;
  static constexpr double kFallPitchRad = 1.0;
  bool terminate_when_unhealthy_, no_pos_, use_contact_force_, render_mode_;
  bool terminate_on_height_band_{false};
  bool randomize_friction_{false};
  mjtNum friction_{0.0};  // effective contact coefficient in use
  mjtNum ctrl_cost_weight_, forward_reward_weight_, healthy_reward_;
  mjtNum healthy_z_min_, healthy_z_max_;
  mjtNum contact_cost_weight_, contact_cost_max_;

  std::unique_ptr<quadruped::RLPipelineRuntime> runtime_;

  int frame_skip_{};
  int max_episode_steps_{};
  int elapsed_step_{0};
  bool done_{true};
  bool disabled_{false};

  quadruped::GaitSchedule last_gait_schedule_{};
  quadruped::StateEstimate<float> last_state_est_{};

  float lastReward = 0;
  // Stance-height reference, refreshed from the controller every step.
  mjtNum desired_h{static_cast<mjtNum>(0.32)};
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
  std::vector<mjtNum> last_action_vector_;
  std::vector<mjtNum> prev_action_vector_;
  mjtNum last_ctrl_cost_{0.0};
  mjtNum last_contact_cost_{0.0};
  mjtNum last_healthy_reward_{0.0};
  mjtNum last_x_velocity_{0.0};
  mjtNum last_y_velocity_{0.0};
  mjtNum last_x_position_{0.0};
  mjtNum last_y_position_{0.0};
  bool last_is_healthy_{true};
  std::string last_done_reason_{"init"};
  std::string sim_config_path_{
      "/app/quadcontrol/config/robots/sim/envpool.toml"};
  std::string sim_config_overlay_;

 public:
  QuadrupedWBCEnv(const Spec& spec, int env_id)
      : Env<QuadrupedWBCEnvSpec>(spec, env_id),
        terminate_when_unhealthy_(spec.config["terminate_when_unhealthy"_]),
        no_pos_(spec.config["exclude_current_positions_from_observation"_]),
        use_contact_force_(spec.config["use_contact_force"_]),
        render_mode_(spec.config["render_mode"_]),
        ctrl_cost_weight_(spec.config["ctrl_cost_weight"_]),
        forward_reward_weight_(spec.config["forward_reward_weight"_]),
        healthy_reward_(spec.config["healthy_reward"_]),
        healthy_z_min_(spec.config["healthy_z_min"_]),
        healthy_z_max_(spec.config["healthy_z_max"_]),
        contact_cost_weight_(spec.config["contact_cost_weight"_]),
        contact_cost_max_(spec.config["contact_cost_max"_]),
        frame_skip_(spec.config["frame_skip"_]),
        max_episode_steps_(spec.config["max_episode_steps"_]),
        velocity_tracking_weight_(spec.config["velocity_tracking_weight"_]),
        yaw_tracking_weight_(spec.config["yaw_tracking_weight"_]),
        orientation_penalty_weight_(spec.config["orientation_penalty_weight"_]),
        height_penalty_weight_(spec.config["height_penalty_weight"_]),
        foot_slip_penalty_weight_(spec.config["foot_slip_penalty_weight"_]),
        action_penalty_weight_(spec.config["action_penalty_weight"_]),
        locomotion_reward_config_(),
        locomotion_reward_(locomotion_reward_config_) {
    const std::string cfg_sim_path = spec.config["sim_config_path"_];
    if (!cfg_sim_path.empty()) {
      sim_config_path_ = cfg_sim_path;
    }
    sim_config_overlay_ = spec.config["sim_config_overlay"_];

    {
      static std::mutex s_init_mutex;
      std::lock_guard<std::mutex> lock(s_init_mutex);
      quadruped::RLPipelineRuntime::Options rt_opts;
      rt_opts.env_id = env_id_;
      rt_opts.sim_config_path = sim_config_path_;
      rt_opts.sim_config_overlay = sim_config_overlay_;
      rt_opts.pipeline = quadruped::RLPipelineRuntime::Pipeline::kWbic;
      runtime_ = std::make_unique<quadruped::RLPipelineRuntime>(rt_opts);
      runtime_->initialize();
      disabled_ = runtime_->disabled();
    }

    // One fixed coefficient per env for its whole lifetime, drawn from the
    // pool-wide buckets, exactly as the Rudin arm does. With randomisation off
    // the MJCF's nominal friction stands untouched -- including the
    // `evaluation_friction` robustness cell, which the config applies earlier.
    terminate_on_height_band_ = spec.config["terminate_on_height_band"_];
    randomize_friction_ = spec.config["wbc_randomize_friction"_];
    if (!disabled_ && runtime_) {
      friction_ = runtime_->groundFriction();
      if (randomize_friction_) {
        friction_ = FrictionRandomisation::Sample(
            friction_, static_cast<unsigned>(spec.config["seed"_]), &gen_);
        runtime_->setFriction(friction_);
      }
    }

    done_ = false;
    elapsed_step_ = 0;
    locomotion_reward_.Reset();
    UpdateRewardReferences();
    if (!disabled_ && runtime_) {
      runtime_->stepFrame();
      last_state_est_ = runtime_->stateEstimate();
      last_gait_schedule_ = runtime_->gaitSchedule();
    }
    prev_action_vector_.clear();
  }

  ~QuadrupedWBCEnv() override = default;

  bool IsDone() override { return done_; }

  void Reset() override {
    if (disabled_) {
      done_ = true;
      return;
    }
    last_action_vector_.clear();
    prev_action_vector_.clear();
    last_penalties_.fill(static_cast<mjtNum>(0.0));
    last_smoothed_penalties_.fill(static_cast<mjtNum>(0.0));
    last_term_rewards_.fill(static_cast<mjtNum>(0.0));
    locomotion_reward_.Reset();
    if(render_mode_)
      std::cout << "Resetting the environment..." << std::endl;
    done_ = false;
    elapsed_step_ = 0;
    if (runtime_) {
      constexpr int kMaxResetAttempts = 3;
      int reset_attempt = 0;
      for (; reset_attempt < kMaxResetAttempts; ++reset_attempt) {
        runtime_->resetEpisode();
        runtime_->setPolicyAction(nullptr, 0);
        last_state_est_ = runtime_->stateEstimate();
        last_gait_schedule_ = runtime_->gaitSchedule();
        const mjtNum z = static_cast<mjtNum>(last_state_est_.position[2]);
        if ((healthy_z_min_ < z) && (z < healthy_z_max_)) {
          break;
        }
        if (env_id_ == 0) {
          std::cerr << "[QuadrupedWBCEnv] Reset health retry " << (reset_attempt + 1)
                    << "/" << kMaxResetAttempts
                    << " (z=" << z << ", bounds=[" << healthy_z_min_ << ", "
                    << healthy_z_max_ << "])\n";
        }
      }
      if (env_id_ == 0) {
        const mjtNum z = static_cast<mjtNum>(last_state_est_.position[2]);
        std::cout << "[QuadrupedWBCEnv] Reset state env_id=" << env_id_
                  << " z=" << z
                  << " rpy=[" << static_cast<mjtNum>(last_state_est_.rpy[0]) << ", "
                  << static_cast<mjtNum>(last_state_est_.rpy[1]) << ", "
                  << static_cast<mjtNum>(last_state_est_.rpy[2]) << "]\n";
      }
    }
    WriteState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    UpdateRewardReferences();
  }

  void Step(const Action& action) override {
    if (disabled_) {
      done_ = true;
      return;
    }
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
      std::cerr << "[QuadrupedWBCEnv] Non-finite action detected; replaced with zeros." << std::endl;
    }
    last_action_vector_.assign(act, act + action_count);
    if (runtime_) {
      runtime_->setPolicyAction(act, action_count);
    }

    mjtNum ctrl_cost = 0.0;

    for (int i = 0; i < frame_skip_; ++i) {
      if (runtime_) {
        runtime_->stepFrame();
        ctrl_cost += runtime_->computeControlCost(ctrl_cost_weight_);
        last_state_est_ = runtime_->stateEstimate();
        last_gait_schedule_ = runtime_->gaitSchedule();
      }
    }

    // Compute contact cost.
    mjtNum contact_cost = 0.0;
    if (use_contact_force_) {
      contact_cost = 0.0;
    }

    // Reference-height randomisation is deliberately absent: it is only sound
    // now that desired_h is observed, and Gamma1 holds the reference fixed at
    // the controller's 0.32 m so the tracking result is unambiguous.
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
        std::cerr << "[QuadrupedWBCEnv] Non-finite reward detected; forcing termination." << std::endl;
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
        if (env_id_ == 0) {
          std::cerr << "[QuadrupedWBCEnv] unhealthy terminate at step=" << elapsed_step_
                    << " z=" << static_cast<mjtNum>(last_state_est_.position[2])
                    << " rpy=[" << static_cast<mjtNum>(last_state_est_.rpy[0]) << ", "
                    << static_cast<mjtNum>(last_state_est_.rpy[1]) << ", "
                    << static_cast<mjtNum>(last_state_est_.rpy[2]) << "]"
                    << " v_body=[" << static_cast<mjtNum>(last_state_est_.vBody[0]) << ", "
                    << static_cast<mjtNum>(last_state_est_.vBody[1]) << ", "
                    << static_cast<mjtNum>(last_state_est_.vBody[2]) << "]\n";
        }
      } else if (hit_timeout) {
        last_done_reason_ = "timeout";
      } else {
        last_done_reason_ = "done";
      }
    }
    WriteState(reward, base_lin_vel[0], base_lin_vel[1], ctrl_cost,
               contact_cost, base_pos[0], base_pos[1], last_healthy_reward_);
  }

 private:
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

    const mjtNum cmd_vx = runtime_
                              ? static_cast<mjtNum>(
                                    runtime_->desiredVelocity().body_velocity[0])
                              : static_cast<mjtNum>(0.0);
    write_value(cmd_vx);

    for (int i = 0; i < 3; ++i) {
      write_value(static_cast<mjtNum>(last_state_est_.vBody[i]));
    }
    for (int i = 0; i < 3; ++i) {
      write_value(static_cast<mjtNum>(last_state_est_.omegaBody[i]));
    }

    for (int i = 0; i < 3; ++i) {
      write_value(static_cast<mjtNum>(last_state_est_.rpy[i]));
    }

    auto* leg_controller = runtime_ ? runtime_->legController() : nullptr;
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
    // All four legs. Before Gamma1 only the front pair was written, so a
    // backward push loaded exactly the legs the policy could not see.
    for (int leg = 0; leg < RLConstants::kNumLegs; ++leg) {
      for (int axis = 0; axis < 3; ++axis) {
        write_value(foot_pos_body[leg][axis]);
      }
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

    // Height and its reference. The reward has always penalised |z - z_ref|,
    // but neither term was observable, so the height task was partially
    // observed and the policy had no way to close the loop on it.
    write_value(static_cast<mjtNum>(last_state_est_.position[2]));
    write_value(static_cast<mjtNum>(desired_h));

    // Delta asymmetric actor-critic: these final seven simulator-truth values
    // are sliced away before the actor network and are consumed only by the
    // critic. Force is the applied world-frame vector; contacts come from
    // MuJoCo's physical foot contacts rather than the commanded gait schedule.
    const SimForceState force = runtime_ ? runtime_->forceState() : SimForceState{};
    for (int axis = 0; axis < 3; ++axis) write_value(force.applied[axis]);
    const std::array<float, RLConstants::kNumLegs> contacts =
        runtime_ ? runtime_->footContactState()
                 : std::array<float, RLConstants::kNumLegs>{};
    for (float contact : contacts) write_value(static_cast<mjtNum>(contact));
  }

  // Termination is `common_fall_v1`, the SAME rule evaluation scores on and the
  // same one the Rudin arm already trains against (quadruped_pd.h
  // check_termination): trunk contact above 1 N, roll past 0.8 rad or pitch
  // past 1.0 rad, or a non-finite state. This used to be a base-height band,
  // which meant the policy was optimised to stay between 0.20 m and 0.75 m
  // while being judged on attitude and contact -- a robot could pitch to
  // 1.1 rad and count as a training success but an evaluation failure.
  //
  // The height band is retained only as a backstop, gated on
  // `terminate_on_height_band`, off by default.
  bool IsHealthy() {
    const auto& p = last_state_est_.position;
    const mjtNum roll = static_cast<mjtNum>(last_state_est_.rpy[0]);
    const mjtNum pitch = static_cast<mjtNum>(last_state_est_.rpy[1]);

    const bool finite_state =
        std::isfinite(static_cast<double>(p[0])) &&
        std::isfinite(static_cast<double>(p[1])) &&
        std::isfinite(static_cast<double>(p[2])) &&
        std::isfinite(static_cast<double>(roll)) &&
        std::isfinite(static_cast<double>(pitch));
    const bool base_contact =
        runtime_ && runtime_->baseContactForceNorm() > kFallContactForceN;
    const bool attitude =
        std::abs(roll) > kFallRollRad || std::abs(pitch) > kFallPitchRad;

    bool healthy = finite_state && !base_contact && !attitude;
    if (healthy && terminate_on_height_band_) {
      const mjtNum z = static_cast<mjtNum>(p[2]);
      healthy = (healthy_z_min_ < z) && (z < healthy_z_max_);
    }
    if (!healthy && render_mode_) {
      std::cout << "[IsHealthy] fall: contact=" << base_contact
                << " attitude=" << attitude << " finite=" << finite_state
                << " (roll=" << roll << ", pitch=" << pitch
                << ", z=" << p[2] << ")" << std::endl;
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
    auto* leg_ctrl = runtime_ ? runtime_->legController() : nullptr;
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

  void UpdateRewardReferences() {
    locomotion_reward_config_.refs.xdot_ref = runtime_
                                                  ? static_cast<mjtNum>(
                                                        runtime_->desiredVelocity()
                                                            .body_velocity[0])
                                                  : static_cast<mjtNum>(0.0);
    locomotion_reward_config_.refs.zdot_ref = static_cast<mjtNum>(0.0);
    // The controller's user_parameters.body_height is the single stance-height
    // reference in the system -- the same 0.32 m the convex-MPC baseline
    // commands. Before Gamma1 this read a separate envpool-side 0.35 m default
    // that nothing overrode, so the reward asked for a height the controller
    // never commanded. NOMINAL, deliberately: the policy's height residual is
    // not added here, or it would be free and the sag would return.
    if (runtime_) {
      desired_h = static_cast<mjtNum>(runtime_->nominalBodyHeight());
    }
    locomotion_reward_config_.refs.z_ref =
        static_cast<mjtNum>(desired_h);
    locomotion_reward_.SetReferences(locomotion_reward_config_.refs);
  }

  int EncodeDoneReason() const {
    if (!done_) return 0;
    if (last_done_reason_ == "unhealthy") return 1;
    if (last_done_reason_ == "timeout") return 2;
    if (last_done_reason_ == "nonfinite_reward") return 3;
    return 4;
  }

  void PublishDebugFrame(const mjtNum* obs, std::size_t obs_count, mjtNum reward,
                         mjtNum ctrl_cost, mjtNum contact_cost,
                         mjtNum healthy_reward) {
    if (!runtime_ || env_id_ != 0) {
      return;
    }

    quadruped::RLPipelineRuntime::EnvDebugFrame frame;
    frame.env_id = env_id_;
    frame.step = elapsed_step_;
    frame.done = done_ ? 1 : 0;
    frame.done_reason = EncodeDoneReason();
    frame.is_healthy = last_is_healthy_ ? 1 : 0;
    frame.reward_total = static_cast<float>(reward);
    frame.ctrl_cost = static_cast<float>(ctrl_cost);
    frame.contact_cost = static_cast<float>(contact_cost);
    frame.healthy_reward = static_cast<float>(healthy_reward);

    const std::size_t action_n =
        std::min(last_action_vector_.size(), frame.action.size());
    for (std::size_t i = 0; i < action_n; ++i) {
      frame.action[i] = static_cast<float>(last_action_vector_[i]);
    }

    if (obs) {
      const std::size_t obs_n = std::min(obs_count, frame.observation.size());
      for (std::size_t i = 0; i < obs_n; ++i) {
        frame.observation[i] = static_cast<float>(obs[i]);
      }
    }

    for (std::size_t i = 0; i < frame.reward_terms.size(); ++i) {
      frame.reward_terms[i] = static_cast<float>(last_term_rewards_[i]);
      frame.reward_penalties[i] = static_cast<float>(last_penalties_[i]);
    }
    for (int i = 0; i < 3; ++i) {
      frame.base_pos[i] = static_cast<float>(last_state_est_.position[i]);
      frame.base_vel_world[i] = static_cast<float>(last_state_est_.vWorld[i]);
    }

    runtime_->setEnvDebugFrame(frame);
  }

  // ------------------------------------------------------------------------

  void WriteState(float reward, mjtNum xv, mjtNum yv, mjtNum ctrl_cost,
                  mjtNum contact_cost, mjtNum x_after, mjtNum y_after,
                  mjtNum healthy_reward) {
    State state = Allocate();
    state["reward"_] = reward;
    const SimForceState force = runtime_ ? runtime_->forceState() : SimForceState{};
    state["info:sim_time"_] = force.time;
    state["info:force_onset"_] = force.onset;
    state["info:force_phase"_] = force.phase;
    state["info:force_direction"_] = force.direction;
    state["info:force_requested_impulse"_] = force.requested_impulse;
    const SimRobotState truth = runtime_ ? runtime_->robotState() : SimRobotState{};
    state["info:body_height"_] = truth.base_pos[2];
    state["info:time_out"_] =
        (done_ && last_done_reason_ == "timeout") ? 1 : 0;
    const mjtNum qw = truth.base_quat_wxyz[0];
    const mjtNum qx = truth.base_quat_wxyz[1];
    const mjtNum qy = truth.base_quat_wxyz[2];
    const mjtNum qz = truth.base_quat_wxyz[3];
    const mjtNum roll = std::atan2(2 * (qw * qx + qy * qz),
                                  1 - 2 * (qx * qx + qy * qy));
    const mjtNum pitch = std::asin(std::clamp(
        2 * (qw * qy - qz * qx), static_cast<mjtNum>(-1),
        static_cast<mjtNum>(1)));
    bool finite_truth = true;
    for (int i = 0; i < 3; ++i) {
      finite_truth = finite_truth && std::isfinite(truth.base_pos[i]) &&
                     std::isfinite(truth.lin_vel_world[i]);
    }
    const bool common_fall = !finite_truth ||
        (runtime_ && runtime_->baseContactForceNorm() > 1.0) ||
        std::abs(roll) > 0.8 || std::abs(pitch) > 1.0;
    state["info:fall"_] = common_fall ? 1 : 0;
    auto reward_terms = state["info:reward_terms"_];
    auto* reward_term_data = static_cast<mjtNum*>(reward_terms.Data());
    for (std::size_t i = 0; i < last_term_rewards_.size(); ++i) {
      reward_term_data[i] = last_term_rewards_[i];
    }
    auto applied = state["info:force_applied"_];
    auto* force_data = static_cast<mjtNum*>(applied.Data());
    for (int i = 0; i < 3; ++i) force_data[i] = force.applied[i];
    // Debug the state allocation - check if it's properly created
    // std::cout << "State obs buffer size: "
    // << state["obs"_].Shape()[0]  << state["obs"_].Shape()[1] << std::endl;
    auto obs_array = state["obs"_];
    auto* obs = static_cast<mjtNum*>(obs_array.Data());

    FillObservation(obs, obs_array.size);
    bool invalid_obs = false;
    for (std::size_t idx = 0; idx < obs_array.size; ++idx) {
      if (!std::isfinite(static_cast<double>(obs[idx]))) {
        obs[idx] = 0.0;
        invalid_obs = true;
      }
    }
    if (invalid_obs && env_id_ == 0) {
      std::cerr << "[QuadrupedWBCEnv] Non-finite observation detected; replaced with zeros." << std::endl;
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
    state["info:x_position"_] = truth.base_pos[0];
    state["info:y_position"_] = truth.base_pos[1];
    state["info:distance_from_origin"_] =
        std::hypot(truth.base_pos[0], truth.base_pos[1]);
    state["info:x_velocity"_] = truth.lin_vel_world[0];
    state["info:y_velocity"_] = truth.lin_vel_world[1];

    PublishDebugFrame(obs, obs_array.size, reward, ctrl_cost, contact_cost,
                      healthy_reward);
    lastReward = reward;
  }
};

using QuadrupedWBCEnvPool = AsyncEnvPool<QuadrupedWBCEnv>;

}  // namespace mujoco_gym

#endif  // ENVPOOL_MUJOCO_GYM_QUADRUPED_WBC_H_
