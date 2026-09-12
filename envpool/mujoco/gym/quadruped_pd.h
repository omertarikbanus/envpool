// End-to-end joint-position PD policy on the Unitree Go2: an exact recreation
// of unitree_rl_gym's Go2 task (GO2RoughCfg over LeggedRobotCfg, commit
// 276801e), driven through the quadcontrol runtime in its joint-PD pipeline.
//
//   QuadrupedPD-v1   48 obs / 12 actions
//
// This task shares the simulator and the runtime wrapper with QuadrupedWBC-*,
// and nothing else: different observation space, action space, reward,
// termination rule and reset distribution. It was previously a branch inside
// quadruped_wbc.h and is a separate environment now.
//
// Every constant here is read from the recipe; see
// quadcontrol/docs/RL_ONLY_BASELINE_PLAN.md for the deviation table.

#ifndef ENVPOOL_MUJOCO_GYM_QUADRUPED_PD_H_
#define ENVPOOL_MUJOCO_GYM_QUADRUPED_PD_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/mujoco/gym/quadruped_domain_rand.h"

#include <Eigen/Dense>

#include "estimators/StateEstimatorTypes.hh"
#include "supervisor/RLPipelineRuntime.hh"
#include "types/cppTypes.h"

namespace mujoco_gym {

// unitree_rl_gym (commit 276801e), GO2RoughCfg over LeggedRobotCfg. Every value
// here is read from that source.
struct PDConstants {
  static constexpr int kActionDim = 12;
  static constexpr int kObservationDim = 48;
  static constexpr int kRewardTermDim = 10;
  static constexpr double kPolicyDt = 0.02;          // sim.dt 0.005 x decimation 4
  static constexpr int kMaxEpisodeLength = 1000;     // ceil(episode_length_s 20 / dt)
  static constexpr int kResampleSteps = 500;         // commands.resampling_time 10 s
  static constexpr int kPushInterval = 750;          // ceil(push_interval_s 15 / dt)
  static constexpr double kMaxPushVelXY = 1.0;       // domain_rand.max_push_vel_xy
  static constexpr double kClipObservations = 100.0;
  static constexpr double kClipActions = 100.0;
  static constexpr double kTrackingSigma = 0.25;
  static constexpr double kSoftDofPosLimit = 0.9;    // GO2 override
  static constexpr double kInitHeight = 0.42;        // GO2 init_state.pos z
  // Evaluation fairness takes precedence over the source recipe's stale Go1
  // calf effort (35.55 N.m). Every arm uses the same Go2 plant limits, which
  // come from the MJCF's actuator forcerange, never from this task.
  // Go2 URDF joint limits in URDF convention, [lower, upper] per joint type.
  // Front and rear thighs differ.
  static constexpr double kHipLimits[2] = {-1.0472, 1.0472};
  static constexpr double kFrontThighLimits[2] = {-1.5708, 3.4907};
  static constexpr double kRearThighLimits[2] = {-0.5236, 4.5379};
  static constexpr double kCalfLimits[2] = {-2.7227, -0.83776};
  // This MJCF's thigh and knee axes are (0,-1,0) against the URDF's (0,1,0):
  // q_urdf = kUrdfSign * q_here, per [abad, hip, knee].
  static constexpr double kUrdfSign[3] = {1.0, -1.0, -1.0};
  // Friction randomisation is shared with the WBC arm: see
  // quadruped_domain_rand.h for the range, buckets and PhysX equivalence.
};

class QuadrupedPDEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict(
        "frame_skip"_.Bind(10), "post_constraint"_.Bind(true),
        "terminate_when_unhealthy"_.Bind(true),
        "render_mode"_.Bind(false),
        "sim_config_path"_.Bind(
            std::string("/app/quadcontrol/config/robots/sim/envpool.toml")),
        // TOML text layered over sim_config_path, so a caller can override
        // force settings without writing a temp config file.
        "sim_config_overlay"_.Bind(std::string("")),
        // Task switches, all ON in the recipe. Exposed so an evaluation can
        // turn the training-only randomisation off.
        "pd_obs_noise"_.Bind(true), "pd_push_robots"_.Bind(true),
        "pd_randomize_friction"_.Bind(true),
        "pd_init_at_random_ep_len"_.Bind(true),
        // false: reset to the default pose at rest (evaluation), instead of
        // legged_gym's q0 * U(0.5, 1.5) and U(-0.5, 0.5) base velocities.
        "pd_randomize_reset"_.Bind(true),
        // Evaluation: replace legged_gym's command sampler with a fixed
        // (vx, vy, heading) command. Heading mode stays on, as in the recipe.
        "pd_fixed_command"_.Bind(false), "pd_command_vx"_.Bind(0.0),
        "pd_command_vy"_.Bind(0.0), "pd_command_heading"_.Bind(0.0));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    return MakeDict(
        "obs"_.Bind(Spec<mjtNum>({PDConstants::kObservationDim},
                                 {-PDConstants::kClipObservations,
                                  PDConstants::kClipObservations})),
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
        // legged_gym's time_out_buf. EnvPool's own `trunc` counts steps since
        // reset, which the recipe's randomised first episode length makes
        // wrong, so the flag is explicit.
        "info:time_out"_.Bind(Spec<int>({-1})),
        // 1 if this step meets legged_gym's fall rule (base contact > 1 N,
        // |pitch| > 1.0, |roll| > 0.8), whether or not termination is enabled.
        "info:fall"_.Bind(Spec<int>({-1})),
        "info:reward_terms"_.Bind(
            Spec<mjtNum>({PDConstants::kRewardTermDim})),
        "info:reward_impact"_.Bind(Spec<mjtNum>({-1})),
        "info:x_position"_.Bind(Spec<mjtNum>({-1})),
        "info:y_position"_.Bind(Spec<mjtNum>({-1})),
        "info:distance_from_origin"_.Bind(Spec<mjtNum>({-1})),
        "info:x_velocity"_.Bind(Spec<mjtNum>({-1})),
        "info:y_velocity"_.Bind(Spec<mjtNum>({-1})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    // 12 joint-position targets, scaled by action_scale and added to the
    // default pose. Clipped to +-100, as the recipe does.
    return MakeDict("action"_.Bind(
        Spec<mjtNum>({-1, PDConstants::kActionDim},
                     {-PDConstants::kClipActions, PDConstants::kClipActions})));
  }
};

using QuadrupedPDEnvSpec = EnvSpec<QuadrupedPDEnvFns>;

class QuadrupedPDEnv : public Env<QuadrupedPDEnvSpec> {
 protected:
  bool terminate_when_unhealthy_;

  std::unique_ptr<quadruped::RLPipelineRuntime> runtime_;

  int frame_skip_{};
  int elapsed_step_{0};
  bool done_{true};
  bool disabled_{false};

  // Task state, named after the legged_gym buffers it mirrors.
  std::array<mjtNum, PDConstants::kActionDim> actions_{};
  std::array<mjtNum, PDConstants::kActionDim> last_actions_{};
  std::array<mjtNum, PDConstants::kActionDim> last_dof_vel_{};
  std::array<mjtNum, PDConstants::kActionDim> default_q_{};
  std::array<mjtNum, 4> feet_air_time_{};
  std::array<bool, 4> last_contacts_{};
  std::array<mjtNum, 4> commands_{};  // vx, vy, yaw rate, heading
  std::array<mjtNum, PDConstants::kObservationDim> obs_{};
  std::array<mjtNum, PDConstants::kRewardTermDim> terms_{};
  bool time_out_{false};
  bool fall_{false};
  int reset_count_{0};
  bool obs_noise_{true}, push_robots_{true}, random_friction_{true};
  bool random_ep_len_{true}, fixed_command_{false}, random_reset_{true};
  mjtNum fixed_vx_{0.0}, fixed_vy_{0.0}, fixed_heading_{0.0};
  mjtNum friction_{0.0};  // set at Init from the MJCF ground

  std::string sim_config_path_{
      "/app/quadcontrol/config/robots/sim/envpool.toml"};
  std::string sim_config_overlay_;

 public:
  QuadrupedPDEnv(const Spec& spec, int env_id)
      : Env<QuadrupedPDEnvSpec>(spec, env_id),
        terminate_when_unhealthy_(spec.config["terminate_when_unhealthy"_]),
        frame_skip_(spec.config["frame_skip"_]) {
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
      rt_opts.pipeline = quadruped::RLPipelineRuntime::Pipeline::kJointPD;
      runtime_ = std::make_unique<quadruped::RLPipelineRuntime>(rt_opts);
      runtime_->initialize();
      disabled_ = runtime_->disabled();
    }

    done_ = false;
    elapsed_step_ = 0;
    // One frame is stepped here before Init and before the first Reset. This
    // is inherited from the shared constructor this task used to live in, and
    // it is load-bearing: every measured result was produced with it. Removing
    // it shifts the initial state and must be verified by re-measurement, not
    // assumed safe.
    if (!disabled_ && runtime_) {
      runtime_->stepFrame();
    }
    Init(spec);
  }

  ~QuadrupedPDEnv() override = default;

  bool IsDone() override { return done_; }

  void Reset() override {
    if (disabled_) {
      done_ = true;
      return;
    }
    done_ = false;
    time_out_ = false;
    fall_ = false;
    runtime_->resetEpisode();
    runtime_->setPolicyAction(nullptr, 0);

    // _reset_root_states + _reset_dofs.
    SimRobotState st{};
    st.base_pos[2] = PDConstants::kInitHeight;
    for (int i = 0; i < 3; ++i) {
      st.lin_vel_world[i] = random_reset_ ? Uniform(-0.5, 0.5) : 0.0;
      st.ang_vel_world[i] = random_reset_ ? Uniform(-0.5, 0.5) : 0.0;
    }
    for (int i = 0; i < 12; ++i) {
      // default_dof_pos * U(0.5, 1.5). The convention map is a per-joint sign,
      // so scaling commutes with it.
      st.q[i] = default_q_[i] * (random_reset_ ? Uniform(0.5, 1.5) : 1.0);
      st.qd[i] = 0.0;
    }
    runtime_->setRobotState(st);
    ResampleCommands();

    // reset_idx buffer resets. last_contacts is deliberately NOT cleared:
    // legged_gym does not clear it either.
    actions_.fill(0.0);
    last_actions_.fill(0.0);
    last_dof_vel_.fill(0.0);
    feet_air_time_.fill(0.0);

    // init_at_random_ep_len: OnPolicyRunner.learn() randomises
    // episode_length_buf once, over [0, max_episode_length).
    elapsed_step_ = 0;
    if (reset_count_ == 0 && random_ep_len_) {
      elapsed_step_ = std::uniform_int_distribution<int>(
          0, PDConstants::kMaxEpisodeLength - 1)(gen_);
    }

    const SimRobotState now = runtime_->robotState();
    // Policy inputs, including heading feedback, must follow the same sensor
    // path as the other arms.  `now` below is retained solely for simulator
    // bookkeeping and truth-labelled evaluation signals.
    const PDBase observed = ComputeObservedBase();
    UpdateHeadingCommand(observed);
    ComputeObservation(observed);
    terms_.fill(0.0);

    // In legged_gym an env reset inside post_physics_step has
    // episode_length_buf == 0 when _push_robots runs, and 0 % interval == 0,
    // so every reset except the very first is followed by a push. The
    // observation above predates it, as legged_gym's does.
    if (push_robots_ && reset_count_ > 0) {
      runtime_->setBaseLinearVelocityXY(
          Uniform(-PDConstants::kMaxPushVelXY, PDConstants::kMaxPushVelXY),
          Uniform(-PDConstants::kMaxPushVelXY, PDConstants::kMaxPushVelXY));
    }
    ++reset_count_;
    WriteState(0.0, now);
  }

  void Step(const Action& action) override {
    if (disabled_) {
      done_ = true;
      return;
    }
    auto action_array = action["action"_];
    const auto* act = static_cast<const mjtNum*>(action_array.Data());
    const std::size_t n =
        std::min<std::size_t>(action_array.size, PDConstants::kActionDim);
    for (std::size_t i = 0; i < PDConstants::kActionDim; ++i) {
      const mjtNum a = (i < n && std::isfinite(static_cast<double>(act[i])))
                           ? act[i]
                           : 0.0;
      actions_[i] = std::clamp(a, -PDConstants::kClipActions,
                               PDConstants::kClipActions);
    }
    runtime_->setPolicyAction(actions_.data(), PDConstants::kActionDim);
    for (int i = 0; i < frame_skip_; ++i) runtime_->stepFrame();

    // post_physics_step
    ++elapsed_step_;
    const SimRobotState st = runtime_->robotState();
    const PDBase b = ComputeBase(st);

    // _post_physics_step_callback
    if (elapsed_step_ % PDConstants::kResampleSteps == 0) ResampleCommands();
    const PDBase observed = ComputeObservedBase();
    UpdateHeadingCommand(observed);

    // check_termination
    const bool base_contact = runtime_->baseContactForceNorm() > 1.0;
    const bool attitude = std::abs(b.pitch) > 1.0 || std::abs(b.roll) > 0.8;
    const bool finite_state = std::isfinite(st.base_pos[0]) &&
        std::isfinite(st.base_pos[1]) && std::isfinite(st.base_pos[2]) &&
        std::isfinite(b.roll) && std::isfinite(b.pitch) &&
        std::isfinite(b.lin_vel[0]) && std::isfinite(b.lin_vel[1]);
    fall_ = base_contact || attitude || !finite_state;
    time_out_ = elapsed_step_ > PDConstants::kMaxEpisodeLength;
    const bool terminated =
        terminate_when_unhealthy_ && (base_contact || attitude || !finite_state);

    const mjtNum reward = ComputeReward(st, b);

    const bool done = terminated || time_out_;

    // _push_robots for envs not being reset (reset envs are pushed in Reset).
    // The observation uses the pre-push base velocity, as legged_gym's does.
    ComputeObservation(observed);
    if (push_robots_ && !done &&
        elapsed_step_ % PDConstants::kPushInterval == 0) {
      runtime_->setBaseLinearVelocityXY(
          Uniform(-PDConstants::kMaxPushVelXY, PDConstants::kMaxPushVelXY),
          Uniform(-PDConstants::kMaxPushVelXY, PDConstants::kMaxPushVelXY));
    }

    for (int i = 0; i < 12; ++i) last_dof_vel_[i] = st.qd[i];
    last_actions_ = actions_;
    done_ = done;
    WriteState(reward, st);
  }

 private:
  // Body-frame quantities from the simulator's world-frame root state.
  struct PDBase {
    Eigen::Matrix<mjtNum, 3, 3> R;  // body -> world
    Eigen::Matrix<mjtNum, 3, 1> lin_vel, ang_vel, gravity;
    mjtNum roll{0}, pitch{0}, heading{0};
  };

  static mjtNum WrapToPi(mjtNum angle) {
    constexpr mjtNum kTwoPi = static_cast<mjtNum>(6.28318530717958647692);
    angle = std::fmod(angle, kTwoPi);
    if (angle < 0) angle += kTwoPi;             // torch % is non-negative
    if (angle > kTwoPi / 2) angle -= kTwoPi;    // legged_gym wrap_to_pi
    return angle;
  }

  mjtNum Uniform(mjtNum lo, mjtNum hi) {
    return std::uniform_real_distribution<mjtNum>(lo, hi)(gen_);
  }

  template <typename Spec_>
  void Init(const Spec_& spec) {
    obs_noise_ = spec.config["pd_obs_noise"_];
    push_robots_ = spec.config["pd_push_robots"_];
    random_friction_ = spec.config["pd_randomize_friction"_];
    random_ep_len_ = spec.config["pd_init_at_random_ep_len"_];
    random_reset_ = spec.config["pd_randomize_reset"_];
    fixed_command_ = spec.config["pd_fixed_command"_];
    fixed_vx_ = spec.config["pd_command_vx"_];
    fixed_vy_ = spec.config["pd_command_vy"_];
    fixed_heading_ = spec.config["pd_command_heading"_];
    if (!runtime_ || disabled_) return;

    const double dt = frame_skip_ * runtime_->stepPeriodSeconds();
    if (std::abs(dt - PDConstants::kPolicyDt) > 1e-9) {
      throw std::runtime_error(
          "QuadrupedPD: frame_skip * control period = " + std::to_string(dt) +
          " s, but the recipe's policy dt is 0.02 s");
    }
    const auto q_default = runtime_->jointDefaultPosition();
    for (int i = 0; i < PDConstants::kActionDim; ++i) {
      default_q_[i] = static_cast<mjtNum>(q_default[i]);
    }
    // Actuator force limits are not set from code: the Go2 MJCF is the single
    // source of truth for the plant, and already declares the Go2's
    // [23.7, 23.7, 45.43] N.m forcerange for every arm.
    // _process_rigid_shape_props: 64 friction buckets shared by all envs, one
    // bucket per env, fixed for the env's lifetime. The buckets come from the
    // pool seed so every env sees the same 64 values.
    // Randomisation off (evaluation) leaves the simulator's nominal friction,
    // the value every arm is evaluated at, rather than the recipe's 1.0 ground.
    const double ground = runtime_->groundFriction();
    friction_ = ground;
    if (!random_friction_) return;
    friction_ = FrictionRandomisation::Sample(
        ground, static_cast<unsigned>(spec.config["seed"_]), &gen_);
    runtime_->setFriction(friction_);
  }

  void ResampleCommands() {
    if (fixed_command_) {
      commands_[0] = fixed_vx_;
      commands_[1] = fixed_vy_;
      commands_[3] = fixed_heading_;
      return;
    }
    commands_[0] = Uniform(-1.0, 1.0);    // ranges.lin_vel_x
    commands_[1] = Uniform(-1.0, 1.0);    // ranges.lin_vel_y
    commands_[3] = Uniform(-3.14, 3.14);  // ranges.heading
    // "set small commands to zero": kept only if the norm exceeds 0.2.
    if (!(std::hypot(commands_[0], commands_[1]) > 0.2)) {
      commands_[0] = 0.0;
      commands_[1] = 0.0;
    }
  }

  static PDBase ComputeBase(const SimRobotState& st) {
    PDBase b;
    const mjtNum w = st.base_quat_wxyz[0], x = st.base_quat_wxyz[1],
                 y = st.base_quat_wxyz[2], z = st.base_quat_wxyz[3];
    b.R = Eigen::Quaternion<mjtNum>(w, x, y, z).normalized().toRotationMatrix();
    const Eigen::Matrix<mjtNum, 3, 1> v(st.lin_vel_world[0], st.lin_vel_world[1],
                                        st.lin_vel_world[2]);
    const Eigen::Matrix<mjtNum, 3, 1> om(st.ang_vel_world[0], st.ang_vel_world[1],
                                         st.ang_vel_world[2]);
    b.lin_vel = b.R.transpose() * v;       // quat_rotate_inverse
    b.ang_vel = b.R.transpose() * om;
    b.gravity = b.R.transpose() * Eigen::Matrix<mjtNum, 3, 1>(0.0, 0.0, -1.0);
    // isaacgym_utils.get_euler_xyz
    b.roll = std::atan2(2.0 * (w * x + y * z), w * w - x * x - y * y + z * z);
    const mjtNum sinp = 2.0 * (w * y - z * x);
    b.pitch = std::abs(sinp) >= 1.0 ? std::copysign(static_cast<mjtNum>(1.57079632679489661923), sinp)
                                    : std::asin(sinp);
    // _post_physics_step_callback: heading of the body x axis.
    const Eigen::Matrix<mjtNum, 3, 1> fwd =
        b.R * Eigen::Matrix<mjtNum, 3, 1>(1.0, 0.0, 0.0);
    b.heading = std::atan2(fwd[1], fwd[0]);
    return b;
  }

  // Controller-facing base state.  The estimator is fed by MdlSimDriver's
  // noisy IMU and encoder path when cheater_mode=false; do not substitute
  // simulator root state here.  Ground truth remains deliberately confined to
  // reward/termination and the `info:` diagnostics used by the evaluator.
  PDBase ComputeObservedBase() const {
    PDBase b;
    const auto& est = runtime_->stateEstimate();
    const mjtNum w = est.orientation[0], x = est.orientation[1],
                 y = est.orientation[2], z = est.orientation[3];
    const Eigen::Quaternion<mjtNum> q(w, x, y, z);
    b.R = q.normalized().toRotationMatrix();  // body -> world
    b.lin_vel = est.vBody.cast<mjtNum>();
    b.ang_vel = est.omegaBody.cast<mjtNum>();
    b.gravity = b.R.transpose() * Eigen::Matrix<mjtNum, 3, 1>(0.0, 0.0, -1.0);
    b.roll = est.rpy[0];
    b.pitch = est.rpy[1];
    const Eigen::Matrix<mjtNum, 3, 1> fwd =
        b.R * Eigen::Matrix<mjtNum, 3, 1>(1.0, 0.0, 0.0);
    b.heading = std::atan2(fwd[1], fwd[0]);
    return b;
  }

  void UpdateHeadingCommand(const PDBase& b) {
    commands_[2] = std::clamp(0.5 * WrapToPi(commands_[3] - b.heading),
                              static_cast<mjtNum>(-1.0),
                              static_cast<mjtNum>(1.0));
  }

  // compute_observations(), noise and clip included.
  void ComputeObservation(const PDBase& b) {
    int k = 0;
    auto put = [&](mjtNum value, mjtNum noise) {
      if (obs_noise_ && noise > 0.0) value += (2.0 * Uniform(0.0, 1.0) - 1.0) * noise;
      obs_[k++] = std::clamp(value, -PDConstants::kClipObservations,
                             PDConstants::kClipObservations);
    };
    // noise_scales x noise_level 1.0 x obs_scales, per _get_noise_scale_vec.
    for (int i = 0; i < 3; ++i) put(b.lin_vel[i] * 2.0, 0.1 * 2.0);
    for (int i = 0; i < 3; ++i) put(b.ang_vel[i] * 0.25, 0.2 * 0.25);
    for (int i = 0; i < 3; ++i) put(b.gravity[i], 0.05);
    put(commands_[0] * 2.0, 0.0);
    put(commands_[1] * 2.0, 0.0);
    put(commands_[2] * 0.25, 0.0);
    // MotorHW measurements have already received the configured encoder noise
    // in MdlSimDriver.  Keeping the recipe's explicit observation noise on
    // top preserves the Rudin training distribution without leaking truth.
    const auto* legs = runtime_->legController();
    for (int i = 0; i < 12; ++i) {
      const int leg = i / 3, joint = i % 3;
      const mjtNum q = legs ? legs->datas[leg].q[joint] : 0.0;
      put(q - default_q_[i], 0.01);
    }
    for (int i = 0; i < 12; ++i) {
      const int leg = i / 3, joint = i % 3;
      const mjtNum qd = legs ? legs->datas[leg].qd[joint] : 0.0;
      put(qd * 0.05, 1.5 * 0.05);
    }
    for (int i = 0; i < 12; ++i) put(actions_[i], 0.0);
  }

  // compute_reward(): each scale is pre-multiplied by dt, the total is clipped
  // at zero (only_positive_rewards), termination's scale is 0 and dropped.
  mjtNum ComputeReward(const SimRobotState& st, const PDBase& b) {
    const mjtNum dt = PDConstants::kPolicyDt;
    const mjtNum lin_err = std::pow(commands_[0] - b.lin_vel[0], 2) +
                           std::pow(commands_[1] - b.lin_vel[1], 2);
    const mjtNum yaw_err = std::pow(commands_[2] - b.ang_vel[2], 2);

    mjtNum torques = 0.0, dof_acc = 0.0, dof_limits = 0.0, action_rate = 0.0;
    for (int axis = 0; axis < 12; ++axis) {
      const int leg = axis / 3, joint = axis % 3;
      torques += st.tau[axis] * st.tau[axis];
      const mjtNum acc = (last_dof_vel_[axis] - st.qd[axis]) / dt;
      dof_acc += acc * acc;
      action_rate += std::pow(last_actions_[axis] - actions_[axis], 2);
      // _process_dof_props soft limits, evaluated in URDF convention.
      const double* lim = joint == 0   ? PDConstants::kHipLimits
                          : joint == 2 ? PDConstants::kCalfLimits
                          : leg < 2    ? PDConstants::kFrontThighLimits
                                       : PDConstants::kRearThighLimits;
      const mjtNum m = 0.5 * (lim[0] + lim[1]);
      const mjtNum r = lim[1] - lim[0];
      const mjtNum lo = m - 0.5 * r * PDConstants::kSoftDofPosLimit;
      const mjtNum hi = m + 0.5 * r * PDConstants::kSoftDofPosLimit;
      const mjtNum q_urdf = PDConstants::kUrdfSign[joint] * st.q[axis];
      dof_limits += std::max(lo - q_urdf, static_cast<mjtNum>(0.0));
      dof_limits += std::max(q_urdf - hi, static_cast<mjtNum>(0.0));
    }

    // _reward_feet_air_time: contact is the foot's vertical force > 1 N.
    const auto contacts = runtime_->footContactState(1.0f);
    mjtNum air = 0.0;
    for (int leg = 0; leg < 4; ++leg) {
      const bool contact = contacts[leg] > 0.5f;
      const bool contact_filt = contact || last_contacts_[leg];
      last_contacts_[leg] = contact;
      const bool first_contact = feet_air_time_[leg] > 0.0 && contact_filt;
      feet_air_time_[leg] += dt;
      if (first_contact) air += feet_air_time_[leg] - 0.5;
      if (contact_filt) feet_air_time_[leg] = 0.0;
    }
    if (!(std::hypot(commands_[0], commands_[1]) > 0.1)) air = 0.0;

    const mjtNum collision =
        static_cast<mjtNum>(runtime_->penalizedCollisionCount());

    // Order matches info:reward_terms, documented in the trainer.
    const std::array<mjtNum, PDConstants::kRewardTermDim> raw{
        std::exp(-lin_err / PDConstants::kTrackingSigma) * 1.0,   // tracking_lin_vel
        std::exp(-yaw_err / PDConstants::kTrackingSigma) * 0.5,   // tracking_ang_vel
        b.lin_vel[2] * b.lin_vel[2] * -2.0,                       // lin_vel_z
        (b.ang_vel[0] * b.ang_vel[0] + b.ang_vel[1] * b.ang_vel[1]) * -0.05,
        torques * -0.0002,                                        // torques (GO2)
        dof_acc * -2.5e-7,                                        // dof_acc
        air * 1.0,                                                // feet_air_time
        collision * -1.0,                                         // collision
        action_rate * -0.01,                                      // action_rate
        dof_limits * -10.0};                                      // dof_pos_limits (GO2)
    mjtNum total = 0.0;
    for (int i = 0; i < PDConstants::kRewardTermDim; ++i) {
      terms_[i] = raw[i] * dt;
      total += terms_[i];
    }
    return std::max(total, static_cast<mjtNum>(0.0));
  }

  void WriteState(mjtNum reward, const SimRobotState& st) {
    State state = Allocate();
    state["reward"_] = static_cast<float>(reward);
    auto obs_array = state["obs"_];
    obs_array.Assign(obs_.data(), PDConstants::kObservationDim);
    state["info:time_out"_] = time_out_ ? 1 : 0;
    state["info:fall"_] = fall_ ? 1 : 0;
    auto terms = state["info:reward_terms"_];
    auto* term_data = static_cast<mjtNum*>(terms.Data());
    for (int i = 0; i < PDConstants::kRewardTermDim; ++i) term_data[i] = terms_[i];
    const SimForceState force = runtime_->forceState();
    state["info:sim_time"_] = force.time;
    state["info:force_onset"_] = force.onset;
    state["info:force_phase"_] = force.phase;
    state["info:force_direction"_] = force.direction;
    state["info:force_requested_impulse"_] = force.requested_impulse;
    auto applied = state["info:force_applied"_];
    auto* force_data = static_cast<mjtNum*>(applied.Data());
    for (int i = 0; i < 3; ++i) force_data[i] = force.applied[i];
    state["info:body_height"_] = st.base_pos[2];
    state["info:reward_linvel"_] = terms_[0];
    state["info:reward_quadctrl"_] = terms_[4];
    state["info:reward_alive"_] = 0.0;
    state["info:reward_impact"_] = terms_[7];
    state["info:x_position"_] = st.base_pos[0];
    state["info:y_position"_] = st.base_pos[1];
    state["info:distance_from_origin"_] = std::hypot(st.base_pos[0], st.base_pos[1]);
    state["info:x_velocity"_] = st.lin_vel_world[0];
    state["info:y_velocity"_] = st.lin_vel_world[1];
  }
};

using QuadrupedPDEnvPool = AsyncEnvPool<QuadrupedPDEnv>;

}  // namespace mujoco_gym

#endif  // ENVPOOL_MUJOCO_GYM_QUADRUPED_PD_H_
