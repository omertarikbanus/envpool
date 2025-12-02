#ifndef LEGGED_GAIT_SCHEDULER_H
#define LEGGED_GAIT_SCHEDULER_H

#include <array>
#include <algorithm>

#include <Controllers/FootSwingTrajectory.h>
#include <Controllers/convexMPC/Gait.h>

// Lightweight gait scheduler used by the Mujoco model-based controller.
// Provides contact states, swing phases and swing trajectories for each leg.
class LeggedGaitScheduler {
 public:
  static constexpr int kNumLegs = 4;

  LeggedGaitScheduler()
      : gait_(40, makeVec4i(0, 20, 20, 0), makeVec4i(20, 20, 20, 20), "Trotting") {
    reset();
  }

  void reset() {
    iteration_counter_ = 0;
    iterations_between_mpc_ = 1;
    gait_.setIterations(iterations_between_mpc_, iteration_counter_);
    updateStates();
    for (auto& traj : swing_traj_) {
      traj.setHeight(0.10f);
      traj.setInitialPosition(::Vec3<float>::Zero());
      traj.setFinalPosition(::Vec3<float>::Zero());
    }
  }

  void update(int iterations_between_mpc, int current_iteration) {
    iterations_between_mpc_ = std::max(iterations_between_mpc, 1);
    iteration_counter_ = current_iteration;
    gait_.setIterations(iterations_between_mpc_, iteration_counter_);
    updateStates();
  }

  const std::array<float, kNumLegs>& contactState() const { return contact_state_; }
  const std::array<float, kNumLegs>& swingPhase() const { return swing_phase_; }

  FootSwingTrajectory<float>& swingTrajectory(int leg) {
    return swing_traj_[leg];
  }

  float stanceDurationSeconds(int leg, float control_dt) {
    const float dt_mpc = control_dt * static_cast<float>(iterations_between_mpc_);
    return gait_.getCurrentStanceTime(dt_mpc, leg);
  }

  float swingDurationSeconds(int leg, float control_dt) {
    const float dt_mpc = control_dt * static_cast<float>(iterations_between_mpc_);
    return gait_.getCurrentSwingTime(dt_mpc, leg);
  }

  int iterationsBetweenMpc() const { return iterations_between_mpc_; }

 private:
  static ::Vec4<int> makeVec4i(int a, int b, int c, int d) {
    ::Vec4<int> v;
    v << a, b, c, d;
    return v;
  }

  void updateStates() {
    const ::Vec4<float> contact = gait_.getContactState();
    const ::Vec4<float> swing = gait_.getSwingState();
    for (int i = 0; i < kNumLegs; ++i) {
      contact_state_[i] = (contact[i] > 0.f) ? 1.f : 0.f;
      swing_phase_[i] = swing[i];
    }
  }

  OffsetDurationGait gait_;
  int iteration_counter_{0};
  int iterations_between_mpc_{1};
  std::array<float, kNumLegs> contact_state_{{1.f, 1.f, 1.f, 1.f}};
  std::array<float, kNumLegs> swing_phase_{{0.f, 0.f, 0.f, 0.f}};
  std::array<FootSwingTrajectory<float>, kNumLegs> swing_traj_{};
};

#endif  // LEGGED_GAIT_SCHEDULER_H
