#ifndef LEGGED_GAIT_SCHEDULER_H
#define LEGGED_GAIT_SCHEDULER_H

#include <algorithm>
#include <array>
#include <cmath>

#include <Controllers/FootSwingTrajectory.h>

// Continuous-time gait helper that tracks a normalized phase (theta in [0, 1))
// and per-leg contact/swing progress. Offsets and durations are provided as
// fractions (0..1) of the gait cycle.
class ContinuousTimeGait {
 public:
  static constexpr int kNumLegs = 4;

  ContinuousTimeGait(const std::array<float, kNumLegs>& offsets,
                     const std::array<float, kNumLegs>& contact_durations) {
    setPattern(offsets, contact_durations);
  }

  void reset(float theta = 0.0f) { theta_ = wrapUnitInterval(theta); }

  void setPattern(const std::array<float, kNumLegs>& offsets,
                  const std::array<float, kNumLegs>& contact_durations) {
    for (int leg = 0; leg < kNumLegs; ++leg) {
      phase_offsets_[leg] = wrapUnitInterval(offsets[leg]);
      contact_durations_[leg] = std::clamp(contact_durations[leg], 0.0f, 1.0f);
      swing_durations_[leg] = 1.0f - contact_durations_[leg];
    }
  }

  void setPhase(float theta) { theta_ = wrapUnitInterval(theta); }

  void advancePhase(float delta_theta) {
    theta_ = wrapUnitInterval(theta_ + delta_theta);
  }

  float phase() const { return theta_; }

  float dutyFraction(int leg) const { return contact_durations_[leg]; }
  float swingFraction(int leg) const { return swing_durations_[leg]; }
  bool isLegInContact(int leg) const {
    const float duration = contact_durations_[leg];
    if (duration <= 1e-5f) {
      return false;
    }
    const float shifted = wrapUnitInterval(theta_ - phase_offsets_[leg]);
    return shifted < duration;
  }

  std::array<float, kNumLegs> contactPhase() const {
    std::array<float, kNumLegs> phases{};
    for (int leg = 0; leg < kNumLegs; ++leg) {
      const float duration = contact_durations_[leg];
      float progress = 0.0f;
      if (duration > 1e-5f) {
        const float shifted = wrapUnitInterval(theta_ - phase_offsets_[leg]);
        if (shifted < duration) {
          progress = shifted / duration;
        }
      }
      phases[leg] = progress;
    }
    return phases;
  }

  std::array<float, kNumLegs> swingPhase() const {
    std::array<float, kNumLegs> phases{};
    for (int leg = 0; leg < kNumLegs; ++leg) {
      const float duration = swing_durations_[leg];
      const float swing_offset =
          wrapUnitInterval(phase_offsets_[leg] + contact_durations_[leg]);
      float progress = 0.0f;
      if (duration > 1e-5f) {
        const float shifted = wrapUnitInterval(theta_ - swing_offset);
        if (shifted < duration) {
          progress = shifted / duration;
        }
      }
      phases[leg] = progress;
    }
    return phases;
  }

 private:
  static float wrapUnitInterval(float value) {
    float wrapped = std::fmod(value, 1.0f);
    if (wrapped < 0.0f) {
      wrapped += 1.0f;
    }
    return wrapped;
  }

  std::array<float, kNumLegs> phase_offsets_{{0.0f, 0.0f, 0.0f, 0.0f}};
  std::array<float, kNumLegs> contact_durations_{{0.5f, 0.5f, 0.5f, 0.5f}};
  std::array<float, kNumLegs> swing_durations_{{0.5f, 0.5f, 0.5f, 0.5f}};
  float theta_{0.0f};
};

// Lightweight gait scheduler used by the Mujoco model-based controller.
// Provides contact states, swing phases and swing trajectories for each leg.
class LeggedGaitScheduler {
 public:
  static constexpr int kNumLegs = 4;

  explicit LeggedGaitScheduler(float control_dt)
      : control_dt_(control_dt),
        gait_(makeArray4f(0.0f, 0.5f, 0.5f, 0.0f),
              makeArray4f(0.5f, 0.5f, 0.5f, 0.5f)) {
    reset();
  }

  void reset() {
    cycle_time_seconds_ = std::max(control_dt_ * 40.0f, 1e-4f);
    gait_.reset();
    updateStates();
    for (auto& traj : swing_traj_) {
      traj.setHeight(0.10f);
      traj.setInitialPosition(::Vec3<float>::Zero());
      traj.setFinalPosition(::Vec3<float>::Zero());
    }
  }

  // Advance the gait phase directly by delta_theta (normalized, 0..1 wraps).
  void update(float delta_theta) {
    const float dt = std::max(control_dt_, 1e-6f);
    const float safe_delta = std::max(delta_theta, 1e-5f);
    // Estimate cycle time from phase rate and clamp to reasonable bounds.
    cycle_time_seconds_ = std::clamp(dt / safe_delta, 0.02f, 2.0f);
    gait_.advancePhase(delta_theta);
    updateStates();
  }

  void setPhase(float theta) { gait_.setPhase(theta); updateStates(); }
  void setPattern(const std::array<float, kNumLegs>& offsets,
                  const std::array<float, kNumLegs>& contact_durations) {
    gait_.setPattern(offsets, contact_durations);
    updateStates();
  }

  const std::array<float, kNumLegs>& contactState() const { return contact_state_; }
  const std::array<float, kNumLegs>& swingPhase() const { return swing_phase_; }
  const std::array<float, kNumLegs>& contactPhase() const { return contact_phase_; }

  float phase() const { return gait_.phase(); }

  FootSwingTrajectory<float>& swingTrajectory(int leg) {
    return swing_traj_[leg];
  }

  float stanceDurationSeconds(int leg) const {
    return gait_.dutyFraction(leg) * cycle_time_seconds_;
  }

  float swingDurationSeconds(int leg) const {
    return gait_.swingFraction(leg) * cycle_time_seconds_;
  }

  float cycleTimeSeconds() const { return cycle_time_seconds_; }

 private:
  static std::array<float, kNumLegs> makeArray4f(float a, float b, float c, float d) {
    return std::array<float, kNumLegs>{{a, b, c, d}};
  }

  void updateStates() {
    const auto contact = gait_.contactPhase();
    const auto swing = gait_.swingPhase();
    for (int i = 0; i < kNumLegs; ++i) {
      contact_phase_[i] = contact[i];
      contact_state_[i] = gait_.isLegInContact(i) ? 1.0f : 0.0f;
      swing_phase_[i] = swing[i];
    }
  }

  float control_dt_{0.002f};
  ContinuousTimeGait gait_;
  float cycle_time_seconds_{0.08f};
  std::array<float, kNumLegs> contact_state_{{1.f, 1.f, 1.f, 1.f}};
  std::array<float, kNumLegs> contact_phase_{{0.0f, 0.0f, 0.0f, 0.0f}};
  std::array<float, kNumLegs> swing_phase_{{0.f, 0.f, 0.f, 0.f}};
  std::array<FootSwingTrajectory<float>, kNumLegs> swing_traj_{};
};

#endif  // LEGGED_GAIT_SCHEDULER_H
