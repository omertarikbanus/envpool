#ifndef RL_GAIT_H
#define RL_GAIT_H

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

#include <Eigen/Core>

namespace rl_gait {

inline float clamp01(float value) {
  return std::clamp(value, 0.0f, 1.0f);
}

inline float cubicHermite(float y0, float yf, float phase) {
  const float s = clamp01(phase);
  return y0 + (yf - y0) * (3.0f * s * s - 2.0f * s * s * s);
}

inline float cubicHermiteFirstDerivative(float y0, float yf, float phase) {
  const float s = clamp01(phase);
  return (yf - y0) * (6.0f * s - 6.0f * s * s);
}

inline float cubicHermiteSecondDerivative(float y0, float yf, float phase) {
  const float s = clamp01(phase);
  return (yf - y0) * (6.0f - 12.0f * s);
}

inline Eigen::Vector3f cubicHermite(const Eigen::Vector3f& y0,
                                    const Eigen::Vector3f& yf,
                                    float phase) {
  const float s = clamp01(phase);
  const Eigen::Vector3f delta = yf - y0;
  const float blend = 3.0f * s * s - 2.0f * s * s * s;
  return y0 + delta * blend;
}

inline Eigen::Vector3f cubicHermiteFirstDerivative(
    const Eigen::Vector3f& y0, const Eigen::Vector3f& yf, float phase) {
  const float s = clamp01(phase);
  const Eigen::Vector3f delta = yf - y0;
  const float blend = 6.0f * s - 6.0f * s * s;
  return delta * blend;
}

inline Eigen::Vector3f cubicHermiteSecondDerivative(
    const Eigen::Vector3f& y0, const Eigen::Vector3f& yf, float phase) {
  const float s = clamp01(phase);
  const Eigen::Vector3f delta = yf - y0;
  const float blend = 6.0f - 12.0f * s;
  return delta * blend;
}

}  // namespace rl_gait

class RLGait {
 public:
  static constexpr int kNumLegs = 4;

  struct ContactProfile {
    int segments{20};
    Eigen::Array<int, kNumLegs, 1> offsets;
    Eigen::Array<int, kNumLegs, 1> durations;
  };

  class SwingTrajectory {
   public:
    SwingTrajectory() = default;

    void setInitialPosition(const Eigen::Vector3f& p0) { p0_ = p0; }
    void setFinalPosition(const Eigen::Vector3f& pf) { pf_ = pf; }
    void setHeight(float h) { height_ = std::max(h, 0.0f); }

    void compute(float phase, float swing_time) {
      const float clamped_phase = rl_gait::clamp01(phase);
      const float clamped_time = std::max(swing_time, 1e-3f);

      position_ = rl_gait::cubicHermite(p0_, pf_, clamped_phase);
      velocity_ = rl_gait::cubicHermiteFirstDerivative(p0_, pf_, clamped_phase) /
                  clamped_time;
      acceleration_ =
          rl_gait::cubicHermiteSecondDerivative(p0_, pf_, clamped_phase) /
          (clamped_time * clamped_time);

      const float half_phase =
          (clamped_phase < 0.5f) ? clamped_phase * 2.0f
                                 : (clamped_phase - 0.5f) * 2.0f;
      const float phase_scale = (clamped_phase < 0.5f) ? 2.0f : 2.0f;

      const float start_z =
          (clamped_phase < 0.5f) ? p0_[2] : p0_[2] + height_;
      const float end_z =
          (clamped_phase < 0.5f) ? p0_[2] + height_ : pf_[2];

      const float z_pos =
          rl_gait::cubicHermite(start_z, end_z, half_phase);
      const float z_vel = rl_gait::cubicHermiteFirstDerivative(start_z, end_z,
                                                               half_phase) *
                          phase_scale / clamped_time;
      const float z_acc =
          rl_gait::cubicHermiteSecondDerivative(start_z, end_z, half_phase) *
          (phase_scale * phase_scale) / (clamped_time * clamped_time);

      position_[2] = z_pos;
      velocity_[2] = z_vel;
      acceleration_[2] = z_acc;
    }

    const Eigen::Vector3f& position() const { return position_; }
    const Eigen::Vector3f& velocity() const { return velocity_; }
    const Eigen::Vector3f& acceleration() const { return acceleration_; }

   private:
    Eigen::Vector3f p0_{Eigen::Vector3f::Zero()};
    Eigen::Vector3f pf_{Eigen::Vector3f::Zero()};
    Eigen::Vector3f position_{Eigen::Vector3f::Zero()};
    Eigen::Vector3f velocity_{Eigen::Vector3f::Zero()};
    Eigen::Vector3f acceleration_{Eigen::Vector3f::Zero()};
    float height_{0.05f};
  };

  RLGait() { reset(); }
  explicit RLGait(const ContactProfile& profile) : profile_(profile) { reset(); }

  void reset() {
    frame_skip_ = 1;
    std::fill(contact_state_.begin(), contact_state_.end(), 1.0f);
    std::fill(stance_phase_.begin(), stance_phase_.end(), 0.0f);
    std::fill(swing_phase_.begin(), swing_phase_.end(), 0.0f);
  }

  void setContactProfile(const ContactProfile& profile) {
    profile_ = profile;
    if (profile_.segments <= 0) {
      profile_.segments = 1;
    }
    for (int i = 0; i < kNumLegs; ++i) {
      if (profile_.durations[i] < 0) {
        profile_.durations[i] = 0;
      } else if (profile_.durations[i] > profile_.segments) {
        profile_.durations[i] = profile_.segments;
      }
      profile_.offsets[i] =
          (profile_.offsets[i] % profile_.segments + profile_.segments) %
          profile_.segments;
    }
  }

  void update(int frame_skip, int elapsed_step) {
    frame_skip_ = std::max(frame_skip, 1);
    if (profile_.segments <= 0) {
      profile_.segments = 1;
    }

    const int cycle_frames = frame_skip_ * profile_.segments;
    const int step_in_cycle =
        (cycle_frames > 0) ? (elapsed_step % cycle_frames) : 0;
    const float normalized_phase =
        (cycle_frames > 0) ? static_cast<float>(step_in_cycle) /
                                 static_cast<float>(cycle_frames)
                           : 0.0f;

    updatePhases(normalized_phase);
  }

  const std::array<float, kNumLegs>& contactState() const {
    return contact_state_;
  }

  const std::array<float, kNumLegs>& stancePhase() const {
    return stance_phase_;
  }

  const std::array<float, kNumLegs>& swingPhase() const {
    return swing_phase_;
  }

  float dutyFraction(int leg) const {
    const float denom = static_cast<float>(std::max(profile_.segments, 1));
    const float raw =
        static_cast<float>(profile_.durations[leg]) / std::max(1.0f, denom);
    return std::clamp(raw, 0.0f, 1.0f);
  }

  float cycleTimeSeconds(float control_dt) const {
    const int segments = std::max(profile_.segments, 1);
    return static_cast<float>(frame_skip_) * static_cast<float>(segments) *
           control_dt;
  }

  float stanceDurationSeconds(int leg, float control_dt) const {
    return dutyFraction(leg) * cycleTimeSeconds(control_dt);
  }

  float swingDurationSeconds(int leg, float control_dt) const {
    return (1.0f - dutyFraction(leg)) * cycleTimeSeconds(control_dt);
  }

  bool isLegInContact(int leg) const {
    if (leg < 0 || leg >= kNumLegs) {
      return false;
    }
    return contact_state_[leg] > 0.5f;
  }

  SwingTrajectory& swingTrajectory(int leg) { return swing_traj_[leg]; }
  const SwingTrajectory& swingTrajectory(int leg) const {
    return swing_traj_[leg];
  }

 private:
  void updatePhases(float phase) {
    const float total_segments =
        static_cast<float>(std::max(profile_.segments, 1));

    for (int leg = 0; leg < kNumLegs; ++leg) {
      const float offset =
          static_cast<float>(profile_.offsets[leg]) / total_segments;
      const float duration =
          static_cast<float>(profile_.durations[leg]) / total_segments;

      float stance_progress = phase - offset;
      if (stance_progress < 0.0f) {
        stance_progress += 1.0f;
      }

      if (stance_progress < duration && duration > 1e-3f) {
        contact_state_[leg] = 1.0f;
        stance_phase_[leg] = stance_progress / duration;
      } else {
        contact_state_[leg] = 0.0f;
        stance_phase_[leg] = 0.0f;
      }

      float swing_offset = offset + duration;
      if (swing_offset >= 1.0f) {
        swing_offset -= 1.0f;
      }

      const float swing_duration = std::max(1.0f - duration, 1e-3f);
      float swing_progress = phase - swing_offset;
      if (swing_progress < 0.0f) {
        swing_progress += 1.0f;
      }

      if (contact_state_[leg] < 0.5f) {
        swing_phase_[leg] = swing_progress / swing_duration;
      } else {
        swing_phase_[leg] = 0.0f;
      }
    }
  }

  ContactProfile profile_{60,
                          (Eigen::Array<int, kNumLegs, 1>() << 0, 30, 30, 0)
                              .finished(),
                          (Eigen::Array<int, kNumLegs, 1>() << 30, 30, 30, 30)
                              .finished()};
  int frame_skip_{1};
  std::array<float, kNumLegs> contact_state_{{1.0f, 1.0f, 1.0f, 1.0f}};
  std::array<float, kNumLegs> stance_phase_{{0.0f, 0.0f, 0.0f, 0.0f}};
  std::array<float, kNumLegs> swing_phase_{{0.0f, 0.0f, 0.0f, 0.0f}};
  std::array<SwingTrajectory, kNumLegs> swing_traj_{};
};

#endif  // RL_GAIT_H
