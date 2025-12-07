#ifndef FOOTSTEP_PLANNER_H
#define FOOTSTEP_PLANNER_H

#include <algorithm>
#include <array>

#include <Controllers/LegController.h>
#include <FSM/ControlFSMData.h>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

#include "legged_gait_scheduler.h"

class FootstepPlanner {
 public:
  static constexpr int kNumLegs = LeggedGaitScheduler::kNumLegs;

  explicit FootstepPlanner(LeggedGaitScheduler& scheduler)
      : gait_scheduler_(scheduler) {}

  struct PlanInput {
    const std::array<float, kNumLegs>& contact_schedule;
    const std::array<float, kNumLegs>& previous_contact_schedule;
    const std::array<float, kNumLegs>& swing_phase;
    const Eigen::Vector3f& base_position;
    const Eigen::Vector3f& base_velocity_world;
    const Eigen::Matrix3f& rotation_world_from_body;
    const Eigen::Vector3f& commanded_velocity_world;
    float yaw_rate_des;
    float cmpc_bonus;
    float gravity;
    LegController<float>& leg_controller;
    std::vector<mjtNum> current_action_;
  };

  float Plan(const PlanInput& input, LocomotionCtrlData<float>& loco) {
    float accum_foot_des_z = 0.0f;
    for (int leg = 0; leg < kNumLegs; ++leg) {
      const auto& leg_data = input.leg_controller.datas[leg];
      Eigen::Vector3f foot_local(static_cast<float>(leg_data.p[0]),
                                 static_cast<float>(leg_data.p[1]),
                                 static_cast<float>(leg_data.p[2]));
      Eigen::Vector3f hip = Eigen::Vector3f::Zero();
      if (leg_data.quadruped) {
        const auto hip_loc = leg_data.quadruped->getHipLocation(leg);
        hip[0] = hip_loc[0];
        hip[1] = hip_loc[1];
        hip[2] = hip_loc[2];
      }
      const Eigen::Vector3f foot_body = hip + foot_local;
      const Eigen::Vector3f foot_world =
          input.base_position + input.rotation_world_from_body * foot_body;

      const bool was_in_contact = input.previous_contact_schedule[leg] > 0.0f;
      const bool leg_in_contact = input.contact_schedule[leg] > 0.0f;
      if (was_in_contact && !leg_in_contact) {
        auto& swing_traj = gait_scheduler_.swingTrajectory(leg);
        swing_traj.setInitialPosition(foot_world);
        swing_traj.setHeight(0.10f);
      }

      if (!leg_in_contact) {
        auto& swing_traj = gait_scheduler_.swingTrajectory(leg);
        const float stance_duration =
            gait_scheduler_.stanceDurationSeconds(leg);
        const float swing_duration = std::max(
            gait_scheduler_.swingDurationSeconds(leg), 1e-3f);
        const float clamped_phase =
            std::clamp(input.swing_phase[leg], 0.0f, 1.0f);
        const float swing_time_remaining =
            std::max((1.0f - clamped_phase) * swing_duration, 0.0f);
        const Eigen::AngleAxisf yaw_correction(
            -input.yaw_rate_des * stance_duration * 0.5f,
            Eigen::Vector3f::UnitZ());
        const Eigen::Vector3f p_yaw_corrected = yaw_correction * hip;
        Eigen::Vector3f commanded_velocity_body =
            input.rotation_world_from_body.transpose() *
            input.commanded_velocity_world;
        commanded_velocity_body[2] = 0.0f;
        Eigen::Vector3f foot_target_world =
            input.base_position +
            input.rotation_world_from_body *
                (p_yaw_corrected +
                 commanded_velocity_body * swing_time_remaining);
        constexpr float kPRelMax = 0.15f;
        float pfx_rel =
            input.base_velocity_world[0] * (0.5f + input.cmpc_bonus) *
          stance_duration +
            0.03f * (input.base_velocity_world[0] -
               input.commanded_velocity_world[0]) +
            (0.5f * input.base_position[2] / input.gravity) *
          (input.base_velocity_world[1] * input.yaw_rate_des);
        const float side_sign = Quadruped<float>::getSideSign(leg);
        float pfy_rel =
            input.base_velocity_world[1] * 0.5f * stance_duration +
            0.03f * (input.base_velocity_world[1] -
               input.commanded_velocity_world[1]);
        // Mirror the yaw-coupled lateral placement so left/right feet are
        // symmetric per Raibert's rule instead of all shifting the same way.
        pfy_rel += side_sign *
          ((0.5f * input.base_position[2] / input.gravity) *
           (-input.base_velocity_world[0] * input.yaw_rate_des));
        pfx_rel = std::clamp(pfx_rel, -kPRelMax, kPRelMax) + input.current_action_[3 + leg * 5 + 3] * 0.50f;
        pfy_rel = std::clamp(pfy_rel, -kPRelMax, kPRelMax) + input.current_action_[3 + leg * 5 + 4] * 0.50f;
        float side_offset = side_sign * 0.05f;

        foot_target_world[0] += pfx_rel;
        foot_target_world[1] += pfy_rel + side_offset;
        foot_target_world[2] = -0.003f;

        swing_traj.setFinalPosition(foot_target_world);
        foot_target_world.setZero(); // avoid reusing
        swing_traj.setHeight(0.10f);
        swing_traj.computeSwingTrajectoryBezier(clamped_phase, swing_duration);

        const ::Vec3<float>& pos = swing_traj.getPosition();
        const ::Vec3<float>& vel = swing_traj.getVelocity();
        const ::Vec3<float>& acc = swing_traj.getAcceleration();
        for (int axis = 0; axis < 3; ++axis) {
          loco.pFoot_des[leg][axis] = pos[axis];
          loco.vFoot_des[leg][axis] = vel[axis];
          loco.aFoot_des[leg][axis] = acc[axis];
        }
        accum_foot_des_z += pos[2];
      } else {
        for (int axis = 0; axis < 3; ++axis) {
          loco.pFoot_des[leg][axis] = foot_world[axis];
          loco.vFoot_des[leg][axis] = 0.0f;
          loco.aFoot_des[leg][axis] = 0.0f;
        }
        accum_foot_des_z += foot_world[2];
      }
      
    }
    return accum_foot_des_z;
  }

 private:
  LeggedGaitScheduler& gait_scheduler_;
};

#endif  // FOOTSTEP_PLANNER_H
