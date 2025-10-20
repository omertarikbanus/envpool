#ifndef MBC_INTERFACE_H
#define MBC_INTERFACE_H

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <deque>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <mujoco/mujoco.h>

#include <iostream>
#include <RobotController.h>
#include <RobotRunner.h>
#include <Utilities/RobotCommands.h>

#include <Controllers/EmbeddedController.hpp>
#include <Controllers/WBC_Ctrl/WBC_Ctrl.hpp>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

#include "rl_gait.h"

class ModelBasedControllerInterface {
 public:
  static constexpr int kNumLegs = 4;
  static constexpr int kFootActionDim = 3;
  static constexpr int kActionDim = 15;
  static constexpr int kHistoryLength = 4;
  static constexpr int kBodyCoreDim = 15;
  static constexpr int kHeightDim = 2;
  static constexpr int kBodyHistoryDim = 6;  // linear velocity (3) + angular velocity (3)
  static constexpr int kFootCtrlStateDim = 6;      // commanded foot position (3) + velocity (3)
  static constexpr int kFootMeasuredStateDim = 9;  // measured position (3) + velocity (3) + GRF (3)
  static constexpr int kFootObsPerLeg =
      1 + kFootCtrlStateDim + kFootMeasuredStateDim;  // contact + ctrl state + measured state
  static constexpr int kObservationDim =
      kBodyCoreDim + kHeightDim + kNumLegs * kFootObsPerLeg +
      kActionDim * (1 + kHistoryLength) + kBodyHistoryDim * kHistoryLength;

  ModelBasedControllerInterface() {
    reset();
    // Constructor implementation
  }
    float side_sign[4] = {-1, 1, -1, 1};
    float reset_body_height_{0.35f};
    float filtered_body_height_{0.35f};
    int cheater_mode{1};
    const mjModel* model_{nullptr};
    std::array<int, kNumLegs> foot_body_ids_{-1, -1, -1, -1};
    std::array<int, kNumLegs> foot_geom_ids_{-1, -1, -1, -1};
    std::vector<mjtNum> current_action_;
    std::deque<std::vector<mjtNum>> action_history_;
    std::deque<std::array<mjtNum, kBodyHistoryDim>> body_history_;
    RLGait gait_scheduler_;
    std::array<float, kNumLegs> last_contact_schedule_{{1.0f, 1.0f, 1.0f, 1.0f}};

  ~ModelBasedControllerInterface() = default;

  void reset() {
    reset_body_height_ = 0.35f;
    filtered_body_height_ = reset_body_height_;
    last_feedback_data_ = nullptr;
    clearObservationHistory();
    gait_scheduler_.reset();
    last_contact_schedule_.fill(1.0f);
    _controller = new EmbeddedController();
    _actuator_command = new SpiCommand();
    _actuator_data = new SpiData();
    _imu_data = new VectorNavData();
    _gamepad_command = new GamepadCommand();
    // _control_params = new RobotControlParameters();
    _periodic_task_manager = new PeriodicTaskManager();

    _robot_runner = new RobotRunner(_controller, _periodic_task_manager, 0.002,
                                    "robot-control");

    _robot_runner->driverCommand = _gamepad_command;
    _robot_runner->_ImuData = _imu_data;
    _robot_runner->_Feedback = _actuator_data;
    _robot_runner->_Command = _actuator_command;
    // _robot_runner->controlParameters = _control_params;
    _robot_runner->initializeParameters();
    _robot_runner->init();
    // setModeStandUp();
    // setModeLocomotion();
  }
  void construct()
  {
    _controller->_controlFSM->data.controlParameters->control_mode = 1;

    while (_controller->_controlFSM->currentState->stateName !=
           FSM_StateName::BALANCE_STAND) {
      _robot_runner->run();
    }
    
  }
  void setModeStandUp() {
    _controller->_controlFSM->data.controlParameters->control_mode = 1;

    while (_controller->_controlFSM->currentState->stateName !=
           FSM_StateName::BALANCE_STAND) {
      _robot_runner->run();
    }
    
  }
  void setModeLocomotion() {
    _robot_runner->initializeStateEstimator();
    _controller->_controlFSM->data.controlParameters->control_mode = 4;
    while (_controller->_controlFSM->currentState->stateName !=
           FSM_StateName::LOCOMOTION) {
      _robot_runner->run();
    }
    filtered_body_height_ = reset_body_height_;
    _controller->_controlFSM->data.locomotionCtrlData.pBody_des[2] = filtered_body_height_;
    gait_scheduler_.reset();
    last_contact_schedule_.fill(1.0f);
  }

  void setBaseHeight(float height) {
    reset_body_height_ = height;
    filtered_body_height_ = height;
    if (_controller && _controller->_controlFSM) {
      _controller->_controlFSM->data.locomotionCtrlData.pBody_des[2] = filtered_body_height_;
    }
  }

  void setModel(const mjModel* model) {
    model_ = model;
    static constexpr std::array<const char*, kNumLegs> kFootBodyNames = {
        "opy_v05/knee_1", "opy_v05/knee_2", "opy_v05/knee_3",
        "opy_v05/knee_4"};
    static constexpr std::array<const char*, kNumLegs> kFootGeomNames = {
        "ROBOT_FOOT_1", "ROBOT_FOOT_2", "ROBOT_FOOT_3", "ROBOT_FOOT_4"};
    if (!model_) {
      foot_body_ids_.fill(-1);
      foot_geom_ids_.fill(-1);
      return;
    }
    for (int i = 0; i < kNumLegs; ++i) {
      foot_body_ids_[i] = mj_name2id(model_, mjOBJ_BODY, kFootBodyNames[i]);
      foot_geom_ids_[i] = mj_name2id(model_, mjOBJ_GEOM, kFootGeomNames[i]);
      if (foot_body_ids_[i] < 0 || foot_geom_ids_[i] < 0) {
        std::cerr << "[ModelBasedControllerInterface] Failed to locate foot "
                     "body or geom for leg "
                  << i << std::endl;
      }
    }
  }

  void clearObservationHistory() {
    action_history_.clear();
    body_history_.clear();
    current_action_.assign(kActionDim, 0.0);
  }

  int getMode(){
    return int(_controller->_controlFSM->currentState->stateName);
  }


  void run() {
     _robot_runner->run(); }

  std::array<double, 12> getMotorCommands() {
    std::array<double, 12> motor_commands;
    for (int leg = 0; leg < 4; leg++) {
      motor_commands[leg * 3 + 0 + 0] =
          _actuator_command->tau_abad_ff[leg] +
          _actuator_command->kp_abad[leg] *
              (_actuator_command->q_des_abad[leg] -
               _actuator_data->q_abad[leg]) +
          _actuator_command->kd_abad[leg] *
              (_actuator_command->qd_des_abad[leg] -
               _actuator_data->qd_abad[leg]);  // Torque

      motor_commands[leg * 3 + 1 + 0] =
          _actuator_command->tau_hip_ff[leg] +
          _actuator_command->kp_hip[leg] *
              (_actuator_command->q_des_hip[leg] - _actuator_data->q_hip[leg]) +
          _actuator_command->kd_hip[leg] *
              (_actuator_command->qd_des_hip[leg] -
               _actuator_data->qd_hip[leg]);  // Torque

      motor_commands[leg * 3 + 2 + 0] =
          _actuator_command->tau_knee_ff[leg] +
          _actuator_command->kp_knee[leg] *
              (_actuator_command->q_des_knee[leg] -
               _actuator_data->q_knee[leg]) +
          _actuator_command->kd_knee[leg] *
              (_actuator_command->qd_des_knee[leg] -
               _actuator_data->qd_knee[leg]);  // Torque
    };
    return motor_commands;
  }

  void setFeedback(mjData* data_) {
    last_feedback_data_ = data_;
    for (int leg = 0; leg < 4; leg++) {
      _actuator_data->q_abad[leg] =
          data_->qpos[(leg) * 3 + 0 + 7];  // Add 7 to skip the first 7 dofs
                                           // from body. (Position + Quaternion)
      _actuator_data->q_hip[leg] = data_->qpos[(leg) * 3 + 1 + 7];
      _actuator_data->q_knee[leg] = data_->qpos[(leg) * 3 + 2 + 7];
      _actuator_data->qd_abad[leg] = data_->qvel[(leg) * 3 + 0 + 6];
      _actuator_data->qd_hip[leg] = data_->qvel[(leg) * 3 + 1 + 6];
      _actuator_data->qd_knee[leg] = data_->qvel[(leg) * 3 + 2 + 6];
    }
    _imu_data->acc_x = data_->sensordata[0];
    _imu_data->acc_y = data_->sensordata[1];
    _imu_data->acc_z = data_->sensordata[2];

    _imu_data->accelerometer[0] = data_->sensordata[0];
    _imu_data->accelerometer[1] = data_->sensordata[1];
    _imu_data->accelerometer[2] = data_->sensordata[2];

    _imu_data->heave = data_->qvel[0];
    _imu_data->heave_dt = data_->qvel[1];
    _imu_data->heave_ddt = data_->qvel[2];

    _imu_data->gyr_x = data_->qvel[3];
    _imu_data->gyr_y = data_->qvel[4];
    _imu_data->gyr_z = data_->qvel[5];

    _imu_data->gyro[0] = data_->qvel[3];
    _imu_data->gyro[1] = data_->qvel[4];
    _imu_data->gyro[2] = data_->qvel[5];

    _imu_data->quat[0] = data_->qpos[3];
    _imu_data->quat[1] = data_->qpos[4];
    _imu_data->quat[2] = data_->qpos[5];
    _imu_data->quat[3] = data_->qpos[6];

    _imu_data->pos_x = data_->qpos[0];
    _imu_data->pos_y = data_->qpos[1];
    _imu_data->pos_z = data_->qpos[2];
  }
  // function to map the input in range -1,1 to given limits
  float mapToRange(float input, float in_min=-1, float in_max=1, float out_min=0,
                   float out_max=1) {

    
    return (input - in_min) * (out_max - out_min) / (in_max - in_min) +
           out_min;
  }


  void setAction(const mjtNum* act, std::size_t dim) {
    // Cache the latest action for observation stacking and command mapping.
    if (act == nullptr) {
      current_action_.assign(kActionDim, 0.0);
    } else {
      current_action_.assign(kActionDim, 0.0);
      const std::size_t copy_dim =
          std::min(dim, static_cast<std::size_t>(kActionDim));
      for (std::size_t i = 0; i < copy_dim; ++i) {
        current_action_[i] = act[i];
      }
    }

    gait_scheduler_.update(std::max(frame_skip_, 1), elapsed_step_);
    const auto& contact_schedule = gait_scheduler_.contactState();
    const auto& swing_phase = gait_scheduler_.swingPhase();
    // NOTE: LocomotionCtrl (legged-sim/src/Controllers/WBC_Ctrl/LocomotionCtrl.cpp)
    // treats any value > 0 as a stance constraint. Write back only 1.0 or 0.0
    // so swing legs clear the contact set and the WBC actually lifts them.
    for (int leg = 0; leg < kNumLegs; ++leg) {
      const bool leg_should_contact = contact_schedule[leg] > 0.5f;
      _controller->_controlFSM->data.locomotionCtrlData.contact_state[leg] =
          leg_should_contact ? 1.0f : 0.0f;
    }
    constexpr float kControlDt = 0.002f;
    const auto& seResult =
        _controller->_controlFSM->data._stateEstimator->getResult();
    const Eigen::Matrix3f Rwb = seResult.rBody.transpose();
    const Eigen::Vector3f base_pos = seResult.position;
    const Eigen::Vector3f v_world = seResult.vWorld;
    const Eigen::Vector3f v_body_des_vec(
        _controller->_controlFSM->data.locomotionCtrlData.vBody_des[0],
        _controller->_controlFSM->data.locomotionCtrlData.vBody_des[1], 0.0f);
    const Eigen::Vector3f v_des_world = Rwb * v_body_des_vec;
    const float yaw_rate_des = static_cast<float>(
        _controller->_controlFSM->data.locomotionCtrlData.vBody_Ori_des[2]);
    const float ground_height = base_pos[2] - reset_body_height_;
    const auto* user_params =
        _controller->_controlFSM->data.userParameters;
    const float cmpc_bonus =
        user_params ? static_cast<float>(user_params->cmpc_bonus_swing) : 0.0f;
    constexpr float kGravity = 9.81f;
    _controller->_controlFSM->data.locomotionCtrlData.vBody_des[0] =
        mapToRange(current_action_[0], -1, 1, -0.5f, 0.5f);
    _controller->_controlFSM->data.locomotionCtrlData.vBody_des[1] =
        mapToRange(current_action_[1], -1, 1, -0.2f, 0.2f);
    _controller->_controlFSM->data.locomotionCtrlData.pBody_RPY_des[0] = 0.0f;
    _controller->_controlFSM->data.locomotionCtrlData.pBody_RPY_des[1] = 0.0f;
    _controller->_controlFSM->data.locomotionCtrlData.vBody_Ori_des[2] =
        mapToRange(current_action_[2], -1, 1, -0.5f, 0.5f);

    float accum_foot_des_z = 0.0f;
    for (int leg = 0; leg < kNumLegs; ++leg) {
      const bool was_in_contact = last_contact_schedule_[leg] > 0.5f;
      const bool leg_in_contact = contact_schedule[leg] > 0.5f;
      const auto& leg_data =
          _controller->_controlFSM->data._legController->datas[leg];
      Eigen::Vector3f foot_body(static_cast<float>(leg_data.p[0]),
                                static_cast<float>(leg_data.p[1]),
                                static_cast<float>(leg_data.p[2]));
      const Eigen::Vector3f foot_world = base_pos + Rwb * foot_body;
      if (was_in_contact && !leg_in_contact) {
        auto& swing_traj = gait_scheduler_.swingTrajectory(leg);
        swing_traj.setInitialPosition(foot_world);
        swing_traj.setHeight(0.10f);
      }
      const int base = 3 + leg * kFootActionDim;
      const mjtNum fx = current_action_[base + 0];
      const mjtNum fy = current_action_[base + 1];
      const mjtNum fz = current_action_[base + 2];
      _controller->_controlFSM->data.locomotionCtrlData.Fr_des[leg][0] =
          mapToRange(static_cast<float>(fx), -1, 1, -50.f, 50.0f);
      _controller->_controlFSM->data.locomotionCtrlData.Fr_des[leg][1] =
          mapToRange(static_cast<float>(fy), -1, 1, -50.f, 50.0f);
      _controller->_controlFSM->data.locomotionCtrlData.Fr_des[leg][2] =
          mapToRange(static_cast<float>(fz), -1, 1, 0.f, 100.0f);

      // Enforce: if not in contact, commanded foot force must be zero
      if (!leg_in_contact) {
        _controller->_controlFSM->data.locomotionCtrlData.Fr_des[leg][0] = 0.0f;
        _controller->_controlFSM->data.locomotionCtrlData.Fr_des[leg][1] = 0.0f;
        _controller->_controlFSM->data.locomotionCtrlData.Fr_des[leg][2] = 0.0f;
      } else if (last_contact_schedule_[leg] <= 0.5f &&
                 _controller->_controlFSM->data.locomotionCtrlData.Fr_des[leg][2] <
                     10.0f) {
        // Ensure a minimum normal force when transitioning into stance so the
        // feet settle quickly.
        _controller->_controlFSM->data.locomotionCtrlData.Fr_des[leg][2] = 10.0f;
      }

      if (!leg_in_contact) {
        auto& swing_traj = gait_scheduler_.swingTrajectory(leg);
        const float stance_duration =
            gait_scheduler_.stanceDurationSeconds(leg, kControlDt);
        const float swing_duration =
            std::max(gait_scheduler_.swingDurationSeconds(leg, kControlDt),
                     1e-3f);
        const float clamped_phase = std::clamp(swing_phase[leg], 0.0f, 1.0f);
        const float swing_remaining = swing_duration * (1.0f - clamped_phase);
        Eigen::Vector3f foot_target_world =
            foot_world + v_des_world * swing_remaining;
        // Raibert-inspired foot placement taken from ConvexMPCLocomotion.cpp.
        // Keep the tuning terms in sync with that reference implementation.
        float pfx_rel = v_world[0] * (0.5f + cmpc_bonus) * stance_duration +
                        0.13f * (v_world[0] - v_des_world[0]) +
                        (0.5f * base_pos[2] / kGravity) *
                            (v_world[1] * yaw_rate_des);
        float pfy_rel = v_world[1] * 0.5f * stance_duration +
                        0.13f * (v_world[1] - v_des_world[1]) +
                        (0.5f * base_pos[2] / kGravity) *
                            (-v_world[0] * yaw_rate_des);
        pfx_rel = std::clamp(pfx_rel, -0.35f, 0.35f);
        pfy_rel = std::clamp(pfy_rel, -0.25f, 0.25f);
        foot_target_world[0] += pfx_rel;
        foot_target_world[1] += pfy_rel;
        foot_target_world[2] = ground_height;
        swing_traj.setFinalPosition(foot_target_world);
        swing_traj.setHeight(0.10f);
        swing_traj.compute(clamped_phase, swing_duration);
        const auto& pos = swing_traj.position();
        const auto& vel = swing_traj.velocity();
        const auto& acc = swing_traj.acceleration();
        for (int axis = 0; axis < 3; ++axis) {
          _controller->_controlFSM->data.locomotionCtrlData.pFoot_des[leg][axis] =
              pos[axis];
          _controller->_controlFSM->data.locomotionCtrlData.vFoot_des[leg][axis] =
              vel[axis];
          _controller->_controlFSM->data.locomotionCtrlData.aFoot_des[leg][axis] =
              acc[axis];
        }
        accum_foot_des_z += pos[2];
      } else {
        for (int axis = 0; axis < 3; ++axis) {
          _controller->_controlFSM->data.locomotionCtrlData.pFoot_des[leg][axis] =
              foot_world[axis];
          _controller->_controlFSM->data.locomotionCtrlData.vFoot_des[leg][axis] =
              0.0f;
          _controller->_controlFSM->data.locomotionCtrlData.aFoot_des[leg][axis] =
              0.0f;
        }
        accum_foot_des_z += foot_world[2];
      }
    }
    for (int leg = 0; leg < kNumLegs; ++leg) {
      last_contact_schedule_[leg] = contact_schedule[leg];
    }

    float avg_foot_des_z = accum_foot_des_z / 4.0f;
    float target_height = reset_body_height_ - 0.5f * avg_foot_des_z;
    target_height = std::clamp(target_height, 0.28f, 0.42f);

    int smoothing_horizon = std::max(1, frame_skip_ * 8);
    float phase = 0.0f;
    if (smoothing_horizon > 0) {
      float normalized = static_cast<float>(elapsed_step_ % smoothing_horizon) /
                         static_cast<float>(smoothing_horizon);
      phase = normalized * normalized * (3.0f - 2.0f * normalized);
    }

    float candidate_height = reset_body_height_ +
                             phase * (target_height - reset_body_height_);
    filtered_body_height_ += 0.2f * (candidate_height - filtered_body_height_);
    filtered_body_height_ = std::clamp(filtered_body_height_, 0.28f, 0.42f);
    _controller->_controlFSM->data.locomotionCtrlData.pBody_des[2] =
        filtered_body_height_;
  }

  mjtNum* setObservation(mjtNum* obs, mjtNum desired_body_height) {
    if (obs == nullptr) {
      return obs;
    }
    auto& seResult = _controller->_controlFSM->data._stateEstimator->getResult();
    const bool have_feedback = (last_feedback_data_ != nullptr);
    const bool use_cheater =
        (cheater_mode == 1) && have_feedback;
    const mjData* data = last_feedback_data_;

    mjtNum base_pos[3] = {0, 0, 0};
    mjtNum base_lin_vel[3] = {0, 0, 0};
    mjtNum base_lin_acc[3] = {0, 0, 0};
    mjtNum base_rpy[3] = {0, 0, 0};
    mjtNum base_omega[3] = {0, 0, 0};

    if (use_cheater && data) {
      base_pos[0] = data->qpos[0];
      base_pos[1] = data->qpos[1];
      base_pos[2] = data->qpos[2];

      base_lin_vel[0] = data->qvel[0];
      base_lin_vel[1] = data->qvel[1];
      base_lin_vel[2] = data->qvel[2];

      if (data->qacc) {
        base_lin_acc[0] = data->qacc[0];
        base_lin_acc[1] = data->qacc[1];
        base_lin_acc[2] = data->qacc[2];
      }

      Eigen::Quaternion<mjtNum> quat(data->qpos[6], data->qpos[3], data->qpos[4],
                                     data->qpos[5]);
      const auto euler = quat.toRotationMatrix().eulerAngles(0, 1, 2);
      base_rpy[0] = euler[0];
      base_rpy[1] = euler[1];
      base_rpy[2] = euler[2];

      base_omega[0] = data->qvel[3];
      base_omega[1] = data->qvel[4];
      base_omega[2] = data->qvel[5];
    } else {
      base_pos[0] = seResult.position[0];
      base_pos[1] = seResult.position[1];
      base_pos[2] = seResult.position[2];

      base_lin_vel[0] = seResult.vBody[0];
      base_lin_vel[1] = seResult.vBody[1];
      base_lin_vel[2] = seResult.vBody[2];

      base_lin_acc[0] = seResult.aBody[0];
      base_lin_acc[1] = seResult.aBody[1];
      base_lin_acc[2] = seResult.aBody[2];

      base_rpy[0] = seResult.rpy[0];
      base_rpy[1] = seResult.rpy[1];
      base_rpy[2] = seResult.rpy[2];

      base_omega[0] = seResult.omegaBody[0];
      base_omega[1] = seResult.omegaBody[1];
      base_omega[2] = seResult.omegaBody[2];
    }

    mjtNum* ptr = obs;
    for (int i = 0; i < 3; ++i) {
      *ptr++ = base_pos[i];
    }
    for (int i = 0; i < 3; ++i) {
      *ptr++ = base_lin_vel[i];
    }
    for (int i = 0; i < 3; ++i) {
      *ptr++ = base_lin_acc[i];
    }
    for (int i = 0; i < 3; ++i) {
      *ptr++ = base_rpy[i];
    }
    for (int i = 0; i < 3; ++i) {
      *ptr++ = base_omega[i];
    }

    *ptr++ = static_cast<mjtNum>(filtered_body_height_);
    *ptr++ = desired_body_height;

    for (int leg = 0; leg < kNumLegs; ++leg) {
      const auto& leg_data =
          _controller->_controlFSM->data._legController->datas[leg];
      *ptr++ = (seResult.contactEstimate[leg] > 0) ? 1.0 : 0.0;
      *ptr++ = static_cast<mjtNum>(leg_data.p[0]);
      *ptr++ = static_cast<mjtNum>(leg_data.p[1]);
      *ptr++ = static_cast<mjtNum>(leg_data.p[2]);
      *ptr++ = static_cast<mjtNum>(leg_data.v[0]);
      *ptr++ = static_cast<mjtNum>(leg_data.v[1]);
      *ptr++ = static_cast<mjtNum>(leg_data.v[2]);

      std::array<mjtNum, 3> measured_pos{0.0, 0.0, 0.0};
      std::array<mjtNum, 3> measured_vel{0.0, 0.0, 0.0};
      std::array<mjtNum, 3> measured_force{0.0, 0.0, 0.0};

      if (model_ && data && foot_geom_ids_[leg] >= 0) {
        const int geom_id = foot_geom_ids_[leg];
        measured_pos[0] = data->geom_xpos[3 * geom_id + 0];
        measured_pos[1] = data->geom_xpos[3 * geom_id + 1];
        measured_pos[2] = data->geom_xpos[3 * geom_id + 2];

        mjtNum vel6[6] = {0, 0, 0, 0, 0, 0};
        mj_objectVelocity(model_, data, mjOBJ_GEOM, geom_id, vel6, 0);
        measured_vel[0] = vel6[0];
        measured_vel[1] = vel6[1];
        measured_vel[2] = vel6[2];
      }

      if (data && foot_body_ids_[leg] >= 0) {
        const int body_id = foot_body_ids_[leg];
        measured_force[0] = data->cfrc_ext[6 * body_id + 0];
        measured_force[1] = data->cfrc_ext[6 * body_id + 1];
        measured_force[2] = data->cfrc_ext[6 * body_id + 2];
      }

      for (int i = 0; i < 3; ++i) {
        *ptr++ = measured_pos[i];
      }
      for (int i = 0; i < 3; ++i) {
        *ptr++ = measured_vel[i];
      }
      for (int i = 0; i < 3; ++i) {
        *ptr++ = measured_force[i];
      }
    }

    const int emitted_action_dim =
        std::min(static_cast<int>(current_action_.size()), kActionDim);
    for (int i = 0; i < kActionDim; ++i) {
      const mjtNum value =
          (i < emitted_action_dim) ? current_action_[i] : static_cast<mjtNum>(0);
      *ptr++ = value;
    }

    for (const auto& past_action : action_history_) {
      for (int i = 0; i < kActionDim; ++i) {
        *ptr++ = past_action[i];
      }
    }
    for (std::size_t i = action_history_.size(); i < kHistoryLength; ++i) {
      for (int j = 0; j < kActionDim; ++j) {
        *ptr++ = 0.0;
      }
    }

    for (const auto& past_body : body_history_) {
      for (int i = 0; i < kBodyHistoryDim; ++i) {
        *ptr++ = past_body[i];
      }
    }
    for (std::size_t i = body_history_.size(); i < kHistoryLength; ++i) {
      for (int j = 0; j < kBodyHistoryDim; ++j) {
        *ptr++ = 0.0;
      }
    }

    std::array<mjtNum, kBodyHistoryDim> latest_body_state{};
    latest_body_state[0] = base_lin_vel[0];
    latest_body_state[1] = base_lin_vel[1];
    latest_body_state[2] = base_lin_vel[2];
    latest_body_state[3] = base_omega[0];
    latest_body_state[4] = base_omega[1];
    latest_body_state[5] = base_omega[2];

    if (!current_action_.empty()) {
      action_history_.push_back(current_action_);
      if (action_history_.size() > kHistoryLength) {
        action_history_.pop_front();
      }
    }
    body_history_.push_back(latest_body_state);
    if (body_history_.size() > kHistoryLength) {
      body_history_.pop_front();
    }

    return ptr;
  }

  void clearWBCLogFile(const std::string& filepath = "") {
    if (wbc_log_stream_.is_open()) {
      wbc_log_stream_.close();
    }
    if (!filepath.empty()) {
      std::remove(filepath.c_str());
    } else if (!wbc_log_path_.empty()) {
      std::remove(wbc_log_path_.c_str());
    }
    wbc_log_path_.clear();
  }

  // Write a CSV row with WBC-related data (no LCM). Call this after run().
  // If filepath changes or the stream is closed, it will be re-opened in append mode.
  // Set write_header_if_new=true to write a header when opening a new file.
  void writeWBCLogCSV(const std::string& filepath, bool write_header_if_new = false) {
    if (!_controller || !_controller->_controlFSM) {
      return;
    }
    auto& fsm = *_controller->_controlFSM;
    auto& data = fsm.data;
    if (!data._legController || !data._stateEstimator) {
      return;
    }

    // (Re)open stream if needed
    if (!wbc_log_stream_.is_open() || filepath != wbc_log_path_) {
      if (wbc_log_stream_.is_open()) wbc_log_stream_.close();
      wbc_log_stream_.open(filepath.c_str(), std::ios::out | std::ios::app);
      if (!wbc_log_stream_) return;
      wbc_log_path_ = filepath;
      // Only write header if requested AND file is empty
      auto pos = wbc_log_stream_.tellp();
      bool is_empty = (pos == std::streampos(0));
      if (write_header_if_new && is_empty) writeHeader_();
    }

    const auto& se = data._stateEstimator->getResult();
    auto* legCmd = data._legController->commands;
    auto* legDat = data._legController->datas;
    const auto& loco = data.locomotionCtrlData;

    std::array<double, 12> foot_forces_world{};
    if (auto* locomotion_state = fsm.statesList.locomotion) {
      if (auto* wbc_ctrl = locomotion_state->getWbcCtrl()) {
        const auto& fr = wbc_ctrl->getReactionForces();
        int contact_idx = 0;
        for (int leg = 0; leg < 4; ++leg) {
          if (loco.contact_state[leg] > 0.) {
            for (int axis = 0; axis < 3; ++axis) {
              int idx = 3 * contact_idx + axis;
              if (idx < fr.size()) {
                foot_forces_world[leg * 3 + axis] =
                    static_cast<double>(fr[idx]);
              }
            }
            ++contact_idx;
          }
        }
      }
    }

    // Desired body orientation quaternion from desired RPY (ZYX)
    auto quatFromRPY = [](float roll, float pitch, float yaw) {
      float cr = std::cos(roll * 0.5f);
      float sr = std::sin(roll * 0.5f);
      float cp = std::cos(pitch * 0.5f);
      float sp = std::sin(pitch * 0.5f);
      float cy = std::cos(yaw * 0.5f);
      float sy = std::sin(yaw * 0.5f);
      Eigen::Matrix<float, 4, 1> q;
      q[0] = cr * cp * cy + sr * sp * sy;  // w
      q[1] = sr * cp * cy - cr * sp * sy;  // x
      q[2] = cr * sp * cy + sr * cp * sy;  // y
      q[3] = cr * cp * sy - sr * sp * cy;  // z
      return q;
    };
    Eigen::Matrix<float, 4, 1> q_cmd = quatFromRPY(
        static_cast<float>(loco.pBody_RPY_des[0]),
        static_cast<float>(loco.pBody_RPY_des[1]),
        static_cast<float>(loco.pBody_RPY_des[2]));

    auto append_vec = [&](std::ostringstream& oss, const auto& v, int n) {
      for (int i = 0; i < n; ++i) oss << static_cast<double>(v[i]) << ",";
    };
    auto append_3x4_flat = [&](std::ostringstream& oss, auto getter) {
      for (int leg = 0; leg < 4; ++leg) {
        const auto tmp = getter(leg);
        for (int i = 0; i < 3; ++i) oss << static_cast<double>(tmp[i]) << ",";
      }
    };

    std::ostringstream oss;
    oss.setf(std::ios::fixed); oss.precision(6);

    // contact_est[4]
    for (int leg = 0; leg < 4; ++leg) {
      oss << static_cast<int>(loco.contact_state[leg] > 0 ? 1 : 0) << ",";
    }
    // Fr_des[12]
    append_3x4_flat(oss, [&](int leg){ return loco.Fr_des[leg]; });
    // Fr[12] (estimated contact forces from WBC)
    for (double force : foot_forces_world) oss << force << ",";
    // body_ori_cmd[4]
    for (int i = 0; i < 4; ++i) oss << static_cast<double>(q_cmd[i]) << ",";
    // body_pos_cmd[3], body_vel_cmd[3], body_ang_vel_cmd[3]
    append_vec(oss, loco.pBody_des, 3);
    append_vec(oss, loco.vBody_des, 3);
    append_vec(oss, loco.vBody_Ori_des, 3);
    // body_pos[3], body_vel[3] (use world), body_ori[4], body_ang_vel[3]
    append_vec(oss, se.position, 3);
    append_vec(oss, se.vWorld, 3);
    append_vec(oss, se.orientation, 4);
    append_vec(oss, se.omegaBody, 3);
    // foot_pos_cmd[12], foot_vel_cmd[12], foot_acc_cmd[12]
    append_3x4_flat(oss, [&](int leg){ return loco.pFoot_des[leg]; });
    append_3x4_flat(oss, [&](int leg){ return loco.vFoot_des[leg]; });
    append_3x4_flat(oss, [&](int leg){ return loco.aFoot_des[leg]; });
    // foot_acc_numeric[12] -> zeros
    for (int i = 0; i < 12; ++i) oss << 0.0 << ",";
    // foot_pos[12], foot_vel[12] from leg data (leg frame)
    for (int leg = 0; leg < 4; ++leg) append_vec(oss, legDat[leg].p, 3);
    for (int leg = 0; leg < 4; ++leg) append_vec(oss, legDat[leg].v, 3);
    // foot_local_pos[12], foot_local_vel[12] -> zeros
    for (int i = 0; i < 24; ++i) oss << 0.0 << ",";
    // jpos_cmd[12], jvel_cmd[12]
    for (int leg = 0; leg < 4; ++leg) append_vec(oss, legCmd[leg].qDes, 3);
    for (int leg = 0; leg < 4; ++leg) append_vec(oss, legCmd[leg].qdDes, 3);
    // jacc_cmd[12] -> zeros
    for (int i = 0; i < 12; ++i) oss << 0.0 << ",";
    // jpos[12], jvel[12]
    for (int leg = 0; leg < 4; ++leg) append_vec(oss, legDat[leg].q, 3);
    for (int leg = 0; leg < 4; ++leg) append_vec(oss, legDat[leg].qd, 3);
    // vision_loc[3] -> zeros, end line
    for (int i = 0; i < 3; ++i) oss << 0.0 << (i == 2 ? '\n' : ',');

    wbc_log_stream_ << oss.str();
    wbc_log_stream_.flush();
  }
  // private:
  // std::shared_ptr<EmbeddedController> _controller;
  // std::shared_ptr<SpiCommand> _actuator_command;
  // std::shared_ptr<SpiData> _actuator_data;
  // std::shared_ptr<VectorNavData> _imu_data;
  // std::shared_ptr<GamepadCommand> _gamepad_command;
  // std::shared_ptr<RobotControlParameters> _control_params;
  // std::shared_ptr<PeriodicTaskManager> _periodic_task_manager;
  // std::shared_ptr<RobotRunner> _robot_runner;
  RobotController* _controller;
  SpiCommand* _actuator_command;
  SpiData* _actuator_data;
  VectorNavData* _imu_data;
  GamepadCommand* _gamepad_command;
  // RobotControlParameters* _control_params;
  PeriodicTaskManager* _periodic_task_manager;
  RobotRunner* _robot_runner;
  int data;
  int elapsed_step_ ;
  int frame_skip_ ;
 private:
  mjData* last_feedback_data_{nullptr};
  void writeHeader_() {
    if (!wbc_log_stream_) return;
    // contact_est[4]
    wbc_log_stream_ << "contact_est_FR,contact_est_FL,contact_est_HR,contact_est_HL,";
    // Fr_des[12], Fr[12]
    for (int i = 0; i < 4; ++i) wbc_log_stream_ << "Fr_des_" << i << "_x,Fr_des_" << i << "_y,Fr_des_" << i << "_z,";
    for (int i = 0; i < 4; ++i) wbc_log_stream_ << "Fr_" << i << "_x,Fr_" << i << "_y,Fr_" << i << "_z,";
    // body ori/pos/vel cmd
    wbc_log_stream_ << "body_ori_cmd_w,body_ori_cmd_x,body_ori_cmd_y,body_ori_cmd_z,";
    wbc_log_stream_ << "body_pos_cmd_x,body_pos_cmd_y,body_pos_cmd_z,";
    wbc_log_stream_ << "body_vel_cmd_x,body_vel_cmd_y,body_vel_cmd_z,";
    wbc_log_stream_ << "body_ang_vel_cmd_x,body_ang_vel_cmd_y,body_ang_vel_cmd_z,";
    // body state
    wbc_log_stream_ << "body_pos_x,body_pos_y,body_pos_z,";
    wbc_log_stream_ << "body_vel_x,body_vel_y,body_vel_z,";
    wbc_log_stream_ << "body_ori_w,body_ori_x,body_ori_y,body_ori_z,";
    wbc_log_stream_ << "body_ang_vel_x,body_ang_vel_y,body_ang_vel_z,";
    // foot cmd
    for (int i = 0; i < 4; ++i) wbc_log_stream_ << "foot_pos_cmd_" << i << "_x,foot_pos_cmd_" << i << "_y,foot_pos_cmd_" << i << "_z,";
    for (int i = 0; i < 4; ++i) wbc_log_stream_ << "foot_vel_cmd_" << i << "_x,foot_vel_cmd_" << i << "_y,foot_vel_cmd_" << i << "_z,";
    for (int i = 0; i < 4; ++i) wbc_log_stream_ << "foot_acc_cmd_" << i << "_x,foot_acc_cmd_" << i << "_y,foot_acc_cmd_" << i << "_z,";
    // foot_acc_numeric
    for (int i = 0; i < 4; ++i) wbc_log_stream_ << "foot_acc_num_" << i << "_x,foot_acc_num_" << i << "_y,foot_acc_num_" << i << "_z,";
    // foot state
    for (int i = 0; i < 4; ++i) wbc_log_stream_ << "foot_pos_" << i << "_x,foot_pos_" << i << "_y,foot_pos_" << i << "_z,";
    for (int i = 0; i < 4; ++i) wbc_log_stream_ << "foot_vel_" << i << "_x,foot_vel_" << i << "_y,foot_vel_" << i << "_z,";
    // foot local (unused)
    for (int i = 0; i < 4; ++i) wbc_log_stream_ << "foot_local_pos_" << i << "_x,foot_local_pos_" << i << "_y,foot_local_pos_" << i << "_z,";
    for (int i = 0; i < 4; ++i) wbc_log_stream_ << "foot_local_vel_" << i << "_x,foot_local_vel_" << i << "_y,foot_local_vel_" << i << "_z,";
    // joint cmd
    for (int i = 0; i < 4; ++i) wbc_log_stream_ << "jpos_cmd_" << i << "_h,jpos_cmd_" << i << "_k,jpos_cmd_" << i << "_a,";
    for (int i = 0; i < 4; ++i) wbc_log_stream_ << "jvel_cmd_" << i << "_h,jvel_cmd_" << i << "_k,jvel_cmd_" << i << "_a,";
    for (int i = 0; i < 4; ++i) wbc_log_stream_ << "jacc_cmd_" << i << "_h,jacc_cmd_" << i << "_k,jacc_cmd_" << i << "_a,";
    // joint state
    for (int i = 0; i < 4; ++i) wbc_log_stream_ << "jpos_" << i << "_h,jpos_" << i << "_k,jpos_" << i << "_a,";
    for (int i = 0; i < 4; ++i) wbc_log_stream_ << "jvel_" << i << "_h,jvel_" << i << "_k,jvel_" << i << "_a,";
    // vision
    wbc_log_stream_ << "vision_loc_x,vision_loc_y,vision_loc_z\n";
  }
  std::ofstream wbc_log_stream_;
  std::string wbc_log_path_;
};

#endif  // MBC_INTERFACE_H
