#ifndef MBC_INTERFACE_H
#define MBC_INTERFACE_H

#include <Controllers/LegController.h>
#include <FSM/ControlFSMData.h>
#include <RobotController.h>
#include <RobotRunner.h>
#include <Utilities/RobotCommands.h>
#include <mujoco/mujoco.h>

#include <Controllers/EmbeddedController.hpp>
#include <Controllers/WBC_Ctrl/WBC_Ctrl.hpp>
#include <algorithm>
#include "cppTypes.h"
#include <array>
#include <cmath>
#include <cstdio>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "footstep_planner.h"
#include "legged_gait_scheduler.h"

class ModelBasedControllerInterface {
 public:
  static constexpr int kNumLegs = 4;
  static constexpr int kFootActionDim = 5;
  static constexpr int kActionDim = 24;
  static constexpr int kPhaseDeltaIdx = kActionDim - 1;
  // Nominal phase advance per control tick (~40 ticks per cycle).
  static constexpr float kDefaultPhaseDelta = 1.0f / 200.0f;
  // Keep learned phase deltas near nominal to avoid racing through the gait.
  static constexpr float kMaxPhaseDelta = 2*kDefaultPhaseDelta;
  static constexpr float kControlDt = 0.002f;
  static constexpr int kCommandDim = 3;
  static constexpr int kObservationDim =
      1   /* commanded forward velocity */ +
      6   /* base linear/angular velocity */ +
      3   /* base rpy */ +
      28  /* joint positions/velocities/contact */ +
      6   /* feet positions relative to base (FR, FL) */ +
      2;  /* phase (sin, cos) */

  ModelBasedControllerInterface()
      : gait_scheduler_(kControlDt), footstep_planner_(gait_scheduler_) {
    reset();
    // Constructor implementation
  }

  ~ModelBasedControllerInterface() = default;

  void setDesiredBodyHeight(float height) {
    filtered_body_height_ = height;
    if (auto* loco = mutableLocomotionData_()) {
      loco->pBody_des[2] = height;
    }
  }

  float desiredBodyHeight() const {
    if (const auto* loco = locomotionData()) {
      return loco->pBody_des[2];
    }
    return filtered_body_height_;
  }

  int controlMode() const {
    if (const auto* data = controlData()) {
      return data->controlParameters ? data->controlParameters->control_mode
                                     : -1;
    }
    return -1;
  }

  const LocomotionCtrlData<float>* locomotionData() const {
    if (const auto* data = controlData()) {
      return &data->locomotionCtrlData;
    }
    return nullptr;
  }

  const LegController<float>* legController() const {
    if (const auto* data = controlData()) {
      return data->_legController;
    }
    return nullptr;
  }

  FloatingBaseModel<float>* robotModel() const {
    if (_controller) {
      return _controller->model();
    }
    return nullptr;
  }

  const SpiData* actuatorData() const { return _actuator_data; }

  const VectorNavData* imuData() const { return _imu_data; }

  void reset() {
    reset_body_height_ = 0.36f;
    filtered_body_height_ = reset_body_height_;
    state_.feedback = nullptr;
    clearObservationHistory();
    gait_scheduler_.reset();
    state_.contact_schedule = gait_scheduler_.contactState();

    if (!_controller) {
      _controller = new EmbeddedController();
    }
    if (!_actuator_command) {
      _actuator_command = new SpiCommand();
    }
    if (!_actuator_data) {
      _actuator_data = new SpiData();
    }
    if (!_imu_data) {
      _imu_data = new VectorNavData();
    }
    if (!_gamepad_command) {
      _gamepad_command = new GamepadCommand();
    }
    if (!_periodic_task_manager) {
      _periodic_task_manager = new PeriodicTaskManager();
    }
    if (!_robot_runner) {
      _robot_runner = new RobotRunner(_controller, _periodic_task_manager,
                                      kControlDt, "robot-control");
    }

    _robot_runner->driverCommand = _gamepad_command;
    _robot_runner->_ImuData = _imu_data;
    _robot_runner->_Feedback = _actuator_data;
    _robot_runner->_Command = _actuator_command;
    _robot_runner->initializeParameters();
    _robot_runner->init();

    if (auto* loco = mutableLocomotionData_()) {
      for (int leg = 0; leg < kNumLegs; ++leg) {
        loco->Fr_se[leg].setZero();
        loco->Fr_des[leg].setZero();
      }
    }
  }

  void setModeStandUp() {
    if (!_controller || !_controller->_controlFSM || !_robot_runner) {
      return;
    }
    _controller->_controlFSM->data.controlParameters->control_mode = 1;
    while (_controller->_controlFSM->currentState->stateName !=
           FSM_StateName::BALANCE_STAND) {
      _robot_runner->run();
    }
  }

  void setModeLocomotion() {
    if (!_controller || !_controller->_controlFSM || !_robot_runner) {
      return;
    }
    _robot_runner->initializeStateEstimator();
    _controller->_controlFSM->data.controlParameters->control_mode = 4;
    while (_controller->_controlFSM->currentState->stateName !=
           FSM_StateName::LOCOMOTION) {
      _robot_runner->run();
    }
    filtered_body_height_ = reset_body_height_;
    if (auto* loco = mutableLocomotionData_()) {
      loco->pBody_des[2] = filtered_body_height_;
    }
    gait_scheduler_.reset();
    state_.contact_schedule = gait_scheduler_.contactState();
  }

  void setBaseHeight(float height) {
    reset_body_height_ = height;
    filtered_body_height_ = height;
    if (auto* loco = mutableLocomotionData_()) {
      loco->pBody_des[2] = filtered_body_height_;
    }
  }

  void setCommandVelocity(mjtNum vx, mjtNum vy, mjtNum yaw_rate) {
    latest_cmd_vel_[0] = vx;
    latest_cmd_vel_[1] = vy;
    latest_cmd_vel_[2] = yaw_rate;
  }

  std::array<mjtNum, kCommandDim> commandVelocity() const {
    return latest_cmd_vel_;
  }

  void setCommandResidualLimits(float linear_limit, float yaw_limit) {
    cmd_linear_residual_limit_ = std::max(0.0f, linear_limit);
    cmd_yaw_residual_limit_ = std::max(0.0f, yaw_limit);
  }

  void setGaitPattern(const std::array<float, kNumLegs>& offsets,
                      const std::array<float, kNumLegs>& contact_durations) {
    gait_scheduler_.setPattern(offsets, contact_durations);
  }

  void setGaitPhase(float theta) { gait_scheduler_.setPhase(theta); }

  void setModel(const mjModel* model) {
    model_ = model;
    static constexpr std::array<const char*, kNumLegs> kFootBodyNames = {
        "demir/knee_1", "demir/knee_2", "demir/knee_3", "demir/knee_4"};
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
    current_action_.assign(kActionDim, 0.0);
  }

  int getMode() const {
    if (!_controller || !_controller->_controlFSM) {
      return -1;
    }
    return static_cast<int>(_controller->_controlFSM->currentState->stateName);
  }

  void run() {
    if (_robot_runner) {
      _robot_runner->run();
    }
  }

  std::array<double, 12> getMotorCommands() {
    std::array<double, 12> motor_commands{};
    if (!_actuator_command || !_actuator_data) {
      return motor_commands;
    }
    for (int leg = 0; leg < kNumLegs; ++leg) {
      motor_commands[leg * 3 + 0] = 
          _actuator_command->tau_abad_ff[leg] +
          _actuator_command->kp_abad[leg] *
              (_actuator_command->q_des_abad[leg] -
               _actuator_data->q_abad[leg]) +
          _actuator_command->kd_abad[leg] *
              (_actuator_command->qd_des_abad[leg] -
               _actuator_data->qd_abad[leg]);  // Torque

      motor_commands[leg * 3 + 1] =
          _actuator_command->tau_hip_ff[leg] +
          _actuator_command->kp_hip[leg] *
              (_actuator_command->q_des_hip[leg] - _actuator_data->q_hip[leg]) +
          _actuator_command->kd_hip[leg] *
              (_actuator_command->qd_des_hip[leg] -
               _actuator_data->qd_hip[leg]);  // Torque

      motor_commands[leg * 3 + 2] =
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
    if (!_actuator_data || !_imu_data) {
      return;
    }
    state_.feedback = data_;
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

    if (cheater_mode && _controller && _controller->_controlFSM &&
        data_ && data_->qpos && data_->qvel) {
      auto* se = _controller->_controlFSM->data._stateEstimator;
      if (se) {
        auto* result = se->getResultHandle();
        Eigen::Quaternion<mjtNum> quat(data_->qpos[3], data_->qpos[4],
                                       data_->qpos[5], data_->qpos[6]);
        quat.normalize();

        const Eigen::Matrix<mjtNum, 3, 3> R_body_to_world =
            quat.toRotationMatrix();
        const Eigen::Matrix<mjtNum, 3, 3> R_world_to_body =
            R_body_to_world.transpose();

        const Eigen::Matrix<mjtNum, 3, 1> v_world(data_->qvel[0],
                                                  data_->qvel[1],
                                                  data_->qvel[2]);
        const Eigen::Matrix<mjtNum, 3, 1> v_body = R_world_to_body * v_world;

        const Eigen::Matrix<mjtNum, 3, 1> omega_world(data_->qvel[3],
                                                      data_->qvel[4],
                                                      data_->qvel[5]);
        const Eigen::Matrix<mjtNum, 3, 1> omega_body =
            R_world_to_body * omega_world;

        Eigen::Matrix<mjtNum, 3, 1> a_world =
            Eigen::Matrix<mjtNum, 3, 1>::Zero();
        if (data_->qacc) {
          a_world[0] = data_->qacc[0];
          a_world[1] = data_->qacc[1];
          a_world[2] = data_->qacc[2];
        }
        const Eigen::Matrix<mjtNum, 3, 1> a_body =
            R_world_to_body * a_world;

        result->position[0] = static_cast<float>(data_->qpos[0]);
        result->position[1] = static_cast<float>(data_->qpos[1]);
        result->position[2] = static_cast<float>(data_->qpos[2]);
        result->rBody = R_world_to_body.cast<float>();
        result->vWorld = v_world.cast<float>();
        result->vBody = v_body.cast<float>();

        const auto euler = R_body_to_world.eulerAngles(0, 1, 2);
        result->rpy[0] = static_cast<float>(euler[0]);
        result->rpy[1] = static_cast<float>(euler[1]);
        result->rpy[2] = static_cast<float>(euler[2]);

        result->omegaWorld = omega_world.cast<float>();
        result->omegaBody = omega_body.cast<float>();
        result->aWorld = a_world.cast<float>();
        result->aBody = a_body.cast<float>();

        result->orientation[0] = static_cast<float>(quat.w());
        result->orientation[1] = static_cast<float>(quat.x());
        result->orientation[2] = static_cast<float>(quat.y());
        result->orientation[3] = static_cast<float>(quat.z());
      }
    }
  }
  // function to map the input in range -1,1 to given limits
  float mapToRange(float input, float in_min = -1, float in_max = 1,
                   float out_min = 0, float out_max = 1) {
    return (input - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
  }

  

  void setAction(const mjtNum* act, std::size_t dim) {
    // Cache the latest action for observation and command mapping.
    std::size_t incoming_dim = 0;
    current_action_.assign(kActionDim, 0.0);
    if (act != nullptr) {
      incoming_dim = std::min(dim, static_cast<std::size_t>(kActionDim));
      for (std::size_t i = 0; i < incoming_dim; ++i) {
        current_action_[i] = act[i];
      }
    }

    if (!_controller || !_controller->_controlFSM) {
      return;
    }
    auto& data = _controller->_controlFSM->data;
    if (!data._legController || !data._stateEstimator) {
      return;
    }

    auto& loco = data.locomotionCtrlData;
    const float delta_theta = mapToRange(current_action_[kPhaseDeltaIdx], -1.0f, 1.0f, 0.0f, kMaxPhaseDelta);
    state_.delta_theta = delta_theta;
    gait_scheduler_.update(delta_theta);
    state_.gait_phase = gait_scheduler_.phase();
    state_.gait_cycle_time = gait_scheduler_.cycleTimeSeconds();

    const auto prev_contact_schedule = state_.contact_schedule;
    const auto& contact_schedule = gait_scheduler_.contactState();
    const auto& swing_phase = gait_scheduler_.swingPhase();
    const auto contact_phase = gait_scheduler_.contactPhase();
    // NOTE: LocomotionCtrl
    // (legged-sim/src/Controllers/WBC_Ctrl/LocomotionCtrl.cpp) treats any value
    // > 0 as a stance constraint. Write back only 1.0 or 0.  
    // clear the contact set and the WBC actually lifts them.
    for (int leg = 0; leg < kNumLegs; ++leg) {
      const bool leg_should_contact = contact_schedule[leg] > 0.0f;
      loco.contact_state[leg] = leg_should_contact ? 1.0f : 0.0f;
    }
    std::array<float, kNumLegs> updated_contact_state{};
    for (int leg = 0; leg < kNumLegs; ++leg) {
      updated_contact_state[leg] = loco.contact_state[leg];
    }
    state_.gait_contact_state = updated_contact_state;
    state_.gait_contact_phase = contact_phase;
    state_.gait_swing_phase = swing_phase;
    state_.contact_schedule = updated_contact_state;
    const auto& seResult = data._stateEstimator->getResult();
    const mjData* sim_data = state_.feedback;

    const Eigen::Matrix3f Rwb = seResult.rBody.transpose();
    Eigen::Vector3f base_pos = seResult.position;
    const Eigen::Vector3f v_world = seResult.vWorld;
    // set desired body position in x and y such that max error is 5 cm

    const float residual_vx =
        mapToRange(current_action_[0], -1, 1, -cmd_linear_residual_limit_,
                   cmd_linear_residual_limit_);
    const float residual_vy =
        mapToRange(current_action_[1], -1, 1, -cmd_linear_residual_limit_,
                   cmd_linear_residual_limit_);
    const float target_vx =
      static_cast<float>(latest_cmd_vel_[0]) + residual_vx;
    const float target_vy =
      static_cast<float>(latest_cmd_vel_[1]) + residual_vy;
    const float target_vz = 0.0f;

    constexpr float kMaxStepChange = 0.2f;
    auto apply_rate_limit = [](float current, float target, float max_delta) {
      float delta = target - current;
      delta = std::clamp(delta, -max_delta, max_delta);
      return current + delta;
    };

    loco.vBody_des[0] =
      apply_rate_limit(loco.vBody_des[0], target_vx, kMaxStepChange);
    loco.vBody_des[1] =
      apply_rate_limit(loco.vBody_des[1], target_vy, kMaxStepChange);
    loco.vBody_des[2] =
      apply_rate_limit(loco.vBody_des[2], target_vz, kMaxStepChange);

    loco.pBody_des[0] = base_pos[0] + loco.vBody_des[0] * kControlDt ;
    loco.pBody_des[1] = base_pos[1] + loco.vBody_des[1] * kControlDt;

    loco.vBody_Ori_des[0] = 0.0f;
    loco.vBody_Ori_des[1] = 0.0f;
    const float residual_yaw =
        mapToRange(current_action_[2], -1, 1, -cmd_yaw_residual_limit_,
                   cmd_yaw_residual_limit_);
    loco.vBody_Ori_des[2] =
        static_cast<float>(latest_cmd_vel_[2]) + residual_yaw;

    loco.pBody_RPY_des[0] = 0.0f;
    loco.pBody_RPY_des[1] = 0.0f;
    loco.pBody_RPY_des[2] = seResult.rpy[2] + loco.vBody_Ori_des[2] * kControlDt;


    for (int leg = 0; leg < kNumLegs; ++leg) {
      if (contact_schedule[leg] > 0.0f) { // if leg is in stance, set desired GRF
        const int base = 3 + leg * kFootActionDim;
        const mjtNum fx = current_action_[base + 0];
        const mjtNum fy = current_action_[base + 1];
        const mjtNum fz = current_action_[base + 2];
        loco.Fr_des[leg][0] =
            mapToRange(static_cast<float>(fx), -1, 1, -50.f, 50.0f);
        loco.Fr_des[leg][1] =
            mapToRange(static_cast<float>(fy), -1, 1, -50.f, 50.0f);
        loco.Fr_des[leg][2] =
            mapToRange(static_cast<float>(fz), -1, 1, 0.f, 250.0f);
      } else {
        loco.Fr_des[leg].setZero();
      }


      constexpr float kContactForceThreshold = 5.0f;
      loco.Fr_se[leg].setZero();
      if (sim_data && foot_body_ids_[leg] >= 0) {
        const int body_id = foot_body_ids_[leg];
        Eigen::Vector3f contact_force(
            static_cast<float>(sim_data->cfrc_ext[6 * body_id + 0]),
            static_cast<float>(sim_data->cfrc_ext[6 * body_id + 1]),
            static_cast<float>(sim_data->cfrc_ext[6 * body_id + 2]));
        if (contact_force.norm() > kContactForceThreshold) {
          for (int axis = 0; axis < 3 /*foot force dim*/; ++axis) {
            loco.Fr_se[leg][axis] = contact_force[axis];
          }
        }
      }
    }

    const auto* user_params = data.userParameters;
    const float cmpc_bonus =
        user_params ? static_cast<float>(user_params->cmpc_bonus_swing) : 0.0f;
    constexpr float kGravity = 9.81f;
    const Eigen::Vector3f v_body_des_vec(loco.vBody_des[0], loco.vBody_des[1],
                                         0.0f);
    const Eigen::Vector3f v_des_world = Rwb * v_body_des_vec;
    const float yaw_rate_des = static_cast<float>(loco.vBody_Ori_des[2]);
    FootstepPlanner::PlanInput plan_input{updated_contact_state,
                                          prev_contact_schedule,
                                          swing_phase,
                                          base_pos,
                                          v_world,
                                          Rwb,
                                          v_des_world,
                                          yaw_rate_des,
                                          cmpc_bonus,
                                          kGravity,
                                          *data._legController,
                                          current_action_};
    float accum_foot_des_z = footstep_planner_.Plan(plan_input, loco);
    float avg_foot_des_z = accum_foot_des_z / static_cast<float>(kNumLegs);
    float target_height = reset_body_height_ - 0.5f * avg_foot_des_z;
    target_height = std::clamp(target_height, 0.28f, 0.42f);

    filtered_body_height_ += 0.2f * (target_height - filtered_body_height_);
    filtered_body_height_ = std::clamp(filtered_body_height_, 0.28f, 0.42f);
    // Track the filtered target height instead of a hard-coded value to avoid
    // unintended lift when the command is to hold still.
    loco.pBody_des[2] = filtered_body_height_;

    // Forward contact schedule into the state estimator so the estimator is
    // aware of which feet are in contact. This mirrors calls like
    // data._stateEstimator->setContactPhase(se_contactState) used elsewhere
    // (e.g. ConvexMPCLocomotion).
    if (data._stateEstimator) {
      ::Vec4<float> phase;
      for (int i = 0; i < kNumLegs; ++i) phase[i] = updated_contact_state[i];
      data._stateEstimator->setContactPhase(phase);
    }

  }

  // Allow external code to set the contact phase used by the state estimator.
  // This forwards a 4-element contact phase array into the internal
  // StateEstimatorContainer by building an ::Vec4<float> and calling
  // setContactPhase on the estimator if present.
  void setContactPhase(const std::array<float, kNumLegs>& contact_phase) {
    if (!_controller || !_controller->_controlFSM) return;
    auto* data = &_controller->_controlFSM->data;
    if (!data->_stateEstimator) return;
    ::Vec4<float> phase;
    for (int i = 0; i < kNumLegs; ++i) phase[i] = static_cast<float>(contact_phase[i]);
    data->_stateEstimator->setContactPhase(phase);
  }

  mjtNum* setObservation(mjtNum* obs) {
    if (obs == nullptr) {
      return obs;
    }
    auto& seResult =
        _controller->_controlFSM->data._stateEstimator->getResult();
    const bool have_feedback = (state_.feedback != nullptr);
    const bool use_cheater = (cheater_mode == 1) && have_feedback;
    const mjData* data = state_.feedback;

    mjtNum base_pos[3] = {0, 0, 0};
    mjtNum base_lin_vel[3] = {0, 0, 0};
    mjtNum base_rpy[3] = {0, 0, 0};
    mjtNum base_omega[3] = {0, 0, 0};
    Eigen::Matrix<mjtNum, 3, 3> R_world_to_body =
        Eigen::Matrix<mjtNum, 3, 3>::Identity();

    if (use_cheater && data) {
      base_pos[0] = data->qpos[0];
      base_pos[1] = data->qpos[1];
      base_pos[2] = data->qpos[2];

      Eigen::Quaternion<mjtNum> quat(data->qpos[3], data->qpos[4],
                                     data->qpos[5], data->qpos[6]);
      const auto euler = quat.toRotationMatrix().eulerAngles(0, 1, 2);
      base_rpy[0] = euler[0];
      base_rpy[1] = euler[1];
      base_rpy[2] = euler[2];

      base_omega[0] = data->qvel[3];
      base_omega[1] = data->qvel[4];
      base_omega[2] = data->qvel[5];

      R_world_to_body = quat.toRotationMatrix().transpose();
      const Eigen::Matrix<mjtNum, 3, 1> v_world(data->qvel[0], data->qvel[1],
                                                data->qvel[2]);
      const Eigen::Matrix<mjtNum, 3, 1> v_body = R_world_to_body * v_world;
      base_lin_vel[0] = v_body[0];
      base_lin_vel[1] = v_body[1];
      base_lin_vel[2] = v_body[2];
    } else {
      base_pos[0] = seResult.position[0];
      base_pos[1] = seResult.position[1];
      base_pos[2] = seResult.position[2];

      base_lin_vel[0] = seResult.vBody[0];
      base_lin_vel[1] = seResult.vBody[1];
      base_lin_vel[2] = seResult.vBody[2];

      base_rpy[0] = seResult.rpy[0];
      base_rpy[1] = seResult.rpy[1];
      base_rpy[2] = seResult.rpy[2];

      base_omega[0] = seResult.omegaBody[0];
      base_omega[1] = seResult.omegaBody[1];
      base_omega[2] = seResult.omegaBody[2];

      for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
          R_world_to_body(r, c) =
              static_cast<mjtNum>(seResult.rBody(r, c));
        }
      }
    }

    mjtNum* ptr = obs;
    // 1. Commanded forward velocity.
    *ptr++ = latest_cmd_vel_[0];

    // 2. Base linear (body-frame) and angular velocities.
    for (int i = 0; i < 3; ++i) {
      *ptr++ = base_lin_vel[i];
    }
    for (int i = 0; i < 3; ++i) {
      *ptr++ = base_omega[i];
    }

    // 3. Base orientation (roll, pitch, yaw).
    for (int i = 0; i < 3; ++i) {
      *ptr++ = base_rpy[i];
    }

    // 4. Joint positions and velocities (12 each) + contact estimates (4) = 28.
    for (int leg = 0; leg < kNumLegs; ++leg) {
      *ptr++ = _actuator_data ? _actuator_data->q_abad[leg] : 0.0;
      *ptr++ = _actuator_data ? _actuator_data->q_hip[leg] : 0.0;
      *ptr++ = _actuator_data ? _actuator_data->q_knee[leg] : 0.0;
    }
    for (int leg = 0; leg < kNumLegs; ++leg) {
      *ptr++ = _actuator_data ? _actuator_data->qd_abad[leg] : 0.0;
      *ptr++ = _actuator_data ? _actuator_data->qd_hip[leg] : 0.0;
      *ptr++ = _actuator_data ? _actuator_data->qd_knee[leg] : 0.0;
    }
    for (int leg = 0; leg < kNumLegs; ++leg) {
      const bool contact_est = (seResult.contactEstimate[leg] > 0);
      *ptr++ = contact_est ? static_cast<mjtNum>(1.0)
                           : static_cast<mjtNum>(0.0);
    }

    // 5. Feet Cartesian positions relative to base frame (front-right, front-left).
    std::array<std::array<mjtNum, 3>, kNumLegs> foot_pos_body{};
    for (int leg = 0; leg < kNumLegs; ++leg) {
      Eigen::Matrix<mjtNum, 3, 1> foot_world =
          Eigen::Matrix<mjtNum, 3, 1>::Zero();
      if (model_ && data && foot_geom_ids_[leg] >= 0) {
        const int geom_id = foot_geom_ids_[leg];
        foot_world[0] = data->geom_xpos[3 * geom_id + 0];
        foot_world[1] = data->geom_xpos[3 * geom_id + 1];
        foot_world[2] = data->geom_xpos[3 * geom_id + 2];
        const Eigen::Matrix<mjtNum, 3, 1> base_world(base_pos[0], base_pos[1],
                                                     base_pos[2]);
        const Eigen::Matrix<mjtNum, 3, 1> foot_body =
            R_world_to_body * (foot_world - base_world);
        foot_pos_body[leg][0] = foot_body[0];
        foot_pos_body[leg][1] = foot_body[1];
        foot_pos_body[leg][2] = foot_body[2];
      } else {
        foot_pos_body[leg].fill(static_cast<mjtNum>(0.0));
      }
    }
    // Encode only front-right (0) and front-left (1) for the 6-dim footprint.
    for (int axis = 0; axis < 3; ++axis) {
      *ptr++ = foot_pos_body[0][axis];
    }
    for (int axis = 0; axis < 3; ++axis) {
      *ptr++ = foot_pos_body[1][axis];
    }

    // 6. Phase encoding (sin, cos of averaged swing phase).
    const auto& swing_phase = gait_scheduler_.swingPhase();
    mjtNum phi = static_cast<mjtNum>(0.0);
    for (int leg = 0; leg < kNumLegs; ++leg) {
      phi += static_cast<mjtNum>(swing_phase[leg]);
    }
    phi /= static_cast<mjtNum>(kNumLegs);
    phi = std::clamp(phi, static_cast<mjtNum>(0.0), static_cast<mjtNum>(1.0));
    constexpr mjtNum kTwoPi =
        static_cast<mjtNum>(6.28318530717958647692);  // 2 * pi
    *ptr++ = std::sin(kTwoPi * phi);
    *ptr++ = std::cos(kTwoPi * phi);

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
  // If filepath changes or the stream is closed, it will be re-opened in append
  // mode. Set write_header_if_new=true to write a header when opening a new
  // file.
  void writeWBCLogCSV(const std::string& filepath,
                      bool write_header_if_new = false) {
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

    std::array<double, kNumLegs * 3 /* foot force dim */> foot_forces_world{};
    if (auto* locomotion_state = fsm.statesList.locomotion) {
      if (auto* wbc_ctrl = locomotion_state->getWbcCtrl()) {
        const auto& fr = wbc_ctrl->getReactionForces();
        std::size_t contact_idx = 0;
        for (int leg = 0; leg < kNumLegs; ++leg) {
          if (loco.contact_state[leg] > 0.5f) {
            for (int axis = 0; axis < 3 /* foot force dim */; ++axis) {
              const std::size_t fr_idx = contact_idx * 3 /* foot force dim */ + axis;
              if (fr_idx < static_cast<std::size_t>(fr.size())) {
                foot_forces_world[leg * 3 /* foot force dim */ + axis] =
                    static_cast<double>(fr[fr_idx]);
              }
            }
            ++contact_idx;
          } else {
            for (int axis = 0; axis < 3 /* foot force dim */; ++axis) {
              foot_forces_world[leg * 3 /* foot force dim */ + axis] = 0.0;
            }
          }
        }
        // std::cout << "Reaction Forces (per leg FR, FL, HR, HL): ";
        for (double force : foot_forces_world) {
          // std::cout << force << " ";
        }
        // std::cout << std::endl;
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
    Eigen::Matrix<float, 4, 1> q_cmd =
        quatFromRPY(static_cast<float>(loco.pBody_RPY_des[0]),
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

    auto to_vec3d = [](const auto& vec) {
      return Eigen::Vector3d(static_cast<double>(vec[0]),
                             static_cast<double>(vec[1]),
                             static_cast<double>(vec[2]));
    };

    const mjData* sim_data = state_.feedback;

    Eigen::Matrix3d R_world_to_body_se;
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        R_world_to_body_se(r, c) = static_cast<double>(se.rBody(r, c));
      }
    }
    Eigen::Matrix3d R_body_to_world_se = R_world_to_body_se.transpose();

    const Eigen::Vector3d body_pos_se = to_vec3d(se.position);
    const Eigen::Vector3d v_world_se = to_vec3d(se.vWorld);
    const Eigen::Vector3d v_body_se = to_vec3d(se.vBody);
    const Eigen::Vector3d omega_body_se = to_vec3d(se.omegaBody);
    const Eigen::Vector3d omega_world_se = to_vec3d(se.omegaWorld);
    Eigen::Vector3d a_body_se = to_vec3d(se.aBody);
    Eigen::Vector3d a_world_se = to_vec3d(se.aWorld);
    const Eigen::Vector3d rpy_se = to_vec3d(se.rpy);

    Eigen::Vector4d quat_se;
    for (int i = 0; i < 4; ++i) {
      quat_se[i] = static_cast<double>(se.orientation[i]);
    }

    Eigen::Matrix3d R_body_to_world_sim = R_body_to_world_se;
    Eigen::Vector3d body_pos_sim = body_pos_se;
    Eigen::Vector3d v_world_sim = v_world_se;
    Eigen::Vector3d v_body_sim = v_body_se;
    Eigen::Vector3d omega_body_sim = omega_body_se;
    Eigen::Vector3d omega_world_sim = omega_world_se;
    Eigen::Vector3d a_body_sim = a_body_se;
    Eigen::Vector3d a_world_sim = a_world_se;
    Eigen::Vector3d rpy_sim = rpy_se;
    Eigen::Vector4d quat_sim = quat_se;
    const bool have_sim = (sim_data != nullptr);

    if (have_sim) {
      body_pos_sim << static_cast<double>(sim_data->qpos[0]),
          static_cast<double>(sim_data->qpos[1]),
          static_cast<double>(sim_data->qpos[2]);
      v_world_sim << static_cast<double>(sim_data->qvel[0]),
          static_cast<double>(sim_data->qvel[1]),
          static_cast<double>(sim_data->qvel[2]);
      omega_body_sim << static_cast<double>(sim_data->qvel[3]),
          static_cast<double>(sim_data->qvel[4]),
          static_cast<double>(sim_data->qvel[5]);

      Eigen::Quaternion<mjtNum> quat_tmp(sim_data->qpos[3], sim_data->qpos[4],
                                  sim_data->qpos[5], sim_data->qpos[6]);
      quat_tmp.normalize();
      R_body_to_world_sim = quat_tmp.toRotationMatrix();
      quat_sim << quat_tmp.w(), quat_tmp.x(), quat_tmp.y(), quat_tmp.z();
      rpy_sim = R_body_to_world_sim.eulerAngles(0, 1, 2);

      v_body_sim = R_body_to_world_sim.transpose() * v_world_sim;
      omega_world_sim = R_body_to_world_sim * omega_body_sim;

      if (sim_data->qacc) {
        a_world_sim << static_cast<double>(sim_data->qacc[0]),
            static_cast<double>(sim_data->qacc[1]),
            static_cast<double>(sim_data->qacc[2]);
        a_body_sim = R_body_to_world_sim.transpose() * a_world_sim;
      } else {
        a_world_sim.setZero();
        a_body_sim = R_body_to_world_sim.transpose() * a_world_sim;
      }
    }

    std::array<double, 4> contact_est_se{};
    std::array<double, 4> contact_est_sim{};
    for (int leg = 0; leg < 4; ++leg) {
      contact_est_se[leg] = std::max(
          0.0, std::min(1.0, static_cast<double>(se.contactEstimate[leg])));
    }
    if (have_sim) {
      for (int leg = 0; leg < 4; ++leg) {
        contact_est_sim[leg] = 0.0;
        const int body_id = foot_body_ids_[leg];
        if (body_id >= 0) {
          const double fx =
              static_cast<double>(sim_data->cfrc_ext[6 * body_id + 0]);
          const double fy =
              static_cast<double>(sim_data->cfrc_ext[6 * body_id + 1]);
          const double fz =
              static_cast<double>(sim_data->cfrc_ext[6 * body_id + 2]);
          const double force_norm = std::sqrt(fx * fx + fy * fy + fz * fz);
          if (force_norm > 5.0) {
            contact_est_sim[leg] = 1.0;
          }
        }
      }
    } else {
      for (int leg = 0; leg < 4; ++leg) {
        contact_est_sim[leg] = loco.contact_state[leg] > 0.0f ? 1.0 : 0.0;
      }
    }

    std::array<Eigen::Vector3d, 4> foot_pos_local;
    std::array<Eigen::Vector3d, 4> foot_vel_local;
    std::array<Eigen::Vector3d, 4> foot_pos_world_se;
    std::array<Eigen::Vector3d, 4> foot_vel_world_se;
    std::array<Eigen::Vector3d, 4> foot_pos_world_sim;
    std::array<Eigen::Vector3d, 4> foot_vel_world_sim;

    for (int leg = 0; leg < 4; ++leg) {
      foot_pos_local[leg] = to_vec3d(legDat[leg].p);
      foot_vel_local[leg] = to_vec3d(legDat[leg].v);
      Eigen::Vector3d hip = Eigen::Vector3d::Zero();
      if (legDat[leg].quadruped) {
        hip = to_vec3d(legDat[leg].quadruped->getHipLocation(leg));
      }
      const Eigen::Vector3d foot_body = hip + foot_pos_local[leg];

      const Eigen::Vector3d vel_body_se =
          foot_vel_local[leg] + omega_body_se.cross(foot_body);
      foot_pos_world_se[leg] = body_pos_se + R_body_to_world_se * foot_body;
      foot_vel_world_se[leg] = v_world_se + R_body_to_world_se * vel_body_se;

      const Eigen::Vector3d vel_body_sim =
          foot_vel_local[leg] + omega_body_sim.cross(foot_body);
      foot_pos_world_sim[leg] = body_pos_sim + R_body_to_world_sim * foot_body;
      foot_vel_world_sim[leg] =
          v_world_sim + R_body_to_world_sim * vel_body_sim;
    }

    std::array<double, 12> motor_cmd{};
    if (_actuator_command && _actuator_data) {
      motor_cmd = getMotorCommands();
    }

    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss.precision(6);

    // contact_estimator vs sim [4 each]
    for (int leg = 0; leg < 4; ++leg) {
      oss << contact_est_se[leg] << ",";
    }
    for (int leg = 0; leg < 4; ++leg) {
      oss << contact_est_sim[leg] << ",";
    }
    // Fr_des[12]
    append_3x4_flat(oss, [&](int leg) { return loco.Fr_des[leg]; });
    // Fr_se[12] (Mujoco contact forces when in contact)
    append_3x4_flat(oss, [&](int leg) { return loco.Fr_se[leg]; });
    // Fr[12] (estimated contact forces from WBC)
    for (double force : foot_forces_world) oss << force << ",";
    // body_ori_cmd[4]
    for (int i = 0; i < 4; ++i) oss << static_cast<double>(q_cmd[i]) << ",";
    // body_pos_cmd[3], body_vel_cmd[3], body_ang_vel_cmd[3]
    append_vec(oss, loco.pBody_des, 3);
    append_vec(oss, loco.vBody_des, 3);
    append_vec(oss, loco.vBody_Ori_des, 3);
    // body states: estimator vs sim
    append_vec(oss, body_pos_se, 3);
    append_vec(oss, body_pos_sim, 3);
    append_vec(oss, v_world_se, 3);
    append_vec(oss, v_world_sim, 3);
    append_vec(oss, v_body_se, 3);
    append_vec(oss, v_body_sim, 3);
    append_vec(oss, quat_se, 4);
    append_vec(oss, quat_sim, 4);
    append_vec(oss, rpy_se, 3);
    append_vec(oss, rpy_sim, 3);
    append_vec(oss, omega_body_se, 3);
    append_vec(oss, omega_body_sim, 3);
    append_vec(oss, omega_world_se, 3);
    append_vec(oss, omega_world_sim, 3);
    append_vec(oss, a_body_se, 3);
    append_vec(oss, a_body_sim, 3);
    append_vec(oss, a_world_se, 3);
    append_vec(oss, a_world_sim, 3);
    // foot_pos_cmd[12], foot_vel_cmd[12], foot_acc_cmd[12]
    append_3x4_flat(oss, [&](int leg) { return loco.pFoot_des[leg]; });
    append_3x4_flat(oss, [&](int leg) { return loco.vFoot_des[leg]; });
    append_3x4_flat(oss, [&](int leg) { return loco.aFoot_des[leg]; });
    // foot_acc_numeric[12] -> zeros
    for (int i = 0; i < 12; ++i) oss << 0.0 << ",";
    // foot_pos[12] (world) estimator vs sim, foot_vel[12] (world)
    append_3x4_flat(oss, [&](int leg) { return foot_pos_world_se[leg]; });
    append_3x4_flat(oss, [&](int leg) { return foot_pos_world_sim[leg]; });
    append_3x4_flat(oss, [&](int leg) { return foot_vel_world_se[leg]; });
    append_3x4_flat(oss, [&](int leg) { return foot_vel_world_sim[leg]; });
    // foot_local_pos[12], foot_local_vel[12] (leg frame)
    append_3x4_flat(oss, [&](int leg) { return foot_pos_local[leg]; });
    append_3x4_flat(oss, [&](int leg) { return foot_vel_local[leg]; });
    // jpos_cmd[12], jvel_cmd[12]
    for (int leg = 0; leg < 4; ++leg) append_vec(oss, legCmd[leg].qDes, 3);
    for (int leg = 0; leg < 4; ++leg) append_vec(oss, legCmd[leg].qdDes, 3);
    // jacc_cmd[12] -> zeros
    for (int i = 0; i < 12; ++i) oss << 0.0 << ",";
    // tau_ff_cmd[12]
    for (int leg = 0; leg < 4; ++leg) {
      for (int axis = 0; axis < 3; ++axis) {
        oss << static_cast<double>(legCmd[leg].tauFeedForward[axis]) << ",";
      }
    }
    // motor_cmd[12]
    for (double cmd : motor_cmd) oss << cmd << ",";
    // jpos[12], jvel[12]
    for (int leg = 0; leg < 4; ++leg) append_vec(oss, legDat[leg].q, 3);
    for (int leg = 0; leg < 4; ++leg) append_vec(oss, legDat[leg].qd, 3);
    // gait state
    oss << static_cast<double>(state_.delta_theta) << ","
        << static_cast<double>(state_.gait_phase) << ","
        << static_cast<double>(state_.gait_cycle_time) << ",";
    for (int leg = 0; leg < kNumLegs; ++leg) {
      oss << static_cast<double>(state_.gait_contact_state[leg]) << ",";
    }
    for (int leg = 0; leg < kNumLegs; ++leg) {
      oss << static_cast<double>(state_.gait_contact_phase[leg]) << ",";
    }
    for (int leg = 0; leg < kNumLegs; ++leg) {
      oss << static_cast<double>(state_.gait_swing_phase[leg]) << ",";
    }
    // vision_loc[3] -> zeros, end line
    for (int i = 0; i < 3; ++i) oss << 0.0 << (i == 2 ? '\n' : ',');

    wbc_log_stream_ << oss.str();
    wbc_log_stream_.flush();
  }
 private:
  ControlFSMData<float>* mutableControlData_() {
    if (!_controller || !_controller->_controlFSM) {
      return nullptr;
    }
    return &_controller->_controlFSM->data;
  }

  const ControlFSMData<float>* controlData() const {
    if (!_controller || !_controller->_controlFSM) {
      return nullptr;
    }
    return &_controller->_controlFSM->data;
  }

  LocomotionCtrlData<float>* mutableLocomotionData_() {
    if (auto* data = mutableControlData_()) {
      return &data->locomotionCtrlData;
    }
    return nullptr;
  }

  float reset_body_height_{0.36f};
  float filtered_body_height_{0.36f};
  int cheater_mode{1};  // 0: off, 1: on
  const mjModel* model_{nullptr};
  std::array<int, kNumLegs> foot_body_ids_{-1, -1, -1, -1};
  std::array<int, kNumLegs> foot_geom_ids_{-1, -1, -1, -1};
  std::vector<mjtNum> current_action_;
  std::array<mjtNum, kCommandDim> latest_cmd_vel_{{0.0, 0.0, 0.0}};
  float cmd_linear_residual_limit_{2.0f};
  float cmd_yaw_residual_limit_{1.0f};
  LeggedGaitScheduler gait_scheduler_;
  FootstepPlanner footstep_planner_;
  struct RuntimeState {
    std::array<float, kNumLegs> contact_schedule{{1.0f, 1.0f, 1.0f, 1.0f}};
    std::array<float, kNumLegs> gait_contact_state{{1.0f, 1.0f, 1.0f, 1.0f}};
    std::array<float, kNumLegs> gait_contact_phase{{0.0f, 0.0f, 0.0f, 0.0f}};
    std::array<float, kNumLegs> gait_swing_phase{{0.0f, 0.0f, 0.0f, 0.0f}};
    float delta_theta{0.0f};
    float gait_phase{0.0f};
    float gait_cycle_time{0.0f};
    mjData* feedback{nullptr};
  } state_;
  RobotController* _controller{nullptr};
  SpiCommand* _actuator_command{nullptr};
  SpiData* _actuator_data{nullptr};
  VectorNavData* _imu_data{nullptr};
  GamepadCommand* _gamepad_command{nullptr};
  // RobotControlParameters* _control_params{nullptr};
  PeriodicTaskManager* _periodic_task_manager{nullptr};
  RobotRunner* _robot_runner{nullptr};
  // TODO can we do this better? Automatic log file header and management
  void writeHeader_() {
    if (!wbc_log_stream_) return;
    // contact estimates
    wbc_log_stream_ << "contact_est_se_FR,contact_est_se_FL,contact_est_se_HR,"
                       "contact_est_se_HL,";
    wbc_log_stream_ << "contact_est_sim_FR,contact_est_sim_FL,contact_est_sim_"
                       "HR,contact_est_sim_HL,";
    // Fr_des[12], Fr_se[12], Fr[12]
    for (int i = 0; i < 4; ++i)
      wbc_log_stream_ << "Fr_des_" << i << "_x,Fr_des_" << i << "_y,Fr_des_"
                      << i << "_z,";
    for (int i = 0; i < 4; ++i)
      wbc_log_stream_ << "Fr_se_" << i << "_x,Fr_se_" << i << "_y,Fr_se_"
                      << i << "_z,";
    for (int i = 0; i < 4; ++i)
      wbc_log_stream_ << "Fr_" << i << "_x,Fr_" << i << "_y,Fr_" << i << "_z,";
    // body ori/pos/vel cmd
    wbc_log_stream_
        << "body_ori_cmd_w,body_ori_cmd_x,body_ori_cmd_y,body_ori_cmd_z,";
    wbc_log_stream_ << "body_pos_cmd_x,body_pos_cmd_y,body_pos_cmd_z,";
    wbc_log_stream_ << "body_vel_cmd_x,body_vel_cmd_y,body_vel_cmd_z,";
    wbc_log_stream_
        << "body_ang_vel_cmd_x,body_ang_vel_cmd_y,body_ang_vel_cmd_z,";
    // body state (estimator vs sim)
    wbc_log_stream_ << "body_pos_se_x,body_pos_se_y,body_pos_se_z,";
    wbc_log_stream_ << "body_pos_sim_x,body_pos_sim_y,body_pos_sim_z,";
    wbc_log_stream_
        << "body_vel_se_world_x,body_vel_se_world_y,body_vel_se_world_z,";
    wbc_log_stream_
        << "body_vel_sim_world_x,body_vel_sim_world_y,body_vel_sim_world_z,";
    wbc_log_stream_
        << "body_vel_se_body_x,body_vel_se_body_y,body_vel_se_body_z,";
    wbc_log_stream_
        << "body_vel_sim_body_x,body_vel_sim_body_y,body_vel_sim_body_z,";
    wbc_log_stream_
        << "body_ori_se_w,body_ori_se_x,body_ori_se_y,body_ori_se_z,";
    wbc_log_stream_
        << "body_ori_sim_w,body_ori_sim_x,body_ori_sim_y,body_ori_sim_z,";
    wbc_log_stream_ << "body_rpy_se_r,body_rpy_se_p,body_rpy_se_y,";
    wbc_log_stream_ << "body_rpy_sim_r,body_rpy_sim_p,body_rpy_sim_y,";
    wbc_log_stream_ << "body_ang_vel_se_body_x,body_ang_vel_se_body_y,body_ang_"
                       "vel_se_body_z,";
    wbc_log_stream_ << "body_ang_vel_sim_body_x,body_ang_vel_sim_body_y,body_"
                       "ang_vel_sim_body_z,";
    wbc_log_stream_ << "body_ang_vel_se_world_x,body_ang_vel_se_world_y,body_"
                       "ang_vel_se_world_z,";
    wbc_log_stream_ << "body_ang_vel_sim_world_x,body_ang_vel_sim_world_y,body_"
                       "ang_vel_sim_world_z,";
    wbc_log_stream_
        << "body_acc_se_body_x,body_acc_se_body_y,body_acc_se_body_z,";
    wbc_log_stream_
        << "body_acc_sim_body_x,body_acc_sim_body_y,body_acc_sim_body_z,";
    wbc_log_stream_
        << "body_acc_se_world_x,body_acc_se_world_y,body_acc_se_world_z,";
    wbc_log_stream_
        << "body_acc_sim_world_x,body_acc_sim_world_y,body_acc_sim_world_z,";
    // foot cmd
    for (int i = 0; i < 4; ++i)
      wbc_log_stream_ << "foot_pos_cmd_" << i << "_x,foot_pos_cmd_" << i
                      << "_y,foot_pos_cmd_" << i << "_z,";
    for (int i = 0; i < 4; ++i)
      wbc_log_stream_ << "foot_vel_cmd_" << i << "_x,foot_vel_cmd_" << i
                      << "_y,foot_vel_cmd_" << i << "_z,";
    for (int i = 0; i < 4; ++i)
      wbc_log_stream_ << "foot_acc_cmd_" << i << "_x,foot_acc_cmd_" << i
                      << "_y,foot_acc_cmd_" << i << "_z,";
    // foot_acc_numeric
    for (int i = 0; i < 4; ++i)
      wbc_log_stream_ << "foot_acc_num_" << i << "_x,foot_acc_num_" << i
                      << "_y,foot_acc_num_" << i << "_z,";
    // foot state (estimator vs sim)
    for (int i = 0; i < 4; ++i)
      wbc_log_stream_ << "foot_pos_se_" << i << "_x,foot_pos_se_" << i
                      << "_y,foot_pos_se_" << i << "_z,";
    for (int i = 0; i < 4; ++i)
      wbc_log_stream_ << "foot_pos_sim_" << i << "_x,foot_pos_sim_" << i
                      << "_y,foot_pos_sim_" << i << "_z,";
    for (int i = 0; i < 4; ++i)
      wbc_log_stream_ << "foot_vel_se_" << i << "_x,foot_vel_se_" << i
                      << "_y,foot_vel_se_" << i << "_z,";
    for (int i = 0; i < 4; ++i)
      wbc_log_stream_ << "foot_vel_sim_" << i << "_x,foot_vel_sim_" << i
                      << "_y,foot_vel_sim_" << i << "_z,";
    // foot local (leg frame)
    for (int i = 0; i < 4; ++i)
      wbc_log_stream_ << "foot_local_pos_" << i << "_x,foot_local_pos_" << i
                      << "_y,foot_local_pos_" << i << "_z,";
    for (int i = 0; i < 4; ++i)
      wbc_log_stream_ << "foot_local_vel_" << i << "_x,foot_local_vel_" << i
                      << "_y,foot_local_vel_" << i << "_z,";
    // joint cmd
    for (int i = 0; i < 4; ++i)
      wbc_log_stream_ << "jpos_cmd_" << i << "_h,jpos_cmd_" << i
                      << "_k,jpos_cmd_" << i << "_a,";
    for (int i = 0; i < 4; ++i)
      wbc_log_stream_ << "jvel_cmd_" << i << "_h,jvel_cmd_" << i
                      << "_k,jvel_cmd_" << i << "_a,";
    for (int i = 0; i < 4; ++i)
      wbc_log_stream_ << "jacc_cmd_" << i << "_h,jacc_cmd_" << i
                      << "_k,jacc_cmd_" << i << "_a,";
    for (int i = 0; i < 4; ++i)
      wbc_log_stream_ << "tau_abad_ff_" << i << ",tau_hip_ff_" << i
                      << ",tau_knee_ff_" << i << ",";
    for (int i = 0; i < 4; ++i)
      wbc_log_stream_ << "motor_cmd_" << i << "_abad,motor_cmd_" << i
                      << "_hip,motor_cmd_" << i << "_knee,";
    // joint state
    for (int i = 0; i < 4; ++i)
      wbc_log_stream_ << "jpos_" << i << "_h,jpos_" << i << "_k,jpos_" << i
                      << "_a,";
    for (int i = 0; i < 4; ++i)
      wbc_log_stream_ << "jvel_" << i << "_h,jvel_" << i << "_k,jvel_" << i
                      << "_a,";
    // gait state
    wbc_log_stream_ << "gait_delta_theta,gait_phase,gait_cycle_s,";
    wbc_log_stream_ << "gait_contact_FR,gait_contact_FL,gait_contact_HR,gait_contact_HL,";
    wbc_log_stream_ << "gait_contact_phase_FR,gait_contact_phase_FL,gait_contact_phase_HR,gait_contact_phase_HL,";
    wbc_log_stream_ << "gait_swing_phase_FR,gait_swing_phase_FL,gait_swing_phase_HR,gait_swing_phase_HL,";
    // vision
    wbc_log_stream_ << "vision_loc_x,vision_loc_y,vision_loc_z\n";
  }
  std::ofstream wbc_log_stream_;
  std::string wbc_log_path_;
};

#endif  // MBC_INTERFACE_H
