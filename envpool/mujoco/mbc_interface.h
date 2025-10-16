#ifndef MBC_INTERFACE_H
#define MBC_INTERFACE_H

#include <cstdio>
#include <memory>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>

#include <RobotController.h>
#include <RobotRunner.h>
#include <Utilities/RobotCommands.h>

#include <Controllers/EmbeddedController.hpp>
#include <Controllers/WBC_Ctrl/WBC_Ctrl.hpp>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>


#define N_FOOT_PARAM_ACT 3
#define N_FOOT_PARAM 7
class ModelBasedControllerInterface {
 public:
  ModelBasedControllerInterface() {
    reset();
    // Constructor implementation
  }
    float side_sign[4] = {-1, 1, -1, 1};
    float reset_body_height_{0.35f};
    float filtered_body_height_{0.35f};
    int cheater_mode{1};

  ~ModelBasedControllerInterface() = default;

  void reset() {
    reset_body_height_ = 0.35f;
    filtered_body_height_ = reset_body_height_;
    last_feedback_data_ = nullptr;
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
  }

  void setBaseHeight(float height) {
    reset_body_height_ = height;
    filtered_body_height_ = height;
    if (_controller && _controller->_controlFSM) {
      _controller->_controlFSM->data.locomotionCtrlData.pBody_des[2] = filtered_body_height_;
    }
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


  void setAction(double* act) {
    // sets the action array with the desired state of the robot
    // in the order of:
    // [body position x, body position y, body position z,
    // body orientation roll, body orientation pitch, body orientation yaw,
    // foot position x, foot position y, foot position z,
    // foot force x, foot force y, foot force z,
    // contact state (0 or 1) for each foot]
    // 
    // The action array is expected to have a size of 6 + 4 * 7 = 34
    // where the first 6 elements are for the body position and orientation,
    // and the next 28 elements are for the foot positions, forces, and contact states.

    // simple trot gait. frame_skip_* 10 leg stance time.
    // gait phase is (elapsed_step_% (frame_skip_ * 10 * 2)) / (frame_skip_ * 10 * 2)

    for (int leg = 0; leg < 4; leg++) {
      // determine contact state based on gait phase
      int gait_cycle_steps = frame_skip_ * 5 * 2;
      float gait_cycle_length = static_cast<float>(gait_cycle_steps);
      int step_in_cycle = elapsed_step_ % gait_cycle_steps;
      float gait_phase = static_cast<float>(step_in_cycle) / gait_cycle_length;

      // contact state based on gait phase only
      bool is_diagonal_leg = (leg == 0 || leg == 3);
      bool should_be_in_contact =
          is_diagonal_leg ? (gait_phase < 0.5) : (gait_phase >= 0.5);
      float contact_state = 1; // should_be_in_contact ? 1.0f : 0.0f;

      _controller->_controlFSM->data.locomotionCtrlData.contact_state[leg] =
          contact_state;
    }
    _controller->_controlFSM->data.locomotionCtrlData.vBody_des[0] =
        mapToRange(act[0], -1, 1, -0.5f, 0.5f);
    _controller->_controlFSM->data.locomotionCtrlData.vBody_des[1] =
        mapToRange(act[1], -1, 1, -0.2f, 0.2f);
    _controller->_controlFSM->data.locomotionCtrlData.pBody_RPY_des[0] = 0.0f;
    _controller->_controlFSM->data.locomotionCtrlData.pBody_RPY_des[1] = 0.0f;
    _controller->_controlFSM->data.locomotionCtrlData.vBody_Ori_des[2] =
        mapToRange(act[2], -1, 1, -0.5f, 0.5f);

    float accum_foot_des_z = 0.0f;
    for (int leg = 0; leg < 4; leg++) {
    //   _controller->_controlFSM->data.locomotionCtrlData.pFoot_des[leg][0] =
    //       mapToRange(act[3 + leg * N_FOOT_PARAM_ACT], -1, 1, -0.2f, 0.2f);
    //   _controller->_controlFSM->data.locomotionCtrlData.pFoot_des[leg][1] =
    //       mapToRange(act[4 + leg * N_FOOT_PARAM_ACT], -1, 1, -0.1f, 0.1f);
    //   float foot_des_z = mapToRange(act[5 + leg * N_FOOT_PARAM_ACT], -1, 1, -0.2f, 0.0f);
    //   accum_foot_des_z += foot_des_z;
    //   _controller->_controlFSM->data.locomotionCtrlData.pFoot_des[leg][2] =
    //       foot_des_z;

      _controller->_controlFSM->data.locomotionCtrlData.Fr_des[leg][0] = mapToRange(act[3 + leg * N_FOOT_PARAM_ACT], -1, 1, -50.f, 50.0f);
      _controller->_controlFSM->data.locomotionCtrlData.Fr_des[leg][1] =   mapToRange(act[4 + leg * N_FOOT_PARAM_ACT], -1, 1, -50.f, 50.0f);
      _controller->_controlFSM->data.locomotionCtrlData.Fr_des[leg][2] = mapToRange(act[5 + leg * N_FOOT_PARAM_ACT], -1, 1,0.f, 100.0f);

      // Enforce: if not in contact, commanded foot force must be zero
      // if (_controller->_controlFSM->data.locomotionCtrlData.contact_state[leg] !=
      //     1.0f) {
      //   _controller->_controlFSM->data.locomotionCtrlData.Fr_des[leg][0] = 0.0f;
      //   _controller->_controlFSM->data.locomotionCtrlData.Fr_des[leg][1] = 0.0f;
      //   _controller->_controlFSM->data.locomotionCtrlData.Fr_des[leg][2] = 0.0f;
      // }
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

    void setObservation(mjtNum* obs) {
    // sets the observation array with the current state of the robot
    // in the order of:
    // [position, velocity, acceleration, orientation, angular velocity,
    // contact state, foot position, foot velocity]

    auto& seResult = _controller->_controlFSM->data._stateEstimator->getResult();
    const bool use_cheater = (cheater_mode == 1) && (last_feedback_data_ != nullptr);
    if (use_cheater) {
      const mjData* data = last_feedback_data_;
      obs[0] = data->qpos[0];
      obs[1] = data->qpos[1];
      obs[2] = data->qpos[2];

      obs[3] = data->qvel[0];
      obs[4] = data->qvel[1];
      obs[5] = data->qvel[2];

      if (data->qacc) {
        obs[6] = data->qacc[0];
        obs[7] = data->qacc[1];
        obs[8] = data->qacc[2];
      } else {
        obs[6] = 0.0;
        obs[7] = 0.0;
        obs[8] = 0.0;
      }

      Eigen::Quaternion<mjtNum> quat(data->qpos[6], data->qpos[3], data->qpos[4],
                                     data->qpos[5]);
      const auto euler = quat.toRotationMatrix().eulerAngles(0, 1, 2);
      obs[9] = euler[0];
      obs[10] = euler[1];
      obs[11] = euler[2];

      obs[12] = data->qvel[3];
      obs[13] = data->qvel[4];
      obs[14] = data->qvel[5];
    } else {
      obs[0] = seResult.position[0];
      obs[1] = seResult.position[1];
      obs[2] = seResult.position[2];

      obs[3] = seResult.vBody[0];
      obs[4] = seResult.vBody[1];
      obs[5] = seResult.vBody[2];

      obs[6] = seResult.aBody[0];
      obs[7] = seResult.aBody[1];
      obs[8] = seResult.aBody[2];

      obs[9] = seResult.rpy[0];
      obs[10] = seResult.rpy[1];
      obs[11] = seResult.rpy[2];

      obs[12] = seResult.omegaBody[0];
      obs[13] = seResult.omegaBody[1];
      obs[14] = seResult.omegaBody[2];
    }


    for (int leg = 0; leg < 4; leg++) {
      obs[15 + leg * N_FOOT_PARAM + 0] =
          (seResult.contactEstimate[leg]>0);

      obs[15 + leg * N_FOOT_PARAM + 1] =
          _controller->_controlFSM->data._legController->datas[leg].p[0];
      obs[15 + leg * N_FOOT_PARAM + 2] =
          _controller->_controlFSM->data._legController->datas[leg].p[1];
      obs[15 + leg * N_FOOT_PARAM + 3] =
          _controller->_controlFSM->data._legController->datas[leg].p[2];

      obs[15 + leg * N_FOOT_PARAM + 4] =
          _controller->_controlFSM->data._legController->datas[leg].v[0];
      obs[15 + leg * N_FOOT_PARAM + 5] =
          _controller->_controlFSM->data._legController->datas[leg].v[1];
      obs[15 + leg * N_FOOT_PARAM + 6] =
          _controller->_controlFSM->data._legController->datas[leg].v[2];


      
    }
   obs[43] = filtered_body_height_;
  
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
