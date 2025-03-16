#ifndef MBC_INTERFACE_H
#define MBC_INTERFACE_H

#include <RobotController.h>
#include <RobotRunner.h>
#include <Utilities/RobotCommands.h>

#include <Controllers/EmbeddedController.hpp>
#include <eigen3/Eigen/Dense>
#include <memory>

class ModelBasedControllerInterface {
 public:
  ModelBasedControllerInterface() {
    reset();
    // Constructor implementation
  }

  ~ModelBasedControllerInterface() = default;

  void reset() {
    _controller = new EmbeddedController();
    _actuator_command = new SpiCommand();
    _actuator_data = new SpiData();
    _imu_data = new VectorNavData();
    _gamepad_command = new GamepadCommand();
    _control_params = new RobotControlParameters();
    _periodic_task_manager = new PeriodicTaskManager();

    _robot_runner = new RobotRunner(_controller, _periodic_task_manager, 0.002,
                                    "robot-control");

    _robot_runner->driverCommand = _gamepad_command;
    _robot_runner->_ImuData = _imu_data;
    _robot_runner->_Feedback = _actuator_data;
    _robot_runner->_Command = _actuator_command;
    _robot_runner->controlParameters = _control_params;
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
    _controller->_controlFSM->data.controlParameters->control_mode = 4;
    while (_controller->_controlFSM->currentState->stateName !=
           FSM_StateName::LOCOMOTION) {
      _robot_runner->run();
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

  void setAction(double* act) {
    // double* act = static_cast<double*>(action["action"_].Data());

    _controller->_controlFSM->data.locomotionCtrlData.vBody_des[0] = act[0];
    _controller->_controlFSM->data.locomotionCtrlData.vBody_des[1] = act[1];
    _controller->_controlFSM->data.locomotionCtrlData.vBody_des[2] = act[2];

    _controller->_controlFSM->data.locomotionCtrlData.vBody_Ori_des[0] = act[3];
    _controller->_controlFSM->data.locomotionCtrlData.vBody_Ori_des[1] = act[4];
    _controller->_controlFSM->data.locomotionCtrlData.vBody_Ori_des[2] = act[5];

    for (int leg = 0; leg < 4; leg++) {
      _controller->_controlFSM->data.locomotionCtrlData.vFoot_des[leg][0] =
          act[6 + leg * 4];
      _controller->_controlFSM->data.locomotionCtrlData.vFoot_des[leg][1] =
          act[7 + leg * 4];
      _controller->_controlFSM->data.locomotionCtrlData.vFoot_des[leg][2] =
          act[8 + leg * 4];
      _controller->_controlFSM->data.locomotionCtrlData.contact_state[leg] =
          (act[9 + leg * 4]>0.5);
    }
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
  RobotControlParameters* _control_params;
  PeriodicTaskManager* _periodic_task_manager;
  RobotRunner* _robot_runner;
  int data;
};

#endif  // MBC_INTERFACE_H