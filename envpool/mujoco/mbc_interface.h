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
    float side_sign[4] = {-1, 1, -1, 1};


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
    
    _controller->_controlFSM->data.locomotionCtrlData.pBody_des[0] = 0; //act[0];
    _controller->_controlFSM->data.locomotionCtrlData.pBody_des[1] = 0; //act[1];
    _controller->_controlFSM->data.locomotionCtrlData.pBody_des[2] += act[0] * 0.005;
    _controller->_controlFSM->data.locomotionCtrlData.pBody_des[2] = std::clamp(_controller->_controlFSM->data.locomotionCtrlData.pBody_des[2], 0.2f, 0.40f);
  
    _controller->_controlFSM->data.locomotionCtrlData.pBody_RPY_des[0] = 0; //act[3];
    _controller->_controlFSM->data.locomotionCtrlData.pBody_RPY_des[1] = 0; //act[4];
    _controller->_controlFSM->data.locomotionCtrlData.pBody_RPY_des[2] = 0; //act[5];

    for (int leg = 0; leg < 4; leg++) {

      _controller->_controlFSM->data.locomotionCtrlData.pFoot_des[leg][0] = 0 ;
          // act[6 + leg * 7];
      _controller->_controlFSM->data.locomotionCtrlData.pFoot_des[leg][1] = 0; 
          // act[7 + leg * 7];
      _controller->_controlFSM->data.locomotionCtrlData.pFoot_des[leg][2] = 0;
          // act[1 + leg * 2];

      _controller->_controlFSM->data.locomotionCtrlData.Fr_des[leg][0] = 0;
          // act[9 + leg * 7];
      _controller->_controlFSM->data.locomotionCtrlData.Fr_des[leg][1] = -5 * side_sign[leg];
          // act[10 + leg * 7];
      _controller->_controlFSM->data.locomotionCtrlData.Fr_des[leg][2] = 85;
          // act[2 + leg * 2];

      _controller->_controlFSM->data.locomotionCtrlData.contact_state[leg] = 1.0f;
          // (act[12 + leg * 7] > 0.5);
    }
  }

    void setObservation(double* obs) {
    // sets the observation array with the current state of the robot
    // in the order of:
    // [position, velocity, acceleration, orientation, angular velocity,
    // contact state, foot position, foot velocity]

    auto& seResult = _controller->_controlFSM->data._stateEstimator->getResult();
    // obs[0] = seResult.position[0];
    // obs[1] = seResult.position[1];
      obs[0] = seResult.position[2]*4 - 1; // Scale to [-1, 1] range

    // obs[3] = seResult.vBody[0];
    // obs[4] = seResult.vBody[1];
    // obs[5] = seResult.vBody[2];

    // obs[6] = seResult.aBody[0];
    // obs[7] = seResult.aBody[1];
    // obs[8] = seResult.aBody[2];

    // obs[9] = seResult.rpy[0];
    // obs[10] = seResult.rpy[1];
    // obs[11] = seResult.rpy[2];

    // obs[12] = seResult.omegaBody[0];
    // obs[13] = seResult.omegaBody[1];
    // obs[14] = seResult.omegaBody[2];


    // for (int leg = 0; leg < 4; leg++) {
    //   obs[15 + leg * 3 + 0] =
    //       (seResult.contactEstimate[leg]>0);

    //   obs[15 + leg * 3 + 1] =
    //       _controller->_controlFSM->data._legController->datas[leg].p[0];
    //   obs[15 + leg * 3 + 2] =
    //       _controller->_controlFSM->data._legController->datas[leg].p[1];
    //   obs[15 + leg * 3 + 3] =
    //       _controller->_controlFSM->data._legController->datas[leg].p[2];

    //   obs[15 + leg * 3 + 4] =
    //       _controller->_controlFSM->data._legController->datas[leg].v[0];
    //   obs[15 + leg * 3 + 5] =
    //       _controller->_controlFSM->data._legController->datas[leg].v[1];
    //   obs[15 + leg * 3 + 6] =
    //       _controller->_controlFSM->data._legController->datas[leg].v[2];


      
    // }

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