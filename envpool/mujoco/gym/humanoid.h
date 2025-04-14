#ifndef ENVPOOL_MUJOCO_GYM_HUMANOID_H_
#define ENVPOOL_MUJOCO_GYM_HUMANOID_H_

#include <RobotRunner.h>

#include <algorithm>
#include <cmath>  // For std::sqrt, std::abs
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <string>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/mujoco/gym/mujoco_env.h"
#include "mbc_interface.h"
// #include <Utilities/Utilities_print.h>
// #include <Math/orientation_tools.h>
// #include <eigen3/Eigen/Dense>
#include <RobotController.h>
// #include <legged-sim/include/Controllers/EmbeddedController.hpp>
#include <Controllers/EmbeddedController.hpp>
// #include "Utilities/PeriodicTask.h"
#include <Utilities/RobotCommands.h>

#include <eigen3/Eigen/Dense>

namespace mujoco_gym {

class HumanoidEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict(
        "frame_skip"_.Bind(5), "post_constraint"_.Bind(true),
        "use_contact_force"_.Bind(false), "forward_reward_weight"_.Bind(1.25),
        "terminate_when_unhealthy"_.Bind(true),
        "exclude_current_positions_from_observation"_.Bind(true),
        "ctrl_cost_weight"_.Bind(0.0), "healthy_reward"_.Bind(5.0),
        "healthy_z_min"_.Bind(0.05), "healthy_z_max"_.Bind(0.45),
        "contact_cost_weight"_.Bind(5e-7), "contact_cost_max"_.Bind(10.0),
        "reset_noise_scale"_.Bind(0));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    mjtNum inf = std::numeric_limits<mjtNum>::infinity();
    bool no_pos = conf["exclude_current_positions_from_observation"_];
    return MakeDict("obs"_.Bind(Spec<mjtNum>({361}, {-inf, inf})),
#ifdef ENVPOOL_TEST
                    "info:qpos0"_.Bind(Spec<mjtNum>({24})),
                    "info:qvel0"_.Bind(Spec<mjtNum>({23})),
#endif
                    "info:reward_linvel"_.Bind(Spec<mjtNum>({-1})),
                    "info:reward_quadctrl"_.Bind(Spec<mjtNum>({-1})),
                    "info:reward_alive"_.Bind(Spec<mjtNum>({-1})),
                    "info:reward_impact"_.Bind(Spec<mjtNum>({-1})),
                    "info:x_position"_.Bind(Spec<mjtNum>({-1})),
                    "info:y_position"_.Bind(Spec<mjtNum>({-1})),
                    "info:distance_from_origin"_.Bind(Spec<mjtNum>({-1})),
                    "info:x_velocity"_.Bind(Spec<mjtNum>({-1})),
                    "info:y_velocity"_.Bind(Spec<mjtNum>({-1})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    // Fix action shape: change from 47 to 22 dimensions.
    return MakeDict("action"_.Bind(Spec<mjtNum>({-1, 22}, {-1, 1})));
  }
};

using HumanoidEnvSpec = EnvSpec<HumanoidEnvFns>;

class HumanoidEnv : public Env<HumanoidEnvSpec>, public MujocoEnv {
 protected:
  bool terminate_when_unhealthy_, no_pos_, use_contact_force_;
  mjtNum ctrl_cost_weight_, forward_reward_weight_, healthy_reward_;
  mjtNum healthy_z_min_, healthy_z_max_;
  mjtNum contact_cost_weight_, contact_cost_max_;
  std::uniform_real_distribution<> dist_;
  ModelBasedControllerInterface mbc;
  // New: Backup of the controller
  ModelBasedControllerInterface mbc_backup;
  mjModel* model_backup_;
  mjData* data_backup_;
  std::ofstream outputFile;
  // Added: Torque bound constant (example value; adjust as needed)
  const mjtNum torque_limit_ = 65.0;
  // Added: CSV logging switch (set to 1 manually to enable CSV writing)
  int csv_logging_enabled_ = 1;

 public:
  HumanoidEnv(const Spec& spec, int env_id)
      : Env<HumanoidEnvSpec>(spec, env_id),
        mbc(),
        // Construct MujocoEnv using a fixed XML file path.
        MujocoEnv(std::string("/app/envpool/envpool/mujoco/legged-sim/resource/"
                              "opy_v05/opy_v05.xml"),
                  spec.config["frame_skip"_], spec.config["post_constraint"_],
                  spec.config["max_episode_steps"_]),
        terminate_when_unhealthy_(spec.config["terminate_when_unhealthy"_]),
        no_pos_(spec.config["exclude_current_positions_from_observation"_]),
        use_contact_force_(spec.config["use_contact_force"_]),
        ctrl_cost_weight_(spec.config["ctrl_cost_weight"_]),
        forward_reward_weight_(spec.config["forward_reward_weight"_]),
        healthy_reward_(spec.config["healthy_reward"_]),
        healthy_z_min_(spec.config["healthy_z_min"_]),
        healthy_z_max_(spec.config["healthy_z_max"_]),
        contact_cost_weight_(spec.config["contact_cost_weight"_]),
        contact_cost_max_(spec.config["contact_cost_max"_]),
        dist_(-spec.config["reset_noise_scale"_],
              spec.config["reset_noise_scale"_]) {
    std::string fname;
    fname = "/app/envpool/logs/" + std::to_string(env_id_) + "_log.csv";
    
    outputFile.open(fname.c_str());
    
    MujocoReset();
    // Save the initial controller configuration to backup
    while (mbc.getMode() != 6) {
      
      mjtNum dummy_action[22];
      // set all values to zero;
      for (int i = 0; i < 22; i++) {
        dummy_action[i] = 0;
      }
      mjtNum* act = dummy_action;
      mbc.setAction(act);
      mbc.run();

      mjtNum motor_commands[12];
      std::array<double, 12> mc = mbc.getMotorCommands();
      for (int i = 0; i < 12; ++i) {
        // Clamp motor commands to torque bounds.
        motor_commands[i] =
            std::max(std::min(static_cast<mjtNum>(mc[i]), torque_limit_),
                     -torque_limit_);
      }
      const auto& before = GetMassCenter();

      MujocoStep(motor_commands);

      writeDataToCSV(0);
    }
    writeDataToCSV(2);
    mbc_backup = mbc;
    model_backup_ = mj_copyModel(nullptr, model_);
    data_backup_ = mj_copyData(nullptr,model_, data_ );

  }

  void MujocoResetModel() override {
    // for (int i = 0; i < model_->nq; ++i) {
    //   data_->qpos[i] = init_qpos_[i] + dist_(gen_);
    // }
    // for (int i = 0; i < model_->nv; ++i) {
    //   data_->qvel[i] = init_qvel_[i] + dist_(gen_);
    // }
    int kSideSign_[4] = {-1, 1, -1, 1};

    model_->opt.timestep = 0.002;
    
    
    // data_->qpos[2] = 0.6;  // Set the height to 0.125
   
    for (int leg = 0; leg < 4; leg++) {
      data_->qpos[(leg) * 3 + 0 + 7] =
          0 * (M_PI / 180) *
          kSideSign_[leg];  // Add 7 to skip the first 7 dofs from body.
                            // (Position + Quaternion)
      data_->qpos[(leg) * 3 + 1 + 7] = -90 * (M_PI / 180);  //*kDirSign_[leg];
      data_->qpos[(leg) * 3 + 2 + 7] = 173 * (M_PI / 180);  //*kDirSign_[leg];
    }

#ifdef ENVPOOL_TEST
    std::memcpy(qpos0_, data_->qpos, sizeof(mjtNum) * model_->nq);
    std::memcpy(qvel0_, data_->qvel, sizeof(mjtNum) * model_->nv);
#endif
  }

  bool IsDone() override { return done_; }

  void Reset() override {
    std::cout << "reset" << std::endl;

    writeDataToCSV(1);
    MujocoReset();
    // Instead of resetting the controller, we copy the backup.
    mbc = mbc_backup;
    model_ = mj_copyModel(nullptr, model_backup_);
    data_ = mj_copyData(nullptr,model_, data_backup_ );

    WriteState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    done_ = false;
    elapsed_step_ = 0;
  }

  void Step(const Action& action) override {
    // step
    mjtNum* act = static_cast<mjtNum*>(action["action"_].Data());
    mbc.setAction(act);
    mbc.run();

    mjtNum motor_commands[12];
    std::array<double, 12> mc = mbc.getMotorCommands();
    for (int i = 0; i < 12; ++i) {
      // Clamp motor commands to torque bounds.
      motor_commands[i] = std::max(
          std::min(static_cast<mjtNum>(mc[i]), torque_limit_), -torque_limit_);
    }
    const auto& before = GetMassCenter();

    MujocoStep(motor_commands);
    const auto& after = GetMassCenter();

    // Compute control cost.
    mjtNum ctrl_cost = 0.0;
    for (int i = 0; i < model_->nu; ++i) {
      ctrl_cost += ctrl_cost_weight_ * act[i] * act[i];
    }
    // Compute velocities.
    mjtNum dt = frame_skip_ * model_->opt.timestep;
    mjtNum xv = (after[0] - before[0]) / dt;
    mjtNum yv = (after[1] - before[1]) / dt;
    // Compute contact cost.
    mjtNum contact_cost = 0.0;
    if (use_contact_force_) {
      for (int i = 0; i < 6 * model_->nbody; ++i) {
        mjtNum x = data_->cfrc_ext[i];
        contact_cost += contact_cost_weight_ * x * x;
      }
      contact_cost = std::min(contact_cost, contact_cost_max_);
    }

    // reward and done
    mjtNum healthy_reward =
        terminate_when_unhealthy_ || IsHealthy() ? healthy_reward_ : 0.0;
    // auto reward = static_cast<float>(xv * forward_reward_weight_ +
    //                                  healthy_reward - ctrl_cost -
    //                                  contact_cost);
    // Use the standing reward function.
    auto reward = ComputeStandingReward();

    ++elapsed_step_;
    done_ = (terminate_when_unhealthy_ ? !IsHealthy() : false) ||
            (elapsed_step_ >= max_episode_steps_);
    WriteState(reward, xv, yv, ctrl_cost, contact_cost, after[0], after[1],
               healthy_reward_);
  }

 private:
  bool IsHealthy() {
    if (mbc._controller->_controlFSM->data.controlParameters->control_mode ==
        0) {
      return false;  // end if state is passive
    }

    bool healthy =
        healthy_z_min_ < data_->qpos[2] && data_->qpos[2] < healthy_z_max_;
    return healthy;
  }

  std::array<mjtNum, 2> GetMassCenter() {
    mjtNum mass_sum = 0.0;
    mjtNum mass_x = 0.0;
    mjtNum mass_y = 0.0;
    for (int i = 0; i < model_->nbody; ++i) {
      mjtNum mass = model_->body_mass[i];
      mass_sum += mass;
      mass_x += mass * data_->xipos[i * 3 + 0];
      mass_y += mass * data_->xipos[i * 3 + 1];
    }
    return {mass_x / mass_sum, mass_y / mass_sum};
  }

  // ------------------------------------------------------------------------
  // Additional reward function for encouraging a stable standing posture.
  // This function computes a reward based on:
  //   - The error between the current torso height and a desired height.
  //   - The sum of the absolute roll and pitch (from the base orientation).
  //   - A penalty on the magnitude of linear and angular velocities.
  // A bonus is applied if the height and orientation errors are small, and a
  // large penalty is applied if the robot is considered to have fallen.
  // ------------------------------------------------------------------------
  mjtNum ComputeStandingReward() {
    // Parameters (tune these based on your experimental setup)
    const mjtNum desired_height = 0.35;  // target torso height (meters)
    const mjtNum height_weight = 5.0;   // weight for height error
    const mjtNum orientation_weight =
        2.0;  // weight for orientation error (roll + pitch)
    const mjtNum velocity_weight = 0.5;  // weight for penalizing movement
    const mjtNum fall_penalty = 10.0;    // penalty if the robot falls
    const mjtNum stability_threshold =
        0.05;  // error tolerance for bonus reward

    // Get current torso height (assuming data_->qpos[2] is the z-position)
    mjtNum current_height = data_->qpos[2];

    // Convert base orientation quaternion (stored in qpos[3]-qpos[6]) to Euler
    // angles. Note: Adjust the order if your model uses a different convention.
    Eigen::Quaternion<mjtNum> quat(data_->qpos[3], data_->qpos[4],
                                   data_->qpos[5], data_->qpos[6]);
    Eigen::Vector3<mjtNum> euler = quat.toRotationMatrix().eulerAngles(0, 1, 2);
    mjtNum roll = std::abs(euler[0]);
    mjtNum pitch = std::abs(euler[1]);

    // Compute errors.
    mjtNum height_error = std::abs(current_height - desired_height);
    mjtNum orientation_error = roll + pitch;

    // Compute base linear and angular velocities (from qvel; first 3 = linear,
    // next 3 = angular)
    mjtNum linear_velocity = std::sqrt(data_->qvel[0] * data_->qvel[0] +
                                       data_->qvel[1] * data_->qvel[1] +
                                       data_->qvel[2] * data_->qvel[2]);
    mjtNum angular_velocity = std::sqrt(data_->qvel[3] * data_->qvel[3] +
                                        data_->qvel[4] * data_->qvel[4] +
                                        data_->qvel[5] * data_->qvel[5]);
    mjtNum movement_penalty =
        velocity_weight * (linear_velocity + angular_velocity);

    // Base reward calculation (negative penalty on errors)
    mjtNum reward =
        -(height_weight * height_error +
          orientation_weight * orientation_error + movement_penalty)+10.0;

    // Add a bonus if the errors are within a small threshold (i.e., stable
    // posture)
    if (height_error < stability_threshold &&
        orientation_error < stability_threshold) {
      reward += 1.0;
    }

    // Apply a large penalty if the torso height indicates a fall
    if (current_height < 0.2) {
      reward -= fall_penalty;
    }

    return reward;
  }
  // ------------------------------------------------------------------------

  void WriteState(float reward, mjtNum xv, mjtNum yv, mjtNum ctrl_cost,
                  mjtNum contact_cost, mjtNum x_after, mjtNum y_after,
                  mjtNum healthy_reward) {
    State state = Allocate();
    state["reward"_] = reward;
    // Debug the state allocation - check if it's properly created
    // std::cout << "State obs buffer size: "
    // << state["obs"_].Shape()[0]  << state["obs"_].Shape()[1] << std::endl;
    mjtNum* obs = static_cast<mjtNum*>(state["obs"_].Data());
    mjtNum* obs_start = obs;  // Save the starting pointer for debugging

    // for (int i = no_pos_ ? 2 : 0; i < model_->nq; ++i) {
    // *(obs++) = data_->qpos[i];
    // std::cout << "obs data type: " << typeid(data_->qpos[i]).name() <<
    // std::endl;
    // }
    // for (int i = 0; i < model_->nv; ++i) {
    // *(obs++) = data_->qvel[i];
    // }
    // for (int i = 0; i < 10 * model_->nbody; ++i) {
    // *(obs++) = data_->cinert[i];
    // }
    // for (int i = 0; i < 6 * model_->nbody; ++i) {
    // *(obs++) = data_->cvel[i];
    // }
    // for (int i = 0; i < model_->nv; ++i) {
    // *(obs++) = data_->qfrc_actuator[i];
    // }
    // for (int i = 0; i < 6 * model_->nbody; ++i) {
    // *(obs++) = data_->cfrc_ext[i];
    // }
    // Create a temporary vector to hold all observations
    std::vector<mjtNum> all_obs;
    all_obs.reserve(3);  // Pre-allocate space for efficiency

    // Fill the vector with observations
    for (int i = 0; i < 3; ++i) {
      all_obs.push_back(data_->qpos[i]);
    }
    //  for (int i = 0; i < model_->nv; ++i) {
    //      all_obs.push_back(data_->qvel[i]);
    //  }
    //  for (int i = 0; i < 10 * model_->nbody; ++i) {
    //      all_obs.push_back(data_->cinert[i]);
    //  }
    //  for (int i = 0; i < 6 * model_->nbody; ++i) {
    //      all_obs.push_back(data_->cvel[i]);
    //  }
    //  for (int i = 0; i < model_->nv; ++i) {
    //      all_obs.push_back(data_->qfrc_actuator[i]);
    //  }
    //  for (int i = 0; i < 6 * model_->nbody; ++i) {
    //      all_obs.push_back(data_->cfrc_ext[i]);
    //  }

    // Now copy the vector to the state all at once
    state["obs"_].Assign(all_obs.data(), all_obs.size());

    // Print confirmation
    //  std::cout << "Filled " << all_obs.size() << " elements in observation
    //  array" << std::endl; std::cout << "First few values: "
    //            << all_obs[0] << " "
    //            << all_obs[1] << " "
    //            << all_obs[2] << " "
    //            << std::endl;
    //   // Print the first 30 observation values using the saved starting
    //   pointer std::cout << "obs values: "; for(int i=0; i<30; i++) {
    //   std::cout << obs_start[i] << " ";
    //   }
    //   std::cout << std::endl;

    // Add this to WriteState() after filling all observations
    int filled_elements = obs - obs_start;
    // std::cout << "Filled " << filled_elements << " elements in observation
    // array" << std::endl;
    state["info:reward_linvel"_] = xv * forward_reward_weight_;
    state["info:reward_quadctrl"_] = -ctrl_cost;
    state["info:reward_alive"_] = healthy_reward;
    state["info:reward_impact"_] = -contact_cost;
    state["info:x_position"_] = x_after;
    state["info:y_position"_] = y_after;
    state["info:distance_from_origin"_] =
        std::sqrt(x_after * x_after + y_after * y_after);
    state["info:x_velocity"_] = xv;
    state["info:y_velocity"_] = yv;

    mbc.setFeedback(data_);
    // Write to CSV only if csv_logging_enabled_ is set to 1.
  writeDataToCSV();

#ifdef ENVPOOL_TEST
    state["info:qpos0"_].Assign(qpos0_, model_->nq);
    state["info:qvel0"_].Assign(qvel0_, model_->nv);
#endif
  }

  void writeDataToCSV(int mode = 0) {
    if (csv_logging_enabled_ == 0) {
      return;  // Skip writing if logging is disabled
    }
    // Check if the file is open
    if (!outputFile.is_open()) {
      std::cerr << "Error: Unable to open CSV file for writing." << std::endl;
      return;
    }
    // Write the header only once
    if (outputFile.tellp() == 0) {
      outputFile << "body_x,body_y,body_z,"
                 << "quat_w,quat_x,quat_y,quat_z,"
                 << "FR_abad,FR_hip,FR_knee,"
                 << "FL_abad,FL_hip,FL_knee,"
                 << "HR_abad,HR_hip,HR_knee,"  
                 << "HL_abad,HL_hip,HL_knee" << std::endl;
    }
    // Write the data to CSV
    if (mode == 0) {
      // Write the data to CSV
      for (int i = 0; i < 19; ++i) {
        outputFile << double(data_->qpos[i]);
        if (i < 18) outputFile << ",";
      }
      outputFile << std::endl;
    } else if (mode == 1) {
      // Write the data to CSV
      for (int i = 0; i < 19; ++i) {
        outputFile << 0.2;
        if (i < 18) outputFile << ",";
      }
      outputFile << std::endl;
    }
    else if (mode == 2) {
      // Write the data to CSV
      for (int i = 0; i < 19; ++i) {
        outputFile << 0;
        if (i < 18) outputFile << ",";
      }
      outputFile << std::endl;
    }
  }
};

using HumanoidEnvPool = AsyncEnvPool<HumanoidEnv>;

}  // namespace mujoco_gym

#endif  // ENVPOOL_MUJOCO_GYM_HUMANOID_H_
