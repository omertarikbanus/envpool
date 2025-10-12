#ifndef ENVPOOL_MUJOCO_GYM_HUMANOID_H_
#define ENVPOOL_MUJOCO_GYM_HUMANOID_H_

#include <algorithm>
#include <cmath>  // For std::sqrt, std::abs
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "envpool/core/async_envpool.h"
#include "envpool/core/env.h"
#include "envpool/mujoco/gym/mujoco_env.h"

#include <eigen3/Eigen/Dense>
#include <RobotRunner.h>
#include <RobotController.h>
#include <Controllers/EmbeddedController.hpp>
#include <Utilities/RobotCommands.h>
#include "mbc_interface.h"

namespace mujoco_gym {

class HumanoidEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict(
        "frame_skip"_.Bind(15), "post_constraint"_.Bind(true),
        "use_contact_force"_.Bind(false), "forward_reward_weight"_.Bind(1),
        "terminate_when_unhealthy"_.Bind(true),
        "render_mode"_.Bind(false),
        "csv_logging_enabled"_.Bind(false), 
        "exclude_current_positions_from_observation"_.Bind(true),
        "ctrl_cost_weight"_.Bind(2e-4), "healthy_reward"_.Bind(1.0),
        "healthy_z_min"_.Bind(0.20), "healthy_z_max"_.Bind(0.75),
        "contact_cost_weight"_.Bind(5e-7), "contact_cost_max"_.Bind(10.0),
        "velocity_tracking_weight"_.Bind(5.0),
        "yaw_tracking_weight"_.Bind(0.2),
        "orientation_penalty_weight"_.Bind(5.0),
        "height_penalty_weight"_.Bind(10.0),
        "foot_slip_penalty_weight"_.Bind(0.5),
        "reset_noise_scale"_.Bind(0));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    mjtNum inf = std::numeric_limits<mjtNum>::infinity();
    bool no_pos = conf["exclude_current_positions_from_observation"_];
    return MakeDict("obs"_.Bind(Spec<mjtNum>({45}, {-inf, inf})),
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
    // Action layout (30 dims total, with 1 unused slot per leg to match current indexing):
    // [0] vBody_des.x, [1] vBody_des.y, [2] yaw_rate_des,
    // Per-leg (4 legs) stride of 7 each (N_FOOT_PARAM_ACT):
    //   [3 + 7*leg + 0..2] pFoot_des[leg] (x, y, z)
    //   [3 + 7*leg + 3..5] Fr_des[leg] (x, y, z)  -- currently zeroed in WBC
    //   [3 + 7*leg + 6]    unused (reserved)
    // Total reserved: 3 + 4 * 7 = 31 slots (0..30), but the highest index read is 29,
    // so dimension 30 is sufficient for current access pattern.
    return MakeDict("action"_.Bind(Spec<mjtNum>({-1, 15}, {-1, 1})));
  }
};

using HumanoidEnvSpec = EnvSpec<HumanoidEnvFns>;

class HumanoidEnv : public Env<HumanoidEnvSpec>, public MujocoEnv {
 protected:
  bool terminate_when_unhealthy_, no_pos_, use_contact_force_, render_mode_, csv_logging_enabled_;
  mjtNum ctrl_cost_weight_, forward_reward_weight_, healthy_reward_;
  mjtNum healthy_z_min_, healthy_z_max_;
  mjtNum contact_cost_weight_, contact_cost_max_;
  std::uniform_real_distribution<> dist_;
  ModelBasedControllerInterface mbc;
  // New: Backup of the controller

  std::ofstream outputFile;
  float lastReward = 0;
  float lastAction = 0;
  float prev_action = 0.0;
  mjtNum desired_h;
  mjtNum velocity_tracking_weight_;
  mjtNum yaw_tracking_weight_;
  mjtNum orientation_penalty_weight_;
  mjtNum height_penalty_weight_;
  mjtNum foot_slip_penalty_weight_;
  std::vector<mjtNum> last_action_vector_;
  std::vector<mjtNum> last_observation_;
  mjtNum last_ctrl_cost_{0.0};
  mjtNum last_contact_cost_{0.0};
  mjtNum last_velocity_tracking_cost_{0.0};
  mjtNum last_yaw_rate_tracking_cost_{0.0};
  mjtNum last_healthy_reward_{0.0};
  mjtNum last_x_velocity_{0.0};
  mjtNum last_y_velocity_{0.0};
  mjtNum last_x_position_{0.0};
  mjtNum last_y_position_{0.0};
  mjtNum last_orientation_penalty_{0.0};
  mjtNum last_height_penalty_{0.0};
  mjtNum last_foot_slip_penalty_{0.0};
  bool last_is_healthy_{true};
  bool pending_reset_marker_{true};
  // Added: Torque bound constant (example value; adjust as needed)
  const mjtNum torque_limit_ = 65.0;
  // Added: CSV logging switch (set to 1 manually to enable CSV writing)
  std::string csv_filename_;
  std::string wbc_csv_filename_;

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
        render_mode_(spec.config["render_mode"_]),
        csv_logging_enabled_(spec.config["csv_logging_enabled"_]),
        no_pos_(spec.config["exclude_current_positions_from_observation"_]),
        use_contact_force_(spec.config["use_contact_force"_]),
        ctrl_cost_weight_(spec.config["ctrl_cost_weight"_]),
        forward_reward_weight_(spec.config["forward_reward_weight"_]),
        healthy_reward_(spec.config["healthy_reward"_]),
        healthy_z_min_(spec.config["healthy_z_min"_]),
        healthy_z_max_(spec.config["healthy_z_max"_]),
        contact_cost_weight_(spec.config["contact_cost_weight"_]),
        contact_cost_max_(spec.config["contact_cost_max"_]),
        velocity_tracking_weight_(spec.config["velocity_tracking_weight"_]),
        yaw_tracking_weight_(spec.config["yaw_tracking_weight"_]),
        orientation_penalty_weight_(spec.config["orientation_penalty_weight"_]),
        height_penalty_weight_(spec.config["height_penalty_weight"_]),
        foot_slip_penalty_weight_(spec.config["foot_slip_penalty_weight"_]),
        dist_(-spec.config["reset_noise_scale"_],
              spec.config["reset_noise_scale"_]) {
    if (env_id_ == 0 ) {
      csv_logging_enabled_ = 1;
    }

    if (render_mode_) {
      EnableRender(true);

    } else {
      
    }
  csv_filename_ = "/app/envpool/logs/" + std::to_string(env_id_) + "_log.csv";
  wbc_csv_filename_ = "/app/envpool/logs/" + std::to_string(env_id_) + "_wbc.csv";
  ClearCsvLogs();

    MujocoReset();
    mbc.setModeLocomotion();
    if (csv_logging_enabled_) {
      mbc.writeWBCLogCSV(wbc_csv_filename_, /*write_header_if_new=*/true);
    }

    desired_h = 0.35;  // Set the desired height to 0.35
    mbc.setBaseHeight(static_cast<float>(desired_h));

    // Save the initial controller configuration to backup
    // while (mbc.getMode() != 6) {
    //   mjtNum dummy_action[22];
    //   // set all values to zero;
    //   for (int i = 0; i < 22; i++) {
    //     dummy_action[i] = 0;
    //   }
    //   mjtNum* act = dummy_action;
    //   mbc.setFeedback(data_);
    //   mbc.setAction(act);
    //   mbc.run();

    //   mjtNum motor_commands[12];
    //   std::array<double, 12> mc = mbc.getMotorCommands();
    //   for (int i = 0; i < 12; ++i) {
    //     // Clamp motor commands to torque bounds.
    //     motor_commands[i] =
    //         std::max(std::min(static_cast<mjtNum>(mc[i]), torque_limit_),
    //                  -torque_limit_);
    //   }
    //   const auto& before = GetMassCenter();

    //   MujocoStep(motor_commands);

    //   writeDataToCSV(0);
    // }
    writeDataToCSV(2);
    // model_backup_ = mj_copyModel(nullptr, model_);
    // data_backup_ = mj_copyData(nullptr, model_, data_);
  }

  void MujocoResetModel() override {
    setIC();


#ifdef ENVPOOL_TEST
    std::memcpy(qpos0_, data_->qpos, sizeof(mjtNum) * model_->nq);
    std::memcpy(qvel0_, data_->qvel, sizeof(mjtNum) * model_->nv);
#endif
  }

  bool IsDone() override { return done_; }

  void Reset() override {
    writeDataToCSV(1);
    last_action_vector_.clear();
    last_observation_.clear();
    if(render_mode_)
      std::cout << "Resetting the environment..." << std::endl;
    done_ = false;
    elapsed_step_ = 0;
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<mjtNum> dis( 0.32, 0.4);
    desired_h = dis(gen);
    mbc.setBaseHeight(static_cast<float>(desired_h));
    // MujocoReset();
    // mj_deleteData(data_);
    // mj_deleteModel(model_);
    // model_ = mj_copyModel(nullptr, model_backup_);
    // data_ = mj_copyData(nullptr, model_, data_backup_);
    mj_resetData(model_, data_);
    setIC();
    WriteState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    mbc.setFeedback(data_);
    mbc.setModeLocomotion();
    mbc. _controller->_controlFSM->data.locomotionCtrlData.pBody_des[2] =
        static_cast<float>(desired_h);
    if (csv_logging_enabled_) {
      mbc.writeWBCLogCSV(wbc_csv_filename_, /*write_header_if_new=*/true);
    }

  }

  void ClearCsvLogs() {
    if (csv_logging_enabled_) {
      if (outputFile.is_open()) {
        outputFile.close();
      }
      if (!csv_filename_.empty()) {
        std::remove(csv_filename_.c_str());
      }
      outputFile.open(csv_filename_.c_str(), std::ios::out | std::ios::trunc);
      if (!outputFile.is_open()) {
        std::cerr << "Error: Unable to recreate CSV log file at " << csv_filename_ << std::endl;
      }
      pending_reset_marker_ = true;
    }

    mbc.clearWBCLogFile(wbc_csv_filename_);
    if (csv_logging_enabled_) {
      mbc.writeWBCLogCSV(wbc_csv_filename_, /*write_header_if_new=*/true);
    }
  }

  void Step(const Action& action) override {
    // step
    auto action_array = action["action"_];
    auto* act = static_cast<mjtNum*>(action_array.Data());
    std::size_t action_count = action_array.size;
    bool invalid_action = false;
    for (std::size_t idx = 0; idx < action_count; ++idx) {
      if (!std::isfinite(static_cast<double>(act[idx]))) {
        act[idx] = 0.0;
        invalid_action = true;
      }
    }
    if (invalid_action && env_id_ == 0) {
      std::cerr << "[HumanoidEnv] Non-finite action detected; replaced with zeros." << std::endl;
    }
    lastAction = action_count > 0 ? static_cast<float>(act[0]) : 0.0f;
    last_action_vector_.assign(act, act + action_count);
    // repeat frameskip times
    mjtNum ctrl_cost = 0.0;
    const auto& before = GetMassCenter();

    for (int i = 0; i < frame_skip_; ++i) {
      mbc.frame_skip_ = frame_skip_;
      mbc.elapsed_step_ = elapsed_step_;
      mbc.setAction(act);
      mbc.setFeedback(data_);
      mbc.run();

      mjtNum motor_commands[12];
      std::array<double, 12> mc = mbc.getMotorCommands();
      bool invalid_motor_command = false;
      for (int j = 0; j < 12; ++j) {
        mjtNum candidate = static_cast<mjtNum>(mc[j]);
        if (!std::isfinite(static_cast<double>(candidate))) {
          candidate = 0.0;
          invalid_motor_command = true;
        }
        candidate = std::clamp(candidate, -torque_limit_, torque_limit_);
        motor_commands[j] = candidate;
        ctrl_cost += ctrl_cost_weight_ * motor_commands[j] * motor_commands[j];
      }
      if (invalid_motor_command && env_id_ == 0) {
        std::cerr << "[HumanoidEnv] Non-finite motor command detected; clamped to zero." << std::endl;
      }
      MujocoStep(motor_commands);
    }

    if (csv_logging_enabled_) {
      mbc.writeWBCLogCSV(wbc_csv_filename_, /*write_header_if_new=*/true);
    }

    const auto& after = GetMassCenter();

    // Compute velocities.
    // std::cout << "Model nu: " << model_->nu << std::endl;
    // for (int i = 0; i < model_->nu; ++i) {
    //   ctrl_cost += ctrl_cost_weight_ * act[i] * act[i];
    // }
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

    // if(elapsed_step_ % frame_skip_ * 15 == 0){
    //   static std::random_device rd;
    //   static std::mt19937 gen(rd());
    //   // normal distribution with 0 mean and 0.003 stddev
    //   std::normal_distribution<mjtNum> dis( 0.00, 0.003);
    //   desired_h += dis(gen);
    //   //clamp desired_h to [0.25, 0.4]
    //   desired_h = std::max(0.32, std::min(0.4, desired_h));
    // }
    // reward and done
    mjtNum healthy_reward =
        terminate_when_unhealthy_ || IsHealthy() ? healthy_reward_ : 0.0;
    // Simple locomotion reward: track desired planar velocity and yaw rate from WBC
    mjtNum velocity_cost = 0.0;
    mjtNum yaw_rate_cost = 0.0;
    bool is_healthy = true;
    auto reward =
        ComputeLocomotionReward(xv, yv, healthy_reward, velocity_cost,
                                yaw_rate_cost, is_healthy);

    mjtNum orientation_penalty = ComputeOrientationPenalty();
    mjtNum height_penalty = ComputeHeightPenalty();
    mjtNum foot_slip_penalty = ComputeFootSlipPenalty();

    // reward -= (ctrl_cost + contact_cost + orientation_penalty + height_penalty +
    //            foot_slip_penalty);

    if (!std::isfinite(static_cast<double>(reward))) {
      if (env_id_ == 0) {
        std::cerr << "[HumanoidEnv] Non-finite reward detected; forcing termination." << std::endl;
      }
      reward = -5.0;
      done_ = true;
    }

    last_ctrl_cost_ = ctrl_cost;
    last_contact_cost_ = contact_cost;
    last_orientation_penalty_ = orientation_penalty;
    last_height_penalty_ = height_penalty;
    last_foot_slip_penalty_ = foot_slip_penalty;
    last_velocity_tracking_cost_ = velocity_cost;
    last_yaw_rate_tracking_cost_ = yaw_rate_cost;
    last_healthy_reward_ = healthy_reward;
    last_x_velocity_ = xv;
    last_y_velocity_ = yv;
    last_x_position_ = after[0];
    last_y_position_ = after[1];
    last_is_healthy_ = is_healthy;

    // ++elapsed_step_;
    done_ = done_ || (terminate_when_unhealthy_ ? !IsHealthy() : false) ||
            (elapsed_step_ >= max_episode_steps_);
    WriteState(reward, xv, yv, ctrl_cost, contact_cost, after[0], after[1],
               healthy_reward);
  }

 private:
  bool IsHealthy() {
    if (mbc._controller->_controlFSM->data.controlParameters->control_mode ==
        0) {
      std::cout << "[IsHealthy] Passive state detected. env_id: " << env_id_ << std::endl;
      return false;  // end if state is passive
    }

    bool healthy =
        (healthy_z_min_ < data_->qpos[2]) && (data_->qpos[2] < healthy_z_max_);
    if (!healthy && render_mode_) {
      std::cout << "[IsHealthy] Unhealthy state detected: "
                << "z position = " << data_->qpos[2]
                << ", healthy_z_min = " << healthy_z_min_
                << ", healthy_z_max = " << healthy_z_max_ << std::endl;
    }
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
    // ---------- Tunable constants ----------
    // Desired height generated every reset
    // static mjtNum desired_h = 0.35;  // Example value,
    const mjtNum height_w  = 400.0;          // (= 4 / 0.01^2)

    const mjtNum orient_w  = 0.0;            // cost per deg^2 * 0.01
    const mjtNum vel_w     =  0.00;           // per (m/s)^2 or (rad/s)^2
    const mjtNum fall_pen  = 500.0;          // one-off
    const mjtNum bonus_eps_h = 0.03;         // m
    const mjtNum bonus_eps_o = 1.0* M_PI/180;// rad
    const mjtNum bonus_eps_v = 0.01;         // 
    // ---------------------------------------

    // Height
    mjtNum h = data_->qpos[2];
    mjtNum height_err = h - desired_h;

    // Orientation: quaternion → roll,pitch
    Eigen::Quaternion<mjtNum> q(data_->qpos[6], data_->qpos[3],
                                data_->qpos[4], data_->qpos[5]);
    Eigen::Vector3<mjtNum> eul = q.toRotationMatrix().eulerAngles(0, 1, 2);
    mjtNum roll  = eul[0];
    mjtNum pitch = eul[1];

    // Velocities
    mjtNum lin_v = std::sqrt(data_->qvel[0] * data_->qvel[0] +
                             data_->qvel[1] * data_->qvel[1] +
                             data_->qvel[2] * data_->qvel[2]);
    mjtNum ang_v = std::sqrt(data_->qvel[3] * data_->qvel[3] +
                             data_->qvel[4] * data_->qvel[4] +
                             data_->qvel[5] * data_->qvel[5]);

    // Costs (squared errors)
    mjtNum height_cost = height_w * height_err * height_err;
    mjtNum orient_cost = orient_w * (roll * roll + pitch * pitch);  // rad²
    mjtNum move_cost   = vel_w * (lin_v * lin_v + ang_v * ang_v);
    mjtNum diff = lastAction - prev_action;
    prev_action = lastAction;
    mjtNum action_smooth = 3.0 * diff * diff;

    // state["info:height_cost"_] = height_cost;
    // state["info:orient_cost"_] = orient_cost;
    // state["info:move_cost"_] = move_cost;
    // Debug: print individual costs
    // std::cout << "[StandingReward] height_cost=" << height_cost
    //       << " orient_cost=" << orient_cost
    //       << " move_cost=" << move_cost << std::endl;
    // Alive bonus
      mjtNum reward = .3 - height_cost - orient_cost - move_cost - 0.05 * action_smooth;
      // std::cout << "[StandingReward] Initial reward: " << reward << std::endl;
    // Extra stillness bonus
    if (std::abs(height_err) < bonus_eps_h &&
        std::abs(roll)       < bonus_eps_o &&
        std::abs(pitch)      < bonus_eps_o &&
        lin_v                < bonus_eps_v &&
        ang_v                < bonus_eps_v)
      reward += 1.0;
    // Penalty for falling
    bool healthy = IsHealthy();                 // your existing function
    if (h < 0.20 || h > 0.45 || !healthy) {                 // ← single '!'
      // std::cout << "[StandingReward] Penalty for falling!" << std::endl;
      // state["info:fall_pen"_] = 1.0;  // Debug: indicate a fall
      reward -= fall_pen;                       // one-off? see below
      // optionally end the episode here
    }
    // if (h > 0.3 && h < 0.4 && healthy) {

    //   reward += 2.5;  // Bonus for being in the healthy range
    //   // state["info:fall_pen"_] = 0.0;  // Debug: no fall
    // }

    return reward;
  }

  // ------------------------------------------------------------------------

  // Simple locomotion reward that tracks WBC references.
  // Components:
  //  - Velocity tracking: (vx, vy) to vBody_des[0:2]
  //  - Yaw rate tracking: qvel[5] to vBody_Ori_des[2]
  //  - Alive bonus (healthy_reward) and fall penalty
  mjtNum ComputeLocomotionReward(mjtNum xv, mjtNum yv,
                                 mjtNum healthy_reward,
                                 mjtNum& velocity_cost,
                                 mjtNum& yaw_rate_cost, bool& is_healthy) {
    const mjtNum fall_pen = 10.0;

    // Desired references from WBC (set via mbc_interface::setAction)
    const mjtNum vx_des = mbc._controller->_controlFSM->data.locomotionCtrlData.vBody_des[0];
    const mjtNum vy_des = mbc._controller->_controlFSM->data.locomotionCtrlData.vBody_des[1];
    const mjtNum wz_des = mbc._controller->_controlFSM->data.locomotionCtrlData.vBody_Ori_des[2];

    // Measure yaw rate from generalized velocity (z-axis angular vel)
    const mjtNum wz = data_->qvel[5];

    const mjtNum vx_scale = std::max(std::abs(vx_des), static_cast<mjtNum>(0.2));
    const mjtNum vy_scale = std::max(std::abs(vy_des), static_cast<mjtNum>(0.2));
    const mjtNum wz_scale = std::max(std::abs(wz_des), static_cast<mjtNum>(0.3));

    const mjtNum norm_vx = (xv - vx_des) / vx_scale;
    const mjtNum norm_vy = (yv - vy_des) / vy_scale;
    const mjtNum norm_wz = (wz - wz_des) / wz_scale;

    velocity_cost = velocity_tracking_weight_ * (norm_vx * norm_vx + norm_vy * norm_vy);
    yaw_rate_cost = yaw_tracking_weight_ * (norm_wz * norm_wz);

    mjtNum reward = healthy_reward ;// - (velocity_cost + yaw_rate_cost);

    // Mild forward incentive to avoid standing still when healthy.
    // reward += forward_reward_weight_ * xv;

    // Large penalty on fall/unhealthy
    is_healthy = IsHealthy();
    if (!is_healthy) {
      reward -= fall_pen;
    }

    return reward;
  }

  mjtNum ComputeOrientationPenalty() const {
    Eigen::Quaternion<mjtNum> q(data_->qpos[6], data_->qpos[3], data_->qpos[4],
                                data_->qpos[5]);
    Eigen::Vector3<mjtNum> eul = q.toRotationMatrix().eulerAngles(0, 1, 2);
    mjtNum roll = eul[0];
    mjtNum pitch = eul[1];
    return orientation_penalty_weight_ * (roll * roll + pitch * pitch);
  }

  mjtNum ComputeHeightPenalty() const {
    const mjtNum height = data_->qpos[2];
    const mjtNum desired =
        static_cast<mjtNum>(mbc._controller->_controlFSM->data.locomotionCtrlData.pBody_des[2]);
    const mjtNum err = height - desired;
    return height_penalty_weight_ * err * err;
  }

  mjtNum ComputeFootSlipPenalty() const {
    mjtNum slip_cost = 0.0;
    for (int leg = 0; leg < 4; ++leg) {
      if (mbc._controller->_controlFSM->data.locomotionCtrlData.contact_state[leg] > 0.5f) {
        const auto& leg_data =
            mbc._controller->_controlFSM->data._legController->datas[leg];
        const mjtNum vx = static_cast<mjtNum>(leg_data.v[0]);
        const mjtNum vy = static_cast<mjtNum>(leg_data.v[1]);
        const mjtNum vz = static_cast<mjtNum>(leg_data.v[2]);
        slip_cost += vx * vx + vy * vy + vz * vz;
      }
    }
    return foot_slip_penalty_weight_ * slip_cost;
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
    auto obs_array = state["obs"_];
    auto* obs = static_cast<mjtNum*>(obs_array.Data());
    mjtNum* obs_start = obs;  // Save the starting pointer for debugging

    mbc.setObservation(obs);
    obs[44] = desired_h;
    bool invalid_obs = false;
    for (std::size_t idx = 0; idx < obs_array.size; ++idx) {
      if (!std::isfinite(static_cast<double>(obs[idx]))) {
        obs[idx] = 0.0;
        invalid_obs = true;
      }
    }
    if (invalid_obs && env_id_ == 0) {
      std::cerr << "[HumanoidEnv] Non-finite observation detected; replaced with zeros." << std::endl;
    }
    if (obs_array.size > 0) {
      last_observation_.assign(obs_start, obs_start + obs_array.size);
    } else {
      last_observation_.clear();
    }
    obs_array.Assign(obs, 1);

    // Print confirmation
    // std::cout << "Filled " << 42 << " elements in observation array" << std::endl;
    // std::cout << "First few values: "
    //           << obs[0] << " "
    //           << obs[1] << " "
    //           << obs[2] << " "
    //           << std::endl;
    //   // Print the first 30 observation values using the saved starting pointer
    //   std::cout << "obs values: "; for(int i=0; i<30; i++) {
    //   std::cout << obs_start[i] << " ";
    //   }
    //   std::cout << std::endl;

    // Add this to WriteState() after filling all observations
    int filled_elements = obs - obs_start;
    // std::cout << "Filled " << filled_elements << " elements in observation
    // array" << std::endl;
    // state["info:reward_linvel"_] = xv * forward_reward_weight_;
    // state["info:reward_quadctrl"_] = -ctrl_cost;
    // state["info:reward_alive"_] = healthy_reward;
    // state["info:reward_impact"_] = -contact_cost;
    // state["info:x_position"_] = x_after;
    // state["info:y_position"_] = y_after;
    // state["info:distance_from_origin"_] =
    //     std::sqrt(x_after * x_after + y_after * y_after);
    // state["info:x_velocity"_] = xv;
    // state["info:y_velocity"_] = yv;

    lastReward = reward;
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
    if (!outputFile.is_open()) {
      std::cerr << "Error: Unable to open CSV file for writing." << std::endl;
      return;
    }

    if (mode == 1 || mode == 2) {
      pending_reset_marker_ = true;
      return;
    }
    if (mode != 0) {
      return;
    }

    if (last_observation_.empty() || last_action_vector_.empty()) {
      return;  // Skip until we have both action and observation data
    }

    if (outputFile.tellp() == 0) {
      outputFile << "elapsed_step,reset_marker,body_x,body_y,body_z,"
                 << "quat_w,quat_x,quat_y,quat_z,"
                 << "FR_abad,FR_hip,FR_knee,"
                 << "FL_abad,FL_hip,FL_knee,"
                 << "HR_abad,HR_hip,HR_knee,"
                 << "HL_abad,HL_hip,HL_knee,"
                 << "reward_total,healthy_reward,velocity_tracking_penalty,"
                 << "yaw_rate_tracking_penalty,orientation_penalty,height_penalty,foot_slip_penalty,"
                 << "ctrl_cost,contact_cost,is_healthy,"
                 << "x_position,y_position,x_velocity,y_velocity,"
                 << "pBody_des_z,desired_h";
      for (std::size_t i = 0; i < last_action_vector_.size(); ++i) {
        outputFile << ",action_" << i;
      }
      for (std::size_t i = 0; i < last_observation_.size(); ++i) {
        outputFile << ",obs_" << i;
      }
      outputFile << std::endl;
    }

    const bool reset_marker = pending_reset_marker_;
    pending_reset_marker_ = false;

    outputFile << elapsed_step_ << ',' << (reset_marker ? 1 : 0);
    for (int i = 0; i < 19; ++i) {
      outputFile << ',' << static_cast<double>(data_->qpos[i]);
    }
    outputFile << ',' << static_cast<double>(lastReward)
               << ',' << static_cast<double>(last_healthy_reward_)
               << ',' << static_cast<double>(last_velocity_tracking_cost_)
               << ',' << static_cast<double>(last_yaw_rate_tracking_cost_)
               << ',' << static_cast<double>(last_orientation_penalty_)
               << ',' << static_cast<double>(last_height_penalty_)
               << ',' << static_cast<double>(last_foot_slip_penalty_)
               << ',' << static_cast<double>(last_ctrl_cost_)
               << ',' << static_cast<double>(last_contact_cost_)
               << ',' << (last_is_healthy_ ? 1 : 0)
               << ',' << static_cast<double>(last_x_position_)
               << ',' << static_cast<double>(last_y_position_)
               << ',' << static_cast<double>(last_x_velocity_)
               << ',' << static_cast<double>(last_y_velocity_)
               << ','
               << static_cast<double>(mbc._controller->_controlFSM->data.locomotionCtrlData.pBody_des[2])
               << ',' << static_cast<double>(desired_h);

    for (std::size_t i = 0; i < last_action_vector_.size(); ++i) {
      outputFile << ',' << static_cast<double>(last_action_vector_[i]);
    }
    for (std::size_t i = 0; i < last_observation_.size(); ++i) {
      outputFile << ',' << static_cast<double>(last_observation_[i]);
    }
    outputFile << std::endl;
  }
};

using HumanoidEnvPool = AsyncEnvPool<HumanoidEnv>;

}  // namespace mujoco_gym

#endif  // ENVPOOL_MUJOCO_GYM_HUMANOID_H_
