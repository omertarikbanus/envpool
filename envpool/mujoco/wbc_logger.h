#ifndef WBC_LOGGER_H
#define WBC_LOGGER_H

#include <array>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>

#include <Controllers/LegController.h>
#include <FSM/ControlFSMData.h>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <mujoco/mujoco.h>

// Standalone logger for WBC (Whole Body Control) data.
// Extracts logging functionality from ModelBasedControllerInterface.
class WBCLogger {
 public:
  static constexpr int kNumLegs = 4;

  // Body state in double precision for logging
  struct BodyStateD {
    Eigen::Vector3d pos_world{Eigen::Vector3d::Zero()};
    Eigen::Vector3d vel_world{Eigen::Vector3d::Zero()};
    Eigen::Vector3d vel_body{Eigen::Vector3d::Zero()};
    Eigen::Vector3d omega_body{Eigen::Vector3d::Zero()};
    Eigen::Vector3d omega_world{Eigen::Vector3d::Zero()};
    Eigen::Vector3d acc_body{Eigen::Vector3d::Zero()};
    Eigen::Vector3d acc_world{Eigen::Vector3d::Zero()};
    Eigen::Vector3d rpy{Eigen::Vector3d::Zero()};
    Eigen::Vector4d quat{Eigen::Vector4d::Zero()};
    Eigen::Matrix3d R_body_to_world{Eigen::Matrix3d::Identity()};
    bool from_sim_data{false};
  };

  struct ContactEstimates {
    std::array<double, kNumLegs> se{};
    std::array<double, kNumLegs> sim{};
  };

  struct FootKinematics {
    std::array<Eigen::Vector3d, kNumLegs> pos_local{};
    std::array<Eigen::Vector3d, kNumLegs> vel_local{};
    std::array<Eigen::Vector3d, kNumLegs> pos_world_se{};
    std::array<Eigen::Vector3d, kNumLegs> vel_world_se{};
    std::array<Eigen::Vector3d, kNumLegs> pos_world_sim{};
    std::array<Eigen::Vector3d, kNumLegs> vel_world_sim{};
  };

  struct GaitState {
    std::array<float, kNumLegs> contact_state{{1.0f, 1.0f, 1.0f, 1.0f}};
    std::array<float, kNumLegs> contact_phase{{0.0f, 0.0f, 0.0f, 0.0f}};
    std::array<float, kNumLegs> swing_phase{{0.0f, 0.0f, 0.0f, 0.0f}};
    float delta_theta{0.0f};
    float phase{0.0f};
    float cycle_time{0.0f};
  };

  WBCLogger() = default;
  ~WBCLogger() { close(); }

  // Non-copyable
  WBCLogger(const WBCLogger&) = delete;
  WBCLogger& operator=(const WBCLogger&) = delete;

  void clearFile(const std::string& filepath = "") {
    close();
    if (!filepath.empty()) {
      std::remove(filepath.c_str());
    } else if (!log_path_.empty()) {
      std::remove(log_path_.c_str());
    }
    log_path_.clear();
  }

  void close() {
    if (log_stream_.is_open()) {
      log_stream_.close();
    }
  }

  // Write a CSV row with WBC-related data.
  // Template parameters allow this to work with different state estimator types.
  template <typename StateEstimateT, typename LegCmdArray, typename LegDataArray,
            typename LocomotionStatePtr>
  void writeRow(const std::string& filepath,
                bool write_header_if_new,
                const StateEstimateT& se,
                const LegCmdArray& leg_commands,
                const LegDataArray& leg_data,
                const LocomotionCtrlData<float>& loco,
                const mjData* sim_data,
                const std::array<int, kNumLegs>& foot_body_ids,
                LocomotionStatePtr locomotion_state,
                const std::array<double, 12>& motor_commands,
                const GaitState& gait_state) {
    if (!prepareStream(filepath, write_header_if_new)) {
      return;
    }

    const auto body_est = makeEstimatorBodyState(se);
    const auto body_sim = makeSimBodyState(sim_data, body_est);
    const auto contact_est = computeContactEstimates(se, loco, sim_data, foot_body_ids);
    const auto foot_forces = computeReactionForces(loco, locomotion_state);
    const auto foot_kin = computeFootKinematics(leg_data, body_est, body_sim);
    const auto q_cmd = quatFromRPY(loco.pBody_RPY_des[0],
                                   loco.pBody_RPY_des[1],
                                   loco.pBody_RPY_des[2]);

    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss.precision(6);

    // Contact estimates
    for (int leg = 0; leg < kNumLegs; ++leg) oss << contact_est.se[leg] << ",";
    for (int leg = 0; leg < kNumLegs; ++leg) oss << contact_est.sim[leg] << ",";

    // Forces
    append3x4(oss, [&](int leg) { return loco.Fr_des[leg]; });
    append3x4(oss, [&](int leg) { return loco.Fr_se[leg]; });
    for (double force : foot_forces) oss << force << ",";

    // Body command
    for (int i = 0; i < 4; ++i) oss << static_cast<double>(q_cmd[i]) << ",";
    appendVec(oss, loco.pBody_des, 3);
    appendVec(oss, loco.vBody_des, 3);
    appendVec(oss, loco.vBody_Ori_des, 3);

    // Body states (estimator vs sim)
    appendVec(oss, body_est.pos_world, 3);
    appendVec(oss, body_sim.pos_world, 3);
    appendVec(oss, body_est.vel_world, 3);
    appendVec(oss, body_sim.vel_world, 3);
    appendVec(oss, body_est.vel_body, 3);
    appendVec(oss, body_sim.vel_body, 3);
    appendVec(oss, body_est.quat, 4);
    appendVec(oss, body_sim.quat, 4);
    appendVec(oss, body_est.rpy, 3);
    appendVec(oss, body_sim.rpy, 3);
    appendVec(oss, body_est.omega_body, 3);
    appendVec(oss, body_sim.omega_body, 3);
    appendVec(oss, body_est.omega_world, 3);
    appendVec(oss, body_sim.omega_world, 3);
    appendVec(oss, body_est.acc_body, 3);
    appendVec(oss, body_sim.acc_body, 3);
    appendVec(oss, body_est.acc_world, 3);
    appendVec(oss, body_sim.acc_world, 3);

    // Foot commands
    append3x4(oss, [&](int leg) { return loco.pFoot_des[leg]; });
    append3x4(oss, [&](int leg) { return loco.vFoot_des[leg]; });
    append3x4(oss, [&](int leg) { return loco.aFoot_des[leg]; });

    // foot_acc_numeric (zeros placeholder)
    for (int i = 0; i < 12; ++i) oss << 0.0 << ",";

    // Foot kinematics
    append3x4(oss, [&](int leg) { return foot_kin.pos_world_se[leg]; });
    append3x4(oss, [&](int leg) { return foot_kin.pos_world_sim[leg]; });
    append3x4(oss, [&](int leg) { return foot_kin.vel_world_se[leg]; });
    append3x4(oss, [&](int leg) { return foot_kin.vel_world_sim[leg]; });
    append3x4(oss, [&](int leg) { return foot_kin.pos_local[leg]; });
    append3x4(oss, [&](int leg) { return foot_kin.vel_local[leg]; });

    // Joint commands
    for (int leg = 0; leg < kNumLegs; ++leg) appendVec(oss, leg_commands[leg].qDes, 3);
    for (int leg = 0; leg < kNumLegs; ++leg) appendVec(oss, leg_commands[leg].qdDes, 3);

    // jacc_cmd (zeros placeholder)
    for (int i = 0; i < 12; ++i) oss << 0.0 << ",";

    // Feedforward torques
    for (int leg = 0; leg < kNumLegs; ++leg) {
      for (int axis = 0; axis < 3; ++axis) {
        oss << static_cast<double>(leg_commands[leg].tauFeedForward[axis]) << ",";
      }
    }

    // Motor commands
    for (double cmd : motor_commands) oss << cmd << ",";

    // Joint states
    for (int leg = 0; leg < kNumLegs; ++leg) appendVec(oss, leg_data[leg].q, 3);
    for (int leg = 0; leg < kNumLegs; ++leg) appendVec(oss, leg_data[leg].qd, 3);

    // Gait state
    oss << static_cast<double>(gait_state.delta_theta) << ","
        << static_cast<double>(gait_state.phase) << ","
        << static_cast<double>(gait_state.cycle_time) << ",";
    for (int leg = 0; leg < kNumLegs; ++leg) {
      oss << static_cast<double>(gait_state.contact_state[leg]) << ",";
    }
    for (int leg = 0; leg < kNumLegs; ++leg) {
      oss << static_cast<double>(gait_state.contact_phase[leg]) << ",";
    }
    for (int leg = 0; leg < kNumLegs; ++leg) {
      oss << static_cast<double>(gait_state.swing_phase[leg]) << ",";
    }

    // Vision (zeros placeholder)
    for (int i = 0; i < 3; ++i) oss << 0.0 << (i == 2 ? '\n' : ',');

    log_stream_ << oss.str();
    log_stream_.flush();
  }

 private:
  bool prepareStream(const std::string& filepath, bool write_header_if_new) {
    if (log_stream_.is_open() && filepath == log_path_) {
      return true;
    }
    if (log_stream_.is_open()) {
      log_stream_.close();
    }
    log_stream_.open(filepath.c_str(), std::ios::out | std::ios::app);
    if (!log_stream_) return false;
    log_path_ = filepath;
    if (write_header_if_new && log_stream_.tellp() == std::streampos(0)) {
      writeHeader();
    }
    return true;
  }

  void writeHeader() {
    if (!log_stream_) return;

    // Contact estimates
    log_stream_ << "contact_est_se_FR,contact_est_se_FL,contact_est_se_HR,"
                   "contact_est_se_HL,";
    log_stream_ << "contact_est_sim_FR,contact_est_sim_FL,contact_est_sim_"
                   "HR,contact_est_sim_HL,";

    // Forces
    for (int i = 0; i < 4; ++i)
      log_stream_ << "Fr_des_" << i << "_x,Fr_des_" << i << "_y,Fr_des_"
                  << i << "_z,";
    for (int i = 0; i < 4; ++i)
      log_stream_ << "Fr_se_" << i << "_x,Fr_se_" << i << "_y,Fr_se_"
                  << i << "_z,";
    for (int i = 0; i < 4; ++i)
      log_stream_ << "Fr_" << i << "_x,Fr_" << i << "_y,Fr_" << i << "_z,";

    // Body command
    log_stream_ << "body_ori_cmd_w,body_ori_cmd_x,body_ori_cmd_y,body_ori_cmd_z,";
    log_stream_ << "body_pos_cmd_x,body_pos_cmd_y,body_pos_cmd_z,";
    log_stream_ << "body_vel_cmd_x,body_vel_cmd_y,body_vel_cmd_z,";
    log_stream_ << "body_ang_vel_cmd_x,body_ang_vel_cmd_y,body_ang_vel_cmd_z,";

    // Body state (estimator vs sim)
    log_stream_ << "body_pos_se_x,body_pos_se_y,body_pos_se_z,";
    log_stream_ << "body_pos_sim_x,body_pos_sim_y,body_pos_sim_z,";
    log_stream_ << "body_vel_se_world_x,body_vel_se_world_y,body_vel_se_world_z,";
    log_stream_ << "body_vel_sim_world_x,body_vel_sim_world_y,body_vel_sim_world_z,";
    log_stream_ << "body_vel_se_body_x,body_vel_se_body_y,body_vel_se_body_z,";
    log_stream_ << "body_vel_sim_body_x,body_vel_sim_body_y,body_vel_sim_body_z,";
    log_stream_ << "body_ori_se_w,body_ori_se_x,body_ori_se_y,body_ori_se_z,";
    log_stream_ << "body_ori_sim_w,body_ori_sim_x,body_ori_sim_y,body_ori_sim_z,";
    log_stream_ << "body_rpy_se_r,body_rpy_se_p,body_rpy_se_y,";
    log_stream_ << "body_rpy_sim_r,body_rpy_sim_p,body_rpy_sim_y,";
    log_stream_ << "body_ang_vel_se_body_x,body_ang_vel_se_body_y,body_ang_vel_se_body_z,";
    log_stream_ << "body_ang_vel_sim_body_x,body_ang_vel_sim_body_y,body_ang_vel_sim_body_z,";
    log_stream_ << "body_ang_vel_se_world_x,body_ang_vel_se_world_y,body_ang_vel_se_world_z,";
    log_stream_ << "body_ang_vel_sim_world_x,body_ang_vel_sim_world_y,body_ang_vel_sim_world_z,";
    log_stream_ << "body_acc_se_body_x,body_acc_se_body_y,body_acc_se_body_z,";
    log_stream_ << "body_acc_sim_body_x,body_acc_sim_body_y,body_acc_sim_body_z,";
    log_stream_ << "body_acc_se_world_x,body_acc_se_world_y,body_acc_se_world_z,";
    log_stream_ << "body_acc_sim_world_x,body_acc_sim_world_y,body_acc_sim_world_z,";

    // Foot commands
    for (int i = 0; i < 4; ++i)
      log_stream_ << "foot_pos_cmd_" << i << "_x,foot_pos_cmd_" << i
                  << "_y,foot_pos_cmd_" << i << "_z,";
    for (int i = 0; i < 4; ++i)
      log_stream_ << "foot_vel_cmd_" << i << "_x,foot_vel_cmd_" << i
                  << "_y,foot_vel_cmd_" << i << "_z,";
    for (int i = 0; i < 4; ++i)
      log_stream_ << "foot_acc_cmd_" << i << "_x,foot_acc_cmd_" << i
                  << "_y,foot_acc_cmd_" << i << "_z,";

    // foot_acc_numeric
    for (int i = 0; i < 4; ++i)
      log_stream_ << "foot_acc_num_" << i << "_x,foot_acc_num_" << i
                  << "_y,foot_acc_num_" << i << "_z,";

    // Foot state (estimator vs sim)
    for (int i = 0; i < 4; ++i)
      log_stream_ << "foot_pos_se_" << i << "_x,foot_pos_se_" << i
                  << "_y,foot_pos_se_" << i << "_z,";
    for (int i = 0; i < 4; ++i)
      log_stream_ << "foot_pos_sim_" << i << "_x,foot_pos_sim_" << i
                  << "_y,foot_pos_sim_" << i << "_z,";
    for (int i = 0; i < 4; ++i)
      log_stream_ << "foot_vel_se_" << i << "_x,foot_vel_se_" << i
                  << "_y,foot_vel_se_" << i << "_z,";
    for (int i = 0; i < 4; ++i)
      log_stream_ << "foot_vel_sim_" << i << "_x,foot_vel_sim_" << i
                  << "_y,foot_vel_sim_" << i << "_z,";

    // Foot local (leg frame)
    for (int i = 0; i < 4; ++i)
      log_stream_ << "foot_local_pos_" << i << "_x,foot_local_pos_" << i
                  << "_y,foot_local_pos_" << i << "_z,";
    for (int i = 0; i < 4; ++i)
      log_stream_ << "foot_local_vel_" << i << "_x,foot_local_vel_" << i
                  << "_y,foot_local_vel_" << i << "_z,";

    // Joint commands
    for (int i = 0; i < 4; ++i)
      log_stream_ << "jpos_cmd_" << i << "_h,jpos_cmd_" << i
                  << "_k,jpos_cmd_" << i << "_a,";
    for (int i = 0; i < 4; ++i)
      log_stream_ << "jvel_cmd_" << i << "_h,jvel_cmd_" << i
                  << "_k,jvel_cmd_" << i << "_a,";
    for (int i = 0; i < 4; ++i)
      log_stream_ << "jacc_cmd_" << i << "_h,jacc_cmd_" << i
                  << "_k,jacc_cmd_" << i << "_a,";
    for (int i = 0; i < 4; ++i)
      log_stream_ << "tau_abad_ff_" << i << ",tau_hip_ff_" << i
                  << ",tau_knee_ff_" << i << ",";
    for (int i = 0; i < 4; ++i)
      log_stream_ << "motor_cmd_" << i << "_abad,motor_cmd_" << i
                  << "_hip,motor_cmd_" << i << "_knee,";

    // Joint state
    for (int i = 0; i < 4; ++i)
      log_stream_ << "jpos_" << i << "_h,jpos_" << i << "_k,jpos_" << i << "_a,";
    for (int i = 0; i < 4; ++i)
      log_stream_ << "jvel_" << i << "_h,jvel_" << i << "_k,jvel_" << i << "_a,";

    // Gait state
    log_stream_ << "gait_delta_theta,gait_phase,gait_cycle_s,";
    log_stream_ << "gait_contact_FR,gait_contact_FL,gait_contact_HR,gait_contact_HL,";
    log_stream_ << "gait_contact_phase_FR,gait_contact_phase_FL,gait_contact_phase_HR,gait_contact_phase_HL,";
    log_stream_ << "gait_swing_phase_FR,gait_swing_phase_FL,gait_swing_phase_HR,gait_swing_phase_HL,";

    // Vision
    log_stream_ << "vision_loc_x,vision_loc_y,vision_loc_z\n";
  }

  // Helper: Convert any 3-element vector-like to Eigen::Vector3d
  template <typename Vec3Like>
  static Eigen::Vector3d toVec3d(const Vec3Like& vec) {
    return Eigen::Vector3d(static_cast<double>(vec[0]),
                           static_cast<double>(vec[1]),
                           static_cast<double>(vec[2]));
  }

  // Helper: Append N elements from a vector to the stream
  template <typename Vec>
  static void appendVec(std::ostringstream& oss, const Vec& v, int n) {
    for (int i = 0; i < n; ++i) {
      oss << static_cast<double>(v[i]) << ",";
    }
  }

  // Helper: Append 3 elements for each of 4 legs
  template <typename Getter>
  static void append3x4(std::ostringstream& oss, Getter getter) {
    for (int leg = 0; leg < kNumLegs; ++leg) {
      const auto tmp = getter(leg);
      for (int i = 0; i < 3; ++i) {
        oss << static_cast<double>(tmp[i]) << ",";
      }
    }
  }

  // Convert RPY to quaternion (w, x, y, z)
  static Eigen::Matrix<float, 4, 1> quatFromRPY(float roll, float pitch, float yaw) {
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
  }

  // Build body state from state estimator result
  template <typename StateEstimateT>
  static BodyStateD makeEstimatorBodyState(const StateEstimateT& se) {
    BodyStateD out;
    out.R_body_to_world = se.rBody.transpose().template cast<double>();
    out.pos_world = toVec3d(se.position);
    out.vel_world = toVec3d(se.vWorld);
    out.vel_body = toVec3d(se.vBody);
    out.omega_body = toVec3d(se.omegaBody);
    out.omega_world = toVec3d(se.omegaWorld);
    out.acc_body = toVec3d(se.aBody);
    out.acc_world = toVec3d(se.aWorld);
    out.rpy = toVec3d(se.rpy);
    for (int i = 0; i < 4; ++i) {
      out.quat[i] = static_cast<double>(se.orientation[i]);
    }
    return out;
  }

  // Build body state from simulation data
  static BodyStateD makeSimBodyState(const mjData* sim_data,
                                     const BodyStateD& fallback) {
    if (!sim_data) {
      return fallback;
    }
    BodyStateD out;
    out.from_sim_data = true;
    out.pos_world << static_cast<double>(sim_data->qpos[0]),
        static_cast<double>(sim_data->qpos[1]),
        static_cast<double>(sim_data->qpos[2]);
    out.vel_world << static_cast<double>(sim_data->qvel[0]),
        static_cast<double>(sim_data->qvel[1]),
        static_cast<double>(sim_data->qvel[2]);
    out.omega_body << static_cast<double>(sim_data->qvel[3]),
        static_cast<double>(sim_data->qvel[4]),
        static_cast<double>(sim_data->qvel[5]);

    Eigen::Quaternion<mjtNum> quat_tmp(sim_data->qpos[3], sim_data->qpos[4],
                                       sim_data->qpos[5], sim_data->qpos[6]);
    quat_tmp.normalize();
    out.R_body_to_world = quat_tmp.toRotationMatrix();
    out.quat << quat_tmp.w(), quat_tmp.x(), quat_tmp.y(), quat_tmp.z();
    out.rpy = out.R_body_to_world.eulerAngles(0, 1, 2);

    out.vel_body = out.R_body_to_world.transpose() * out.vel_world;
    out.omega_world = out.R_body_to_world * out.omega_body;

    if (sim_data->qacc) {
      out.acc_world << static_cast<double>(sim_data->qacc[0]),
          static_cast<double>(sim_data->qacc[1]),
          static_cast<double>(sim_data->qacc[2]);
      out.acc_body = out.R_body_to_world.transpose() * out.acc_world;
    } else {
      out.acc_world.setZero();
      out.acc_body.setZero();
    }
    return out;
  }

  // Compute contact estimates from estimator and sim
  template <typename StateEstimateT>
  static ContactEstimates computeContactEstimates(
      const StateEstimateT& se,
      const LocomotionCtrlData<float>& loco,
      const mjData* sim_data,
      const std::array<int, kNumLegs>& foot_body_ids) {
    ContactEstimates out;
    for (int leg = 0; leg < kNumLegs; ++leg) {
      out.se[leg] = std::clamp(static_cast<double>(se.contactEstimate[leg]), 0.0, 1.0);
    }
    if (sim_data) {
      for (int leg = 0; leg < kNumLegs; ++leg) {
        out.sim[leg] = 0.0;
        const int body_id = foot_body_ids[leg];
        if (body_id >= 0) {
          const double fx = static_cast<double>(sim_data->cfrc_ext[6 * body_id + 0]);
          const double fy = static_cast<double>(sim_data->cfrc_ext[6 * body_id + 1]);
          const double fz = static_cast<double>(sim_data->cfrc_ext[6 * body_id + 2]);
          if (std::sqrt(fx * fx + fy * fy + fz * fz) > 5.0) {
            out.sim[leg] = 1.0;
          }
        }
      }
    } else {
      for (int leg = 0; leg < kNumLegs; ++leg) {
        out.sim[leg] = loco.contact_state[leg] > 0.0f ? 1.0 : 0.0;
      }
    }
    return out;
  }

  // Compute reaction forces from WBC controller
  template <typename LocomotionStatePtr>
  static std::array<double, kNumLegs * 3> computeReactionForces(
      const LocomotionCtrlData<float>& loco,
      LocomotionStatePtr locomotion_state) {
    std::array<double, kNumLegs * 3> forces{};
    if (locomotion_state) {
      if (auto* wbc_ctrl = locomotion_state->getWbcCtrl()) {
        const auto& fr = wbc_ctrl->getReactionForces();
        std::size_t contact_idx = 0;
        for (int leg = 0; leg < kNumLegs; ++leg) {
          if (loco.contact_state[leg] > 0.5f) {
            for (int axis = 0; axis < 3; ++axis) {
              const std::size_t fr_idx = contact_idx * 3 + axis;
              if (fr_idx < static_cast<std::size_t>(fr.size())) {
                forces[leg * 3 + axis] = static_cast<double>(fr[fr_idx]);
              }
            }
            ++contact_idx;
          }
        }
      }
    }
    return forces;
  }

  // Compute foot kinematics from leg data and body states
  template <typename LegDataArray>
  static FootKinematics computeFootKinematics(const LegDataArray& leg_data,
                                              const BodyStateD& body_est,
                                              const BodyStateD& body_sim) {
    FootKinematics kin;
    const BodyStateD& sim_or_est = body_sim.from_sim_data ? body_sim : body_est;

    for (int leg = 0; leg < kNumLegs; ++leg) {
      kin.pos_local[leg] = toVec3d(leg_data[leg].p);
      kin.vel_local[leg] = toVec3d(leg_data[leg].v);
      Eigen::Vector3d hip = Eigen::Vector3d::Zero();
      if (leg_data[leg].quadruped) {
        hip = toVec3d(leg_data[leg].quadruped->getHipLocation(leg));
      }
      const Eigen::Vector3d foot_body = hip + kin.pos_local[leg];

      const Eigen::Vector3d vel_body_est =
          kin.vel_local[leg] + body_est.omega_body.cross(foot_body);
      kin.pos_world_se[leg] =
          body_est.pos_world + body_est.R_body_to_world * foot_body;
      kin.vel_world_se[leg] =
          body_est.vel_world + body_est.R_body_to_world * vel_body_est;

      const Eigen::Vector3d vel_body_sim =
          kin.vel_local[leg] + sim_or_est.omega_body.cross(foot_body);
      kin.pos_world_sim[leg] =
          sim_or_est.pos_world + sim_or_est.R_body_to_world * foot_body;
      kin.vel_world_sim[leg] =
          sim_or_est.vel_world + sim_or_est.R_body_to_world * vel_body_sim;
    }
    return kin;
  }

  std::ofstream log_stream_;
  std::string log_path_;
};

#endif  // WBC_LOGGER_H
