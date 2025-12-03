/*
 * Copyright 2022 Garena Online Private Limited
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ENVPOOL_MUJOCO_GYM_MUJOCO_ENV_H_
#define ENVPOOL_MUJOCO_GYM_MUJOCO_ENV_H_
#include <GL/osmesa.h>
#include <mjxmacro.h>
#include <mujoco.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>   // FILE*
#include <cstring>  // std::memcpy
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static OSMesaContext ctx;
static unsigned char fb[1920 * 1080 * 4];  // off‑screen framebuffer

namespace mujoco_gym {

static void write_all(FILE* pipe, const unsigned char* buf, size_t nbytes) {
  while (nbytes) {
    size_t n = fwrite(buf, 1, nbytes, pipe);
    if (n != 1920)  // pipe closed or broken
      std::cerr << "fwrite to ffmpeg pipe failed" << std::endl;
    buf += n;
    nbytes -= n;
  }
}
class MujocoEnv {
 private:
  std::array<char, 1000> error_{};

 protected:
  mjModel* model_;
  mjData* data_;
  mjModel* model_backup_;
  mjData* data_backup_;
  mjtNum *init_qpos_, *init_qvel_;
#ifdef ENVPOOL_TEST
  mjtNum *qpos0_{}, *qvel0_{};  // for align check
#endif
  int frame_skip_{};
  bool post_constraint_{};
  int max_episode_steps_{};
  int elapsed_step_ = 0;

  bool done_{true};

  // -------- headless rendering (zero cost until enabled) --------
  bool render_enabled_ = false;
  int render_w_ = 640, render_h_ = 360, fps_ = 24;
  mjvScene scn_;
  mjvCamera cam_;
  mjrContext con_;
  std::vector<unsigned char> rgb_;
  mjvOption opt_;
  mjvPerturb pert_;
  FILE* ffmpeg_pipe_ = nullptr;
  struct PendingSphere {
    std::array<mjtNum, 3> pos;
    mjtNum radius;
    std::array<float, 4> rgba;
  };
  std::vector<PendingSphere> pending_spheres_;

  void AppendPendingSpheres();
  void QueueSphere(const std::array<mjtNum, 3>& pos, mjtNum radius,
                   const std::array<float, 4>& rgba);

  void RenderInit();
  void RenderFrame();
  void RenderClose();

 public:
  MujocoEnv(const std::string& xml, int frame_skip, bool post_constraint,
            int max_episode_steps);
  ~MujocoEnv();
  void setIC();

  // -------- core --------
  void MujocoReset();
  virtual void MujocoResetModel();
  void MujocoStep(const mjtNum* action);

  // enable/disable rendering after construction
  void EnableRender(bool on = true);

  void addSpheres(
      const std::array<mjtNum, 3>& body_pos,
      const std::vector<std::array<mjtNum, 3>>& joint_positions,
      const std::vector<std::array<mjtNum, 3>>& foot_positions,
      mjtNum body_radius = 0.04, mjtNum joint_radius = 0.025,
      mjtNum foot_radius = 0.02);
};

// ========================================================================
//                            IMPLEMENTATION
// ========================================================================
inline MujocoEnv::MujocoEnv(const std::string& xml, int frame_skip,
                            bool post_constraint, int max_episode_steps)
    : model_(mj_loadXML(xml.c_str(), nullptr, error_.data(), error_.size())),
      data_(mj_makeData(model_)),
      init_qpos_(new mjtNum[model_->nq]),
      init_qvel_(new mjtNum[model_->nv]),
#ifdef ENVPOOL_TEST
      qpos0_(new mjtNum[model_->nq]),
      qvel0_(new mjtNum[model_->nv]),
#endif
      frame_skip_(frame_skip),
      post_constraint_(post_constraint),
      max_episode_steps_(max_episode_steps),
      elapsed_step_(max_episode_steps + 1) {
  if (!model_)
    throw std::runtime_error("MuJoCo loadXML failed: " +
                             std::string(error_.data()));

  std::memcpy(init_qpos_, data_->qpos, sizeof(mjtNum) * model_->nq);
  std::memcpy(init_qvel_, data_->qvel, sizeof(mjtNum) * model_->nv);
  setenv("MUJOCO_GL", "osmesa", 1);
  if (render_enabled_) RenderInit();
}

inline MujocoEnv::~MujocoEnv() {
  if (render_enabled_) RenderClose();
  mj_deleteData(data_);
  mj_deleteModel(model_);
  delete[] init_qpos_;
  delete[] init_qvel_;
#ifdef ENVPOOL_TEST
  delete[] qpos0_;
  delete[] qvel0_;
#endif
}
void MujocoEnv::setIC() {
    int kSideSign_[4] = {-1, 1, -1, 1};

    model_->opt.timestep = 0.002;
    const double minHeight = 0.24;  // minimum height
    const double maxHeight = 0.25;  // maximum height
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<double> distribution(minHeight, maxHeight);
    data_->qpos[2] = distribution(gen);  // Set height to a value sampled uniformly between minHeight and maxHeight

    for (int leg = 0; leg < 4; leg++) {
      data_->qpos[(leg) * 3 + 0 + 7] =
          10 * (M_PI / 180) *
          kSideSign_[leg];  // Add 7 to skip the first 7 dofs from body.
                            // (Position + Quaternion)
      data_->qpos[(leg) * 3 + 1 + 7] = -80 * (M_PI / 180);  //*kDirSign_[leg];
      data_->qpos[(leg) * 3 + 2 + 7] = 130 * (M_PI / 180);  
    }
    }
inline void MujocoEnv::MujocoReset() {
  elapsed_step_ = 0;
  done_ = false;
  mj_resetData(model_, data_);
  MujocoResetModel();
  // mj_forward(model_, data_);
}

inline void MujocoEnv::MujocoResetModel() {
  throw std::runtime_error("reset_model not implemented");
}

inline void MujocoEnv::MujocoStep(const mjtNum* action) {
  for (int i = 0; i < model_->nu; ++i) {
    data_->ctrl[i] = action[i];
  }
  mj_step(model_, data_);

  if (post_constraint_) mj_rnePostConstraint(model_, data_);

  if (render_enabled_) {
    // Render every N simulation steps to achieve target fps
    int render_every =
        std::max(1, static_cast<int>(1.0 / (fps_ * model_->opt.timestep)));
    if (elapsed_step_ % render_every == 0) {
      RenderFrame();
    }
  }
  ++elapsed_step_;
}

inline void MujocoEnv::QueueSphere(const std::array<mjtNum, 3>& pos,
                                   mjtNum radius,
                                   const std::array<float, 4>& rgba) {
  PendingSphere sphere;
  sphere.pos = pos;
  sphere.radius = std::max<mjtNum>(1e-4, radius);
  sphere.rgba = rgba;
  pending_spheres_.push_back(sphere);
}

inline void MujocoEnv::addSpheres(
    const std::array<mjtNum, 3>& body_pos,
    const std::vector<std::array<mjtNum, 3>>& joint_positions,
    const std::vector<std::array<mjtNum, 3>>& foot_positions,
    mjtNum body_radius, mjtNum joint_radius, mjtNum foot_radius) {
  pending_spheres_.clear();
  auto is_valid = [](const std::array<mjtNum, 3>& p) {
    return std::isfinite(p[0]) && std::isfinite(p[1]) &&
           std::isfinite(p[2]);
  };

  const std::array<float, 4> body_color{1.f, 0.2f, 0.2f, 0.85f};
  const std::array<float, 4> joint_color{1.f, 0.8f, 0.2f, 0.75f};
  const std::array<float, 4> foot_color{0.1f, 0.6f, .2f, 0.85f};

  if (is_valid(body_pos)) {
    QueueSphere(body_pos, body_radius, body_color);
  }

  for (const auto& joint : joint_positions) {
    if (is_valid(joint)) {
      QueueSphere(joint, joint_radius, joint_color);
    }
  }

  for (const auto& foot : foot_positions) {
    if (is_valid(foot)) {
      QueueSphere(foot, foot_radius, foot_color);
    }
  }
}

inline void MujocoEnv::AppendPendingSpheres() {
  if (pending_spheres_.empty()) return;

  for (const auto& sphere : pending_spheres_) {
    if (scn_.ngeom >= scn_.maxgeom) {
      break;
    }
    mjvGeom* geom = scn_.geoms + scn_.ngeom++;
    mjtNum size[3] = {sphere.radius, sphere.radius, sphere.radius};
    mjv_initGeom(geom, mjGEOM_SPHERE, size, sphere.pos.data(), nullptr,
                 sphere.rgba.data());
    geom->emission = 0.8f;
  }
  pending_spheres_.clear();
}

// --------------------- Rendering helpers ----------------------
inline void MujocoEnv::RenderInit() {
  ctx = OSMesaCreateContextExt(OSMESA_RGBA, /*depthBits=*/16, 0, 0, nullptr);
  if (!ctx) throw std::runtime_error("OSMesaCreateContextExt failed");
  if (!OSMesaMakeCurrent(ctx, fb, GL_UNSIGNED_BYTE, render_w_, render_h_))
    throw std::runtime_error("OSMesaMakeCurrent failed");

  mjv_defaultScene(&scn_);
  mjv_defaultCamera(&cam_);
  cam_.type = mjCAMERA_FREE;
  cam_.distance = 1.5;
  // Side view: place camera at (2, 0, 0) looking toward origin.
  cam_.azimuth = 90.0;
  cam_.elevation = -10.0;
  cam_.lookat[0] = data_->qpos[0];
  cam_.lookat[1] = data_->qpos[1];
  cam_.lookat[2] = data_->qpos[2];
  mjr_defaultContext(&con_);
  mjv_defaultOption(&opt_);
  mjv_defaultPerturb(&pert_);

  mjv_makeScene(model_, &scn_, 1000);
  mjr_makeContext(model_, &con_, mjFONTSCALE_150);

  rgb_.resize(static_cast<size_t>(render_w_ * render_h_ * 3));
  // Launch ffmpeg process to pipe frames into a video file
  // video file is named data/current/mujoco_env<env_id>_<timestamp>.mp4
  std::string cmd =
      "ffmpeg -loglevel warning -y -f rawvideo -vcodec rawvideo "
      "-pix_fmt rgb24 -s " +
      std::to_string(render_w_) + "x" + std::to_string(render_h_) + " -r " +
      std::to_string(fps_) +
      " -i - -an -vcodec libx264 -preset ultrafast -tune zerolatency "
      "-crf 18 -pix_fmt yuv420p "
      "data/current/mujoco_render" + ".mp4";
  ffmpeg_pipe_ = popen(cmd.c_str(), "w");
  if (!ffmpeg_pipe_) throw std::runtime_error("Cannot open ffmpeg pipe");
}

inline void MujocoEnv::RenderFrame() {
  opt_.flags[15] = 1;  // mjVIS_CONSTRAINT Visualize Contact Force
  // opt_.flags[10] = 1;  // mjVIS_CONSTRAINT Visualize Inertia
  cam_.azimuth += 0.5;  // slowly rotate camera
  cam_.lookat[0] = data_->qpos[0];
  cam_.lookat[1] = data_->qpos[1];
  cam_.lookat[2] = data_->qpos[2];
  if (!OSMesaMakeCurrent(ctx, fb, GL_UNSIGNED_BYTE, render_w_, render_h_))
    throw std::runtime_error("OSMesaMakeCurrent failed");

  mjrRect vp{0, 0, render_w_, render_h_};
  mjv_updateScene(model_, data_, &opt_, &pert_, &cam_, mjCAT_ALL, &scn_);
  AppendPendingSpheres();
  mjr_render(vp, &scn_, &con_);
  mjr_readPixels(rgb_.data(), nullptr, vp, &con_);

  // Flip vertically because OpenGL's origin is bottom‑left
  for (int y = render_h_ - 1; y >= 0; --y) {
    write_all(ffmpeg_pipe_,
              rgb_.data() + static_cast<size_t>(y) * render_w_ * 3,
              static_cast<size_t>(render_w_) * 3);
  }
  fflush(ffmpeg_pipe_);
}

inline void MujocoEnv::RenderClose() {
  std::cout << "MujocoEnv::RenderClose" << std::endl;
  if (ffmpeg_pipe_) pclose(ffmpeg_pipe_);
  mjr_freeContext(&con_);
  mjv_freeScene(&scn_);
  if (ctx) {
    OSMesaDestroyContext(ctx);
    ctx = nullptr;
  }
}

inline void MujocoEnv::EnableRender(bool on) {
  if (on && !render_enabled_) {
    render_enabled_ = true;
    RenderInit();
  } else if (!on && render_enabled_) {
    RenderClose();
    render_enabled_ = false;
  }
}

}  // namespace mujoco_gym

#endif  // ENVPOOL_MUJOCO_GYM_MUJOCO_ENV_H_
