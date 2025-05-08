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
#include <iostream>
#include <array>
#include <cstdio>   // FILE*
#include <cstring>  // std::memcpy
#include <stdexcept>
#include <string>
#include <vector>

static OSMesaContext ctx;
static unsigned char fb[1920 * 1080 * 4];  // off‑screen framebuffer

namespace mujoco_gym {

static void write_all(FILE* pipe, const unsigned char* buf, size_t nbytes) {
  while (nbytes) {
    size_t n = fwrite(buf, 1, nbytes, pipe);
    if (n == 0)                     // pipe closed or broken
      throw std::runtime_error("fwrite to ffmpeg pipe failed");
    buf    += n;
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
  int render_w_ = 640, render_h_ = 360, fps_ = 10;
  mjvScene scn_;
  mjvCamera cam_;
  mjrContext con_;
  std::vector<unsigned char> rgb_;
  mjvOption opt_;
  mjvPerturb pert_;
  FILE* ffmpeg_pipe_ = nullptr;

  void RenderInit();
  void RenderFrame();
  void RenderClose();

 public:
  MujocoEnv(const std::string& xml, int frame_skip, bool post_constraint,
            int max_episode_steps);
  ~MujocoEnv();

  // -------- core --------
  void MujocoReset();
  virtual void MujocoResetModel();
  void MujocoStep(const mjtNum* action);

  // enable/disable rendering after construction
  void EnableRender(bool on = true);
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

inline void MujocoEnv::MujocoReset() {
  mj_resetData(model_, data_);
  MujocoResetModel();
  mj_forward(model_, data_);
  
  elapsed_step_ = 0;
  done_ = false;
}

inline void MujocoEnv::MujocoResetModel() {
  throw std::runtime_error("reset_model not implemented");
}

inline void MujocoEnv::MujocoStep(const mjtNum* action) {
  for (int i = 0; i < model_->nu; ++i) {
    data_->ctrl[i] = action[i];
  }
  // for (int i = 0; i < frame1_skip_; ++i) mj_step(model_, data_);
  mj_step(model_, data_);

  if (post_constraint_) mj_rnePostConstraint(model_, data_);
  
  if (render_enabled_) {
    int render_every = std::max(1, static_cast<int>(1.0 / (fps_ * frame_skip_ * model_->opt.timestep)));
    // std::cout << "Render every: " << render_every << std::endl;
    // std::cout << "Elapsed step: " << elapsed_step_ << std::endl;
    if (  elapsed_step_ % render_every == 0) {
      RenderFrame();


      // std::cout<<"Render frame: " << elapsed_step_ / render_every << std::endl;
    }
  }
  ++elapsed_step_;
  
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
  cam_.distance = std::max(1.0, 3 * model_->stat.extent);
  mjr_defaultContext(&con_);
  mjv_defaultOption(&opt_);
  mjv_defaultPerturb(&pert_);

  mjv_makeScene(model_, &scn_, 1000);
  mjr_makeContext(model_, &con_, mjFONTSCALE_150);

  rgb_.resize(static_cast<size_t>(render_w_ * render_h_ * 3));

  std::string cmd =
      "ffmpeg -loglevel warning -y -f rawvideo -vcodec rawvideo "
      "-pix_fmt rgb24 -s " +
      std::to_string(render_w_) + "x" + std::to_string(render_h_) + " -r " +
      std::to_string(fps_) +
      " -i - -an -vcodec libx264 -preset ultrafast -tune zerolatency "
      "-crf 18 -pix_fmt yuv420p "
      "/app/mujoco_record_init.mp4";
  ffmpeg_pipe_ = popen(cmd.c_str(), "w");
  if (!ffmpeg_pipe_) throw std::runtime_error("Cannot open ffmpeg pipe");
}

inline void MujocoEnv::RenderFrame() {
  // std::cout << "MujocoEnv::RenderFrame" << std::endl;
  if (!OSMesaMakeCurrent(ctx, fb, GL_UNSIGNED_BYTE, render_w_, render_h_))
    throw std::runtime_error("OSMesaMakeCurrent failed");

  mjrRect vp{0, 0, render_w_, render_h_};
  mjv_updateScene(model_, data_, &opt_, &pert_, &cam_, mjCAT_ALL, &scn_);
  mjr_render(vp, &scn_, &con_);
  mjr_readPixels(rgb_.data(), nullptr, vp, &con_);

  // Flip vertically because OpenGL's origin is bottom‑left
  for (int y = render_h_ - 1; y >= 0; --y) {
    write_all(ffmpeg_pipe_,
              rgb_.data() + static_cast<size_t>(y) * render_w_ * 3,
              static_cast<size_t>(render_w_) * 3);
  }
}

inline void MujocoEnv::RenderClose() {
  std::cout << "MujocoEnv::RenderClose" << std::endl;
  if (ffmpeg_pipe_) pclose(ffmpeg_pipe_);
  mjr_freeContext(&con_);
  mjv_freeScene(&scn_);
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
