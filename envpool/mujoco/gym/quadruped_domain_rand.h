// Copyright 2024 Garena Online Private Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef ENVPOOL_MUJOCO_GYM_QUADRUPED_DOMAIN_RAND_H_
#define ENVPOOL_MUJOCO_GYM_QUADRUPED_DOMAIN_RAND_H_

#include <array>
#include <random>

namespace mujoco_gym {

// Ground-friction randomisation shared by both learned arms, so the two train
// against one disturbance distribution rather than two hand-copied ones.
//
// legged_gym's `_process_rigid_shape_props` draws one coefficient per env from
// 64 buckets and holds it for that env's lifetime. PhysX then *averages* the
// robot shape's coefficient with the ground plane's to get the contact pair
// value; MuJoCo instead takes the elementwise max, so the averaged value is
// written to every geom to reproduce the same effective coefficient.
//
// The ground coefficient is passed in, read from the MJCF by the caller. It is
// never hard-coded here: the model is the single source of truth for the plant.
struct FrictionRandomisation {
  static constexpr double kRange[2] = {0.5, 1.25};
  static constexpr int kBuckets = 64;

  // `bucket_seed` is the pool seed, so every env sees the same 64 buckets;
  // `gen` is the per-env stream that picks which bucket this env gets.
  static double Sample(double ground, unsigned bucket_seed, std::mt19937* gen) {
    std::mt19937 bucket_gen(bucket_seed);
    std::uniform_real_distribution<double> bucket_dist(kRange[0], kRange[1]);
    std::array<double, kBuckets> buckets{};
    for (auto& b : buckets) b = bucket_dist(bucket_gen);
    const int bucket =
        std::uniform_int_distribution<int>(0, kBuckets - 1)(*gen);
    return (buckets[bucket] + ground) / 2.0;
  }
};

}  // namespace mujoco_gym

#endif  // ENVPOOL_MUJOCO_GYM_QUADRUPED_DOMAIN_RAND_H_
