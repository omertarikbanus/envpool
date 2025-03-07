# Copyright 2022 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@pip_requirements//:requirements.bzl", "requirement")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

package(default_visibility = ["//visibility:public"])

genrule(
    name = "gen_mujoco_gym_xml",
    srcs = ["@mujoco_gym_xml"],
    outs = ["assets_gym"],
    cmd = "mkdir -p $(OUTS) && cp -r $(SRCS) $(OUTS)",
)

genrule(
    name = "gen_mujoco_dmc_xml",
    srcs = ["@mujoco_dmc_xml"],
    outs = ["assets_dmc"],
    cmd = "mkdir -p $(OUTS) && cp -r $(SRCS) $(OUTS)",
)

genrule(
    name = "gen_mujoco_so",
    srcs = ["@mujoco//:mujoco_so"],
    outs = ["libmujoco.so.2.2.1"],
    cmd = "cp $< $@",
)
# cc_import(
#   name = "qpOASES",
#   hdrs = glob(["legged-sim/third-party/qpOASES/include/qpOASES.hpp"]),
#   shared_library = "legged-sim/build/third-party/qpOASES/libs/libqpOASES.so",
# )
# cc_import(
#   name = "ParamHandler",
#   hdrs = glob([
#     "envpool/mujoco/legged-sim/third-party/ParamHandler/**/*.hpp",
#     "envpool/mujoco/legged-sim/third-party/ParamHandler/**/*.h",
#   ]),
#   shared_library = "legged-sim/build/third-party/ParamHandler/libdynacore_param_handler.so",
# )

# cc_import(
#   name = "JCQP",
#   hdrs = glob([
#     "envpool/mujoco/legged-sim/third-party/JCQP/**/*.hpp",
#     "envpool/mujoco/legged-sim/third-party/JCQP/**/*.h",
#   ]),
#   shared_library = "legged-sim/build/third-party/JCQP/libJCQP.so",
# )

# # cc_import(
# #   name = "osqp",
# #   hdrs = glob([
# #     "legged-sim/third-party/osqp/**/*.hpp",
# #     "legged-sim/third-party/osqp/include/osqp.h",
# #   ]),
# #   includes = [
# #         "legged-sim/third-party/osqp/include",
# #         "legged-sim/third-party/osqp/tests",
# #         "legged-sim/third-party/osqp/lin_sys",
# #         "legged-sim/third-party/osqp/lin_sys/direct/qdldl/qdldl_sources/include",
# #         ],
# #   shared_library = "legged-sim/build/third-party/osqp/out/libosqp.so",
# # )
# cc_import(
#   name = "osqp",
#   hdrs = glob([
#     "legged-sim/third-party/osqp/include/osqp.h",  # Specific header file
#     "legged-sim/third-party/osqp/**/*.h",
#   ]),
#   includes = [
#     "legged-sim/third-party/osqp/tests",
#     "legged-sim/third-party/osqp/lin_sys",
#     "legged-sim/third-party/osqp/lin_sys/direct/qdldl",
#     "legged-sim/third-party/osqp/lin_sys/direct/pardiso",
#     "legged-sim/third-party/osqp/lin_sys/direct/qdldl/qdldl_sources/include",
#     "legged-sim/third-party/osqp/include",
#   ],
#   shared_library = "legged-sim/build/third-party/osqp/out/libosqp.so",
# )
# cc_import(
#   name = "JCQP",
#   hdrs = glob([
#         "legged-sim/third-party/JCQP",
#         "legged-sim/third-party/JCQP/amd/include",
#   ]),
#   includes = [
#         "legged-sim/third-party/JCQP",
#         "legged-sim/third-party/JCQP/amd/include",
#   ],
#   shared_library = "legged-sim/build/third-party/JCQP/libJCQP.so",
# )
# cc_import(
#   name = "qpOASES",
#   hdrs = glob([
#         "legged-sim/third-party/qpOASES/include",
#   ]),
#   includes = [
#         "legged-sim/third-party/qpOASES/include",
#   ],
#   shared_library = "envpool/mujoco/legged-sim/build/third-party/qpOASES/libs/libqpOASES.so",
# )
cc_library(
    name = "mujoco_gym_env",
    hdrs = glob([
        "mbc_interface.h",
        "legged-sim/**/*.h",
        "legged-sim/**/*.hh",
        "legged-sim/**/*.hpp",
        "legged-sim/**/*.cpp",
        "legged-sim/**/*.ipp",
        "gym/ant.h",
        "gym/half_cheetah.h",
        "gym/hopper.h",
        "gym/humanoid.h",
        "gym/humanoid_standup.h",
        "gym/inverted_double_pendulum.h",
        "gym/inverted_pendulum.h",
        "gym/mujoco_env.h",
        "gym/pusher.h",
        "gym/reacher.h",
        "gym/swimmer.h",
        "gym/walker2d.h",
        ]),
    srcs = glob([  
    "mbc_interface.h",
    "legged-sim/src/**/*.cpp",
    "legged-sim/src/FSM/*.cpp",
    "legged-sim/src/*.cpp",
    "legged-sim/src/Utilities/*.cpp",
    "legged-sim/third-party/ParamHandler/**/*.cpp",
    "legged-sim/third-party/qpOASES/**/*.hpp",
    "legged-sim/third-party/qpOASES/**/*.ipp",
    "legged-sim/third-party/Goldfarb_Optimizer/**/*.cpp",
    "legged-sim/third-party/Goldfarb_Optimizer/**/*.cc",
    "legged-sim/third-party/ParamHandler/**/*.cpp",
    "legged-sim/third-party/qpOASES/**/*.cpp",
    "legged-sim/third-party/Goldfarb_Optimizer/**/*.cpp",
    "legged-sim/third-party/Goldfarb_Optimizer/**/*.cc",
    "legged-sim/user/Embedded_Controller/*.cpp",
    "legged-sim/third-party/ParamHandler/*.cpp",
    "legged-sim/third-party/qpOASES/*.cpp",
    "legged-sim/third-party/Goldfarb_Optimizer/*.cpp",
    "legged-sim/third-party/Goldfarb_Optimizer/*.cc",
    # "legged-sim/third-party/osqp/**/*.cpp",
    # "legged-sim/third-party/osqp/**/*.c",
    # "legged-sim/third-party/osqp/**/*.h",
    "legged-sim/third-party/JCQP/**/*.cpp",
    "legged-sim/third-party/JCQP/**/*.c",
            
        ]),
    includes = [
        ".",
        "legged-sim",
        "legged-sim/include",
        "legged-sim/include/Controllers",
        "legged-sim/include/convexMPC",
        "legged-sim/third-party",
        # "legged-sim/third-party/osqp",
        # "legged-sim/third-party/osqp/include",
        # "legged-sim/third-party/osqp/lin_sys",
        # "legged-sim/third-party/osqp/lin_sys/direct",
        # "legged-sim/third-party/osqp/lin_sys/direct/qdldl",
        # "legged-sim/third-party/osqp/tests",
        # "legged-sim/third-party/osqp/lin_sys/direct/qdldl/amd/include",
        # "legged-sim/third-party/osqp/lin_sys/direct/qdldl/qdldl_sources/include",
        # "legged-sim/third-party/osqp/lin_sys/direct/pardiso",
        "legged-sim/third-party/qpOASES/include/qpOASES",
        "legged-sim/third-party/qpOASES/include",
        "legged-sim/third-party/qpOASES",
        "legged-sim/third-party/GoldFarb_Optimizer",
        "legged-sim/third-party/ParamHandler",
        "legged-sim/third-party/JCQP",
        "legged-sim/third-party/JCQP/amd/include",
        "legged-sim/lcm_types/cpp",
        "legged-sim/third-party/lcm/",
        
    ],
    data = [
        ":gen_mujoco_gym_xml",
    ],
    deps = [
        "//envpool/core:async_envpool",
        "@mujoco//:mujoco_lib",

    ],
    linkopts = ["-ltinyxml2 -lpthread  -usr/local/lib/libosqp.so -usr/local/lib/libqdldl.so -usr/local/lib/libqpOASES.so "], 
    copts = [" --debug -g -O0 -ggdb -w  -D__NO_COPYRIGHT__ -Wno-error  -Wall -pedantic -Wfloat-equal -Wshadow -DLINUX -finline-functions"],
)
# -no-pie 
pybind_extension(
    name = "mujoco_gym_envpool",
    srcs = [
        "gym/mujoco_envpool.cc",
    ],
    deps = [
        ":mujoco_gym_env",
        "//envpool/core:py_envpool",
    ],
)

cc_library(
    name = "mujoco_dmc_env",
    srcs = [
        "dmc/mujoco_env.cc",
        "dmc/utils.cc",
    ],
    hdrs = [
        "dmc/acrobot.h",
        "dmc/ball_in_cup.h",
        "dmc/cartpole.h",
        "dmc/cheetah.h",
        "dmc/finger.h",
        "dmc/fish.h",
        "dmc/hopper.h",
        "dmc/humanoid.h",
        "dmc/humanoid_CMU.h",
        "dmc/manipulator.h",
        "dmc/mujoco_env.h",
        "dmc/pendulum.h",
        "dmc/point_mass.h",
        "dmc/reacher.h",
        "dmc/swimmer.h",
        "dmc/utils.h",
        "dmc/walker.h",
    ],
    data = [":gen_mujoco_dmc_xml"],
    deps = [
        "//envpool/core:async_envpool",
        "@mujoco//:mujoco_lib",
        "@pugixml",
    ],
)

pybind_extension(
    name = "mujoco_dmc_envpool",
    srcs = [
        "dmc/mujoco_envpool.cc",
    ],
    deps = [
        ":mujoco_dmc_env",
        "//envpool/core:py_envpool",
    ],
)

py_library(
    name = "mujoco_dmc",
    srcs = ["dmc/__init__.py"],
    data = [
        ":gen_mujoco_dmc_xml",
        ":gen_mujoco_so",
        ":mujoco_dmc_envpool.so",
    ],
    deps = ["//envpool/python:api"],
)

py_library(
    name = "mujoco_gym",
    srcs = ["gym/__init__.py"],
    data = [
        ":gen_mujoco_gym_xml",
        ":gen_mujoco_so",
        ":mujoco_gym_envpool.so",
    ],
    deps = ["//envpool/python:api"],
)

cc_test(
    name = "mujoco_envpool_test",
    size = "enormous",
    srcs = ["gym/mujoco_gym_envpool_test.cc"],
    deps = [
        ":mujoco_gym_env",
        "@com_google_googletest//:gtest_main",
    ],
)

py_library(
    name = "mujoco_dmc_registration",
    srcs = ["dmc/registration.py"],
    deps = ["//envpool:registration"],
)

py_library(
    name = "mujoco_gym_registration",
    srcs = ["gym/registration.py"],
    deps = ["//envpool:registration"],
)

py_test(
    name = "mujoco_gym_deterministic_test",
    size = "enormous",
    srcs = ["gym/mujoco_gym_deterministic_test.py"],
    deps = [
        ":mujoco_gym",
        ":mujoco_gym_registration",
        requirement("numpy"),
        requirement("absl-py"),
        requirement("gym"),
    ],
)

py_test(
    name = "mujoco_gym_align_test",
    size = "enormous",
    srcs = ["gym/mujoco_gym_align_test.py"],
    deps = [
        ":mujoco_gym",
        ":mujoco_gym_registration",
        requirement("numpy"),
        requirement("absl-py"),
        requirement("gym"),
        requirement("mujoco"),
    ],
)

py_test(
    name = "mujoco_dmc_suite_deterministic_test",
    size = "enormous",
    srcs = ["dmc/mujoco_dmc_suite_deterministic_test.py"],
    deps = [
        ":mujoco_dmc",
        ":mujoco_dmc_registration",
        requirement("numpy"),
        requirement("absl-py"),
    ],
)

py_test(
    name = "mujoco_dmc_suite_align_test",
    size = "enormous",
    srcs = ["dmc/mujoco_dmc_suite_align_test.py"],
    deps = [
        ":mujoco_dmc",
        ":mujoco_dmc_registration",
        requirement("numpy"),
        requirement("absl-py"),
        requirement("mujoco"),
        requirement("dm-env"),
        requirement("dm-control"),
    ],
)

py_test(
    name = "mujoco_dmc_suite_ext_deterministic_test",
    size = "enormous",
    srcs = ["dmc/mujoco_dmc_suite_ext_deterministic_test.py"],
    deps = [
        ":mujoco_dmc",
        ":mujoco_dmc_registration",
        requirement("numpy"),
        requirement("absl-py"),
    ],
)

py_test(
    name = "mujoco_dmc_suite_ext_align_test",
    size = "enormous",
    srcs = ["dmc/mujoco_dmc_suite_ext_align_test.py"],
    deps = [
        ":mujoco_dmc",
        ":mujoco_dmc_registration",
        requirement("numpy"),
        requirement("absl-py"),
        requirement("mujoco"),
        requirement("dm-env"),
        requirement("dm-control"),
    ],
)
