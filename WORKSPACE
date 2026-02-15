workspace(name = "envpool")

# in /app/envpool/WORKSPACE (at the top, before any http_archive for mujoco)
new_local_repository(
    name = "mujoco",
    path = "/root/mujoco",                      # your clone
    build_file = "//third_party/mujoco:mujoco.BUILD",
)

new_local_repository(
    name = "quadcontrol",
    path = "../quadcontrol",
    build_file_content = """
package(default_visibility = ["//visibility:public"])

cc_library(
    name = "quadcontrol_rl",
    srcs = [
        "src/controllers/LegController.cpp",
        "src/controllers/FootSwingTrajectory.cpp",
        "src/controllers/wbc/ContactSet/SingleContact.cpp",
        "src/controllers/wbc/LocomotionCtrl/LocomotionCtrl.cpp",
        "src/controllers/wbc/TaskSet/BodyPosTask.cpp",
        "src/controllers/wbc/TaskSet/BodyOriTask.cpp",
        "src/controllers/wbc/TaskSet/LinkPosTask.cpp",
        "src/controllers/wbc/WBIC/KinWBC.cpp",
        "src/controllers/wbc/WBIC/WBIC.cpp",
        "src/controllers/wbc/WBC_Ctrl.cpp",
        "src/dynamics/QuadrupedSingleton.cc",
        "src/dynamics/Quadruped.cpp",
        "src/dynamics/FloatingBaseModel.cpp",
        "src/dynamics/QuadrupedKinematics.cc",
        "src/estimators/StateEstimators.cc",
        "src/core/RLPipelineRuntime.cc",
        "src/hardware/mujocohw/MdlSimDriver.cc",
        "src/hardware/mujocohw/SimClockHW.cc",
        "src/hardware/mujocohw/SimHW.cc",
        "src/hardware/mujocohw/SimIMUHW.cc",
        "src/hardware/mujocohw/SimMotorHW.cc",
        "src/modules/MdlControlParams.cc",
        "src/modules/MdlCtGaitScheduler.cc",
        "src/modules/MdlFootstepPlanner.cc",
        "src/modules/MdlLegController.cc",
        "src/modules/MdlRLCommandSource.cc",
        "src/modules/MdlRLLocomotionState.cc",
        "src/modules/MdlStateEstimator.cc",
        "src/modules/MdlWBIC.cc",
        "src/supervisor/SupervisorStates.cc",
        "src/utilities/Utilities_print.cpp",
        "src/utilities/utilities.cpp",
        "third-party/Goldfarb_Optimizer/Array.cc",
        "third-party/Goldfarb_Optimizer/QuadProg++.cc",
    ],
    hdrs = glob([
        "include/**/*.h",
        "include/**/*.hh",
        "include/**/*.hpp",
        "src/hardware/mujocohw/*.hh",
        "third-party/Goldfarb_Optimizer/*.h",
        "third-party/Goldfarb_Optimizer/*.hh",
        "third-party/JCQP/*.h",
        "third-party/JCQP/*.hh",
        "third-party/JCQP/amd/include/*.h",
    ]),
    includes = [
        "include",
        "src/hardware/mujocohw",
        "third-party",
    ],
    copts = [
        "-DMDL_SIM_OSMESA=1",
    ],
    linkopts = [
        "-lOSMesa",
        "-lGL",
        "-ldl",
        "-lm",
    ],
    deps = [
        "@eigen//:eigen",
        "@mujoco//:mujoco_lib",
        "@rtrobot//:rtcore",
        "@rtrobot//:rtclient",
    ],
)
""",
)

new_local_repository(
    name = "rtrobot",
    path = "../rtrobot",
    build_file_content = """
package(default_visibility = ["//visibility:public"])

cc_library(
    name = "rtcore",
    srcs = [
        "src/rtcore/ModuleManager.cc",
        "src/rtcore/Module.cc",
        "src/rtcore/ThreadUtil.cc",
        "src/rtcore/ThreadedLoop.cc",
        "src/rtcore/toml.c",
        "src/rtcore/ConfigTable.cc",
        "src/rtcore/ConfigInput.cc",
        "src/rtcore/LogJob.cc",
        "src/rtcore/LogServer.cc",
        "src/rtcore/Profiler.cc",
    ],
    hdrs = glob([
        "include/**/*.h",
        "include/**/*.hh",
    ]),
    includes = [
        "include",
    ],
)

cc_library(
    name = "rtclient",
    srcs = [
        "src/rtclient/LogClient.cc",
        "src/rtclient/WriteASCII.cc",
        "src/rtclient/WriteRaw.cc",
        "src/rtclient/WriteML.cc",
        "src/rtclient/WriteCSV.cc",
    ],
    hdrs = glob([
        "include/**/*.h",
        "include/**/*.hh",
    ]),
    includes = [
        "include",
    ],
    deps = [
        ":rtcore",
    ],
    linkopts = [
        "-lpthread",
    ],
)
""",
)

load("//envpool:workspace0.bzl", workspace0 = "workspace")

workspace0()

load("//envpool:workspace1.bzl", workspace1 = "workspace")

workspace1()

# QT special, cannot move to workspace2.bzl, not sure why

load("@local_config_qt//:local_qt.bzl", "local_qt_path")

new_local_repository(
    name = "qt",
    build_file = "@com_justbuchanan_rules_qt//:qt.BUILD",
    path = local_qt_path(),
)

new_local_repository(
    name = "eigen",
    path = "/usr/include/eigen3",
    build_file_content = """
package(default_visibility = ["//visibility:public"])

cc_library(
    name = "eigen",
    hdrs = glob([
        "Eigen/**",
        "unsupported/**",
    ]),
    includes = [
        ".",
    ],
)
""",
)


load("@com_justbuchanan_rules_qt//tools:qt_toolchain.bzl", "register_qt_toolchains")

register_qt_toolchains()

load("//envpool:pip.bzl", pip_workspace = "workspace")

pip_workspace()
