workspace(name = "envpool")

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

local_repository(
    name = "eigen",
    path = "/usr/local/include/eigen3",
)


load("@com_justbuchanan_rules_qt//tools:qt_toolchain.bzl", "register_qt_toolchains")

register_qt_toolchains()

load("//envpool:pip.bzl", pip_workspace = "workspace")

pip_workspace()
