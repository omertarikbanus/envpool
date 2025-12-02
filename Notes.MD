
# Notes 

## TODO

- Clean up codes.
- remove unused log files.


## Development Notes
- number of environments is crucial for succesfull training. I set them to 32 vs 128 and the training was stable only for 32 environments case. 

- frameskip variable also effects the training's success. Under same conditions with 32 envs, when frameskip was set to 15 the policy's learning was much worse. 

- 
## Basic Usage

Basic Install

    export USE_BAZEL_VERSION=6.4.0
    sudo apt-get install qtbase5-dev #run bazel clean --expunge, then try again

If you get this error

    Traceback (most recent call last):
      File "/home/tarik/.cache/bazel/_bazel_tarik/bdd02770c320fc1c335f1fbf6437f65f/execroot/envpool/bazel-out/k8-fastbuild/bin/setup.runfiles/envpool/setup.py", line 3, in <module>
        from setuptools import setup
      File "/home/tarik/.cache/bazel/_bazel_tarik/bdd02770c320fc1c335f1fbf6437f65f/execroot/envpool/bazel-out/k8-fastbuild/bin/setup.runfiles/pip_requirements/pypi__setuptools/site-packages/setuptools/__init__.py", line 21, in <module>
        from .dist import Distribution
      File "/home/tarik/.cache/bazel/_bazel_tarik/bdd02770c320fc1c335f1fbf6437f65f/execroot/envpool/bazel-out/k8-fastbuild/bin/setup.runfiles/pip_requirements/pypi__setuptools/site-packages/setuptools/dist.py", line 29, in <module>
        from . import _entry_points
      File "/home/tarik/.cache/bazel/_bazel_tarik/bdd02770c320fc1c335f1fbf6437f65f/execroot/envpool/bazel-out/k8-fastbuild/bin/setup.runfiles/pip_requirements/pypi__setuptools/site-packages/setuptools/_entry_points.py", line 6, in <module>
        from jaraco.text import yield_lines
      File "/home/tarik/.cache/bazel/_bazel_tarik/bdd02770c320fc1c335f1fbf6437f65f/execroot/envpool/bazel-out/k8-fastbuild/bin/setup.runfiles/pip_requirements/pypi__setuptools/site-packages/setuptools/_vendor/jaraco/text/__init__.py", line 231, in <module>
        files(__name__).joinpath('Lorem ipsum.txt').read_text(encoding='utf-8')
      File "/usr/lib/python3.10/pathlib.py", line 1134, in read_text
        with self.open(mode='r', encoding=encoding, errors=errors) as f:
      File "/usr/lib/python3.10/pathlib.py", line 1119, in open
        return self._accessor.open(self, mode, buffering, encoding, errors,
    FileNotFoundError: [Errno 2] No such file or directory: '/home/tarik/.cache/bazel/_bazel_tarik/bdd02770c320fc1c335f1fbf6437f65f/execroot/envpool/bazel-out/k8-fastbuild/bin/setup.runfiles/pip_requirements/pypi__setuptools/site-packages/setuptools/_vendor/jaraco/text/Lorem ipsum.txt'
    tarik@aselsan:/home/tarik/thesis_project/envpool$ ^C

Comment out these lines 

    # lorem_ipsum: str = (
    #     files(__name__).joinpath('Lorem ipsum.txt').read_text(encoding='utf-8')
    # )

To run in debug mode

    bazel run --config=debug //:setup -- bdist_wheel  > log.txt  

    pip3 install bazel-bin/setup.runfiles/envpool/dist/envpool-0.8.4-cp310-cp310-linux_x86_64.whl --force-reinstall

    cc_library(
      name = "osqp",
      hdrs = glob([
        "legged-sim/third-party/osqp/**/*.h",  # Header file
      ]),
      srcs = glob([
        "legged-sim/third-party/osqp/src/**/*.c",  # Adjust according to source files
        "legged-sim/third-party/osqp/src/**/*.cpp",
      ]),
      includes = [
        #         "legged-sim/third-party/osqp/include",
        "legged-sim/third-party/osqp/tests",
        "legged-sim/third-party/osqp/lin_sys",
        "legged-sim/third-party/osqp/lin_sys/direct/qdldl",
        "legged-sim/third-party/osqp/lin_sys/direct/pardiso",
        "legged-sim/third-party/osqp/lin_sys/direct/qdldl/qdldl_sources/include",
        "legged-sim/third-party/osqp/include",
      ],
    #   copts = [
    #     "-Ilegged-sim/third-party/osqp/include",  # Add this to ensure osqp.h is found
    #   ],
    #   linkopts = ["legged-sim/build/third-party/osqp/out/libosqp.so"],
    )


MAC FIXES 

    symlink qt. install it if not installed.
    ln -s /usr/include/aarch64-linux-gnu/qt5 /usr/include/qt
    ln -s /usr/include/eigen3/Eigen /usr/include/Eigen
    '''