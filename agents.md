this file belongs to envpool project. under this we are creating a custom gym that depend on quadcontrol project.

the code builds and runs only inside the "envpool-dev" docker container. 

the command "make run" is used to build, install and run the python package. 
we work on the ./envpool/mujoco/gym/humanoid.h file and rewrite it as a custom gym environment dependent on quadcontrol module 
