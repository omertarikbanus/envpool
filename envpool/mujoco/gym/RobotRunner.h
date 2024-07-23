#include <iostream>
#include "envpool/core/env.h"
#include "envpool/mujoco/gym/mujoco_env.h"

class myActuator {
public:
    myActuator(){};
    void setGain(double newGain) {
        gain = newGain;
    }

    void run(const mjtNum* act, mjtNum* result, int size)  {
        for (int i = 0; i < size; ++i) {
            result[i] = act[i] * 2;
        }
    }

private:
    double gain=1.0;
};
