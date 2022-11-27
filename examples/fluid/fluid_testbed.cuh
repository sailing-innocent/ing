#pragma once
/**
 * @file: examples/fluid_testbed.h
 * @author: sailing-innocent
 * @create: 2022-11-26
 * @desp: The fluid testbed
*/

#include <testbed/cuda/cuda_gl.cuh>

class FluidTestbed: public ing::TestbedCudaGL
{
public:
    FluidTestbed() = default;
    ~FluidTestbed() {}
protected:
    void gen_vertices();
};

