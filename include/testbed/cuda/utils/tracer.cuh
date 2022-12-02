#pragma once
#ifndef TESTBED_CUDA_UTILS_TRACER_H_
#define TESTBED_CUDA_UTILS_TRACER_H_

#include <testbed/cuda/common_device.cuh>
#include "testbed/cuda/utils/vec4.cuh"

ING_NAMESPACE_BEGIN

struct Camera {
    unsigned int width;
    unsigned int height;
    float whr; // width/height ratio
    float fov; // field of view
};

class Circle {
public:
    ING_CU_HOST_DEVICE float originx() {return m_originx;}
    ING_CU_HOST_DEVICE float originy() {return m_originy;}
    ING_CU_HOST_DEVICE float radius() {return m_radius;}
private:
    float m_originx = 0.0f;
    float m_originy = 0.0f;
    float m_radius = 0.2f;
};

class World {
public:
    float val = 0.25f;
    Circle circle;
};

__global__ void tracer_kernel(float* positions, float time, unsigned int width, unsigned int height, World world);

ING_NAMESPACE_END

#endif // TESTBED_CUDA_BASE_H_