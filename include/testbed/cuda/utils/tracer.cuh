#pragma once
#ifndef TESTBED_CUDA_UTILS_TRACER_H_
#define TESTBED_CUDA_UTILS_TRACER_H_

#include <testbed/cuda/common_device.cuh>
#include "testbed/cuda/utils/vec4.cuh"
#include "testbed/cuda/utils/world.cuh"
#include "testbed/cuda/utils/camera.cuh"
#include "testbed/cuda/utils/hittable.cuh"

ING_NAMESPACE_BEGIN

ING_CU_HOST_DEVICE color ray_color(const Ray& r, Sphere& s);

__global__ void tracer_kernel(float* positions, float time, unsigned int width, unsigned int height, World world);

ING_NAMESPACE_END

#endif // TESTBED_CUDA_BASE_H_