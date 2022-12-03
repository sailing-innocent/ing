#pragma once
#ifndef TESTBED_CUDA_UTILS_CAMERA_H_
#define TESTBED_CUDA_UTILS_CAMERA_H_

#include <testbed/cuda/common_device.cuh>

ING_NAMESPACE_BEGIN

struct Camera {
    unsigned int width;
    unsigned int height;
    float whr; // width/height ratio
    float fov; // field of view
};

ING_NAMESPACE_END

#endif // TESTEBD_CUDA_UTILS_CAMERA_H