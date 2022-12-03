#pragma once
#ifndef TESTBED_CUDA_UTILS_WORLD_H_
#define TESTBED_CUDA_UTILS_WORLD_H_

/**
 * @file: include/testbed/cuda/utils/world.cuh
 * @author: sailing-innocent
 * @create: 2022-12-03
 * @desp: the world scene header
*/

#include <testbed/cuda/common_device.cuh>
#include <testbed/cuda/utils/vec4.cuh>
#include <testbed/cuda/utils/ray.cuh>
#include <vector>

ING_NAMESPACE_BEGIN


class World {
public:
    float val = 0.25f;
};

ING_NAMESPACE_END

#endif // TESTBED_CUDA_UTILS_WORLD_H_