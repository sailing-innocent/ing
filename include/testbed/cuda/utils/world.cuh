#pragma once
#ifndef TESTBED_CUDA_UTILS_WORLD_H_
#define TESTBED_CUDA_UTILS_WORLD_H_

/**
 * @file: include/testbed/cuda/utils/world.cuh
 * @author: sailing-innocent
 * @create: 2022-12-03
 * @desp: the world scene header
*/

#include <testbed/cuda/utils/hittable.cuh>
#include <vector>

ING_NAMESPACE_BEGIN


class World {
public:
    ING_CU_HOST_DEVICE bool add_sphere(Sphere s) {
        m_spheres.push_back(s);
    }
public:
    float val = 0.25f;
protected:
    std::vector<Sphere> m_spheres;
};

ING_NAMESPACE_END

#endif // TESTBED_CUDA_UTILS_WORLD_H_