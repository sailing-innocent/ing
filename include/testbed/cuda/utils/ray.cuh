#pragma once

#ifndef TESTBED_CUDA_RAY_H_
#define TESTBED_CUDA_RAY_H_

/**
 * @file: include/testbed/cuda/utils/ray.cuh
 * @author: sailing-innocent
 * @create: 2022-12-02
 * @desp: the ray header for testbed cuda
*/

#include <testbed/cuda/common_device.cuh>
#include <testbed/cuda/utils/vec4.cuh>

ING_NAMESPACE_BEGIN

class Ray {
public:
    ING_CU_HOST_DEVICE Ray(): m_origin(), m_dir(0.0f,0.0f,1.0f,0.0f) {};
    ING_CU_HOST_DEVICE Ray(point& _origin, vec4& _dir): m_origin(_origin), m_dir(_dir) {}
    ING_CU_HOST_DEVICE point origin() const { return m_origin; }
    ING_CU_HOST_DEVICE vec4 dir() const { return m_dir; }
    ING_CU_HOST_DEVICE point at(float t) const { return m_origin + t * m_dir; }
protected:
    point m_origin;
    vec4 m_dir;
};

ING_NAMESPACE_END

#endif // TESTBED_CUDA_RAY_H_