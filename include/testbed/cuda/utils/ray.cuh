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
    ING_CU_HOST_DEVICE Ray(point& _origin, point& _dir): m_origin(_origin), m_dir(_dir) {}
    ING_CU_HOST_DEVICE bool hit() {
        float t = - m_dir.z() / m_dir.z();
        point s = m_origin + t * m_dir;
        return sqrtf(s.x() * s.x() + s.y() * s.y()) < 0.5f;
    }
protected:
    point m_origin;
    point m_dir;
};

ING_NAMESPACE_END

#endif // TESTBED_CUDA_RAY_H_