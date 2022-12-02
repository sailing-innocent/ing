#pragma once 
#ifndef TESTBED_CUDA_UTILS_VEC4
#define TESTBED_CUDA_UTILS_VEC4

/**
 * @file: include/testbed/cuda/utils/vec4.cuh
 * @author: sailing-innocent
 * @create: 2022-12-02
 * @desp: the testbed cuda vec4 header
*/
#include <testbed/cuda/common_device.cuh>
#include <vector>

ING_NAMESPACE_BEGIN

class vec4 {
public:
    ING_CU_HOST_DEVICE vec4(float _x = 0.0f, float _y=0.0f, float _z=0.0f, float _w=1.0f) {
        m_data[0] = _x;
        m_data[1] = _y;
        m_data[2] = _z;
        m_data[3] = _w;
    }
    ING_CU_HOST_DEVICE vec4(float* _data){
        for (auto i = 0; i < 4; i++) {
            m_data[i] = _data[i];
        }
    }
    ING_CU_HOST_DEVICE vec4(const vec4& rhs) {
        for (auto i = 0; i < 4; i++) {
            m_data[i] = rhs[i];
        }
    }
    ING_CU_HOST_DEVICE float x() { return m_data[0]; }
    ING_CU_HOST_DEVICE float y() { return m_data[1]; }
    ING_CU_HOST_DEVICE float z() { return m_data[2]; }
    ING_CU_HOST_DEVICE float w() { return m_data[3]; }
    ING_CU_HOST_DEVICE float& operator[](const size_t index) { return m_data[index]; }
    ING_CU_HOST_DEVICE float operator[](const size_t index) const { return m_data[index]; }
protected:
    float m_data[4] = { 0.0f, 0.0f, 0.0f, 0.0f};
};

ING_CU_HOST_DEVICE inline vec4 operator+(const vec4 &u, const vec4& v) {
    vec4 res;
    for (auto i = 0; i < 4; i++) {
        res[i] = u[i] + v[i];
    }
    return res;
}

ING_CU_HOST_DEVICE inline vec4 operator*(const float k, const vec4& u) {
    vec4 res;
    for (auto i = 0; i < 4; i++) {
        res[i] = k*u[i];
    }
    return res;
}

ING_CU_HOST_DEVICE inline vec4 operator-(const vec4& u, const vec4& v) {
    vec4 res;
    for (auto i = 0; i < 4; i++) {
        res[i] = u[i] - v[i];
    }
    return res;
}

using point = vec4;
using color = vec4;

ING_NAMESPACE_END
#endif // TESTBED_CUDA_UTILS_VEC4