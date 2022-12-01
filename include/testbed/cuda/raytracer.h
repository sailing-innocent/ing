#pragma once
/**
 * @file: testbed/cuda/raytracer.h
 * @author: sailing-innocent
 * @create: 2022-12-01
 * @desp: The Raytracer implementation using cuda
*/

#include <testbed/cuda/cuda_gl.cuh>

ING_NAMESPACE_BEGIN

class RayTracerCudaGL: public TestbedCudaGL
{
public:
    RayTracerCudaGL() = default;
    ~RayTracerCudaGL();
protected:
    void gen_verticies();
};

ING_NAMESPACE_END
