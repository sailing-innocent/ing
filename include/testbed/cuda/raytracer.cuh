#pragma once
/**
 * @file: testbed/cuda/raytracer.h
 * @author: sailing-innocent
 * @create: 2022-12-01
 * @desp: The Raytracer implementation using cuda
*/

#include <testbed/cuda/cuda_gl.cuh>
#include <testbed/cuda/utils/tracer.cuh>

ING_NAMESPACE_BEGIN

class RayTracerCudaGL: public TestbedCudaGL
{
public:
    RayTracerCudaGL() = default;
    ~RayTracerCudaGL();
protected:
    void gen_verticies();
protected:
    World m_world;
    Camera m_camera;
};

ING_NAMESPACE_END
