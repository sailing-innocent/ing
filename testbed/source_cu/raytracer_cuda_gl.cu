#include <testbed/cuda/raytracer.h>
#include <testbed/cuda/utils/tracer.cuh>

ING_NAMESPACE_BEGIN

RayTracerCudaGL::~RayTracerCudaGL() {

}

void RayTracerCudaGL::gen_verticies()
{
    float timeValue = static_cast<float>(glfwGetTime());
    float* positions;

    cudaGraphicsMapResources(1, &m_positionsVBO_CUDA, 0);
    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer((void**)&positions, &num_bytes, m_positionsVBO_CUDA);
    dim3 dimBlock(16, 16, 1);
    dim3 dimGrid(m_window_res[0]/dimBlock.x, m_window_res[1]/dimBlock.y, 1);
    
    // init triangles

    
    createVertices1<<<dimGrid, dimBlock>>>(positions, timeValue, m_window_res[0], m_window_res[1]);
    cudaGraphicsUnmapResources(1, &m_positionsVBO_CUDA, 0);
}

ING_NAMESPACE_END
