#include "testbed/cuda/cuda_gl.cuh"
#include <iostream>

__global__ void createVerticesEx(float4* positions, float time, unsigned int width, unsigned int height)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    // calculate uv coordinate
    float u = x / (float)width;
    float v = y / (float)height;

    u = u * 2.0f - 1.0f;
    v = v * 2.0f - 1.0f;

    // calculate simple sine wave pattern
    float freq = 4.0f;
    // float w = sinf(u * freq + time) * cosf(v * freq + time) * 0.5f;
    float w = sinf(freq * sqrtf( u * u + v * v) - time * 6.0f);
    // write position
    positions[y*width + x] = make_float4(u, v, w, 1.0f);
}

class CudaTestbedEx: public ing::TestbedCudaGL
{
public:
    CudaTestbedEx() = default;
    ~CudaTestbedEx() {
        destroy_buffers();
        destroy_window();
    };
protected:
    void gen_verticies() {
        float timeValue = static_cast<float>(glfwGetTime());
        float4* positions;
        cudaGraphicsMapResources(1, &m_positionsVBO_CUDA, 0);
        size_t num_bytes;
        cudaGraphicsResourceGetMappedPointer((void**)&positions, &num_bytes, m_positionsVBO_CUDA);
        dim3 dimBlock(16, 16, 1);
        dim3 dimGrid(m_window_res[0]/dimBlock.x, m_window_res[1]/dimBlock.y, 1);
        createVerticesEx<<<dimGrid, dimBlock>>>(positions, timeValue, m_window_res[0], m_window_res[1]);
        cudaGraphicsUnmapResources(1, &m_positionsVBO_CUDA, 0);
    }
};

int main(int argc, char** argv)
{
    try {
        CudaTestbedEx testbed{};
        testbed.init();
        while (!testbed.frame()) {}
    } catch (const std::exception& e) {
        std::cerr << "EXCEPTION: " << std::endl;
        return 1;
    }
    return 0;
}