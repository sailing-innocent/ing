#include <testbed/cuda/utils/base.cuh>

__device__ float trans(float u) {
    return u * 2.0f - 1.0f;
}

__global__ void createVertices(float* positions, float time, unsigned int width, unsigned int height)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    // calculate uv coordinate
    float u = x / (float)width;
    float v = y / (float)height;

    u = trans(u);
    v = trans(v);

    // calculate simple sine wave pattern
    float freq = 18.0f;
    // float w = sinf(u * freq + time) * cosf(v * freq + time) * 0.5f;
    float w = sinf(freq * sqrtf( u * u + v * v) - time * 6.0f);
    // write position
    // positions[y*width + x] = make_float4(u, v, w, 1.0f);
    positions[8*(y*width+x)+0] = u;
    positions[8*(y*width+x)+1] = v;
    positions[8*(y*width+x)+2] = w;
    positions[8*(y*width+x)+3] = 1.0f;
    // generate color
    positions[8*(y*width+x)+4] = w + 0.5f;
    positions[8*(y*width+x)+5] = 0.3f;
    positions[8*(y*width+x)+6] = 0.8f;
    positions[8*(y*width+x)+7] = 1.0f;
}
