#include <testbed/cuda/utils/tracer.cuh>
#include <testbed/cuda/utils/ray.cuh>

ING_NAMESPACE_BEGIN

__global__ void tracer_kernel(
        float* positions, 
        float time, 
        unsigned int width, 
        unsigned int height,
        World world
    )
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    // calculate uv coordinate
    float u = x / (float)width;
    float v = y / (float)height;

    float ou = 2 * u - 1;
    float ov = 2 * v - 1;



    point origin;
    point ray_origin{0.0f, 0.0f, -1.0f};
    point ray_dir = point(ou,ov) - ray_origin;
    Ray ray{ray_origin, ray_dir};
    color c1;
    c1[0] = 1.0f;
    color c2;
    c2[1] = 1.0f;
    color c = ray.hit() ? c1 : c2;

    positions[8*(y*width+x)+0] = ou;
    positions[8*(y*width+x)+1] = ov;
    positions[8*(y*width+x)+2] = 0.0f;
    positions[8*(y*width+x)+3] = 1.0f;
    // generate color
    positions[8*(y*width+x)+4] = c[0];
    positions[8*(y*width+x)+5] = c[1];
    positions[8*(y*width+x)+6] = c[2];
    positions[8*(y*width+x)+7] = c[3];
}

ING_NAMESPACE_END