#include <testbed/cuda/utils/tracer.cuh>
#include <testbed/cuda/utils/ray.cuh>

ING_NAMESPACE_BEGIN

ING_CU_HOST_DEVICE color ray_color(const Ray& r, Sphere& s)
{
    hit_record rec;
    if (s.hit(r, 0, INFINITY, rec)) {
        return 0.5 * (rec.normal + color(1,1,1));
    }
    // background is blended blue & white
    vec4 unit_direction = unit_vector(r.dir());
    auto t = 0.5 * (unit_direction.y() + 1.0);
    return (1.0 - t)*color(1.0f, 1.0f, 1.0f) + t * color(0.5, 0.7, 1.0);
}

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

    float aspect_ratio = (float)width/(float)height;

    // camera
    float viewport_height = 2.0f;
    float viewport_width = aspect_ratio * viewport_height;
    float focal_length = 1.0f;

    point origin = point(0,0,0);
    auto horizontal = vec4(viewport_width, 0, 0, 0);
    auto vertical = vec4(0, viewport_height, 0, 0);
    point lower_left_corner = origin - horizontal/2 - vertical/2 - vec4(0, 0, focal_length, 0);

    // generate ray
    point ray_dir = lower_left_corner + u * horizontal + v * vertical - origin;
    Ray ray{origin, ray_dir};

    // if (world.hit(ray)) color = ray_color(world, ray)
    Sphere s;
    color c = ray_color(ray, s);

    // add debug info
    color coverc{(blockIdx.x+blockIdx.y)%2,1.0f,0.0f};
    c = (coverc + c)/2;
    // write vertex and color
    positions[8*(y*width+x)+0] = ou;
    positions[8*(y*width+x)+1] = ov;
    positions[8*(y*width+x)+2] = 0.0f;
    positions[8*(y*width+x)+3] = 1.0f;
    // color
    positions[8*(y*width+x)+4] = c[0];
    positions[8*(y*width+x)+5] = c[1];
    positions[8*(y*width+x)+6] = c[2];
    positions[8*(y*width+x)+7] = c[3];
}

ING_NAMESPACE_END