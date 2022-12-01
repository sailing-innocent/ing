#include <iostream>
#include "testbed/cuda/raytracer.h"

using namespace ing;

int main(int argc, char** argv)
{
    std::cout << "Hello CUDA GL raytracer" << std::endl;
    try {
        RayTracerCudaGL tracer;
        tracer.init();
        while (!tracer.frame()) { }
    } catch (const std::exception& e) {
        std::cerr << "EXCEPTION: " << std::endl;
        return 1;
    }
    return 0;
}