#include <iostream>

#include "testbed/cuda/cuda_gl.cuh"

using namespace ing;

int main(int argc, char** argv)
{
    std::cout << "Hello CUDA GL Testbed!" << std::endl;
    try {
        ITestbedMode mode{RaytraceMesh};
        TestbedCudaGL testbed{mode};
        testbed.init();
        while (!testbed.frame()) {
            // std::cout << "tick" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "EXCEPTION: " << std::endl;
        return 1;
    }
    return 0;
}