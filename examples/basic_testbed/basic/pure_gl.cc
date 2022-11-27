#include <iostream>

#include "testbed/pure/pure_gl.h"

using namespace ing;

int main(int argc, char** argv)
{
    std::cout << "Hello Pure GL Testbed!" << std::endl;
    try {
        ITestbedMode mode{RaytraceMesh};
        TestbedPureGL testbed{mode};
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