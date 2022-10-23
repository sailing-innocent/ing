#include <ing/app/vk_hello.hpp>

#include <iostream>
#include <stdexcept>
#include <cstdlib>

using namespace ing;

int main() {
    HelloTriangleApplication app;

    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}
