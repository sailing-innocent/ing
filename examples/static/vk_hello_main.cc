#include <ing/app/vk_hello.hpp>

using namespace ing;

int main() {
    HelloTriangleApplication app;

    try {
        app.init();
        app.run();
        app.terminate();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}
