#include <ing/app/vk_hello.hpp>

using namespace ing;

int main() {
    HelloTriangleApplication app;
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* m_window = glfwCreateWindow(800, 600, "Hello Triangle", nullptr, nullptr);
    try {
        app.init(m_window);
        // app.run();
        while (!glfwWindowShouldClose(m_window)) {
            app.tick(0);
        }
        app.wait();
        app.terminate();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}
