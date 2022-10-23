#include <ing/app/vk_hello.hpp>
ING_NAMESPACE_BEGIN

void HelloTriangleApplication::initWindow()
{
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    mWindow = glfwCreateWindow(mWidth, mHeight, "VK_HELLO", nullptr, nullptr);
}

void HelloTriangleApplication::initVulkan()
{}

void HelloTriangleApplication::mainLoop()
{
    while (!glfwWindowShouldClose(mWindow))
    {
        glfwPollEvents();
    }
}

void HelloTriangleApplication::cleanup()
{
    glfwDestroyWindow(mWindow);
    glfwTerminate();
}

ING_NAMESPACE_END
