#pragma once
/**
 * @file: include/app/vk_hello.hpp
 * @author: sailing-innocent
 * @create: 2022-10-23
*/

#ifndef ING_APP_VK_HELLO_H_
#define ING_APP_VK_HELLO_H_

#include <ing/common.h>

#include <vulkan/vulkan.h>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

ING_NAMESPACE_BEGIN

class HelloTriangleApplication {
public:
    void run() {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }
private:
    void initWindow();
    void initVulkan();
    void mainLoop();
    void cleanup();

private:
    GLFWwindow* mWindow = NULL;
    int mWidth = 800; // in pixel
    int mHeight = 600; // in pixel
};

ING_NAMESPACE_END

#endif // ING_APP_VK_HELLO_H_