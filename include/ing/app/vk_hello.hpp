#pragma once
/**
 * @file: include/app/vk_hello.hpp
 * @author: sailing-innocent
 * @create: 2022-10-23
*/

#ifndef ING_APP_VK_HELLO_H_
#define ING_APP_VK_HELLO_H_

#include <ing/app/vk_common.hpp>

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
// procedure methods
    void initWindow();
    void initVulkan();
    void mainLoop();
    void cleanup();

    void createInstance();
    void setupDebugMessenger();
    void pickPhysicalDevice();
    void createLogicalDevice();

private:
// utils methods
    bool checkValidationLayerSupport();
    std::vector<const char*> getRequiredExtensions();
    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
        VkDebugUtilsMessageTypeFlagsEXT messageType,
        const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
        void* pUserData) {
            if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
                std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
            }
            return VK_FALSE;
        }

private:
    GLFWwindow* mWindow = NULL;
    const uint32_t mWidth = 800; // in pixel
    const uint32_t mHeight = 600; // in pixel
    VkInstance mInstance;
    VkDebugUtilsMessengerEXT mDebugMessenger;

    const std::vector<const char*> mValidationLayers = {
        "VK_LAYER_KHRONOS_validation"
    };

    VkPhysicalDevice mPhysicalDevice = VK_NULL_HANDLE; // Physical Device Handle
    VkDevice mDevice; // Logical Device
    VkQueue mGraphicsQueue;

#ifdef NDEBUG
    bool mEnableValidationLayers = false;
#else
    bool mEnableValidationLayers = true;
#endif

};

ING_NAMESPACE_END

#endif // ING_APP_VK_HELLO_H_