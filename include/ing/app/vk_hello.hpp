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
    void createSurface();
    void createSwapChain();
    void createImageViews();
    void createRenderPass();
    void createGraphicsPipeline();
    void createFramebuffers();
    void createCommandPool();
    void createCommandBuffer();
    void createSyncObjects();

    void drawFrame();

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
    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device);
    bool isDeviceSuitable(VkPhysicalDevice device);
    bool checkDeviceExtensionSupport(VkPhysicalDevice device);
    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device);
    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats);
    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes);
    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities);
    VkShaderModule createShaderModule(const std::vector<char>& code);
    void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex);

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
    VkQueue mGraphicsQueue; // Directly fetch graphics queue
    VkSurfaceKHR mSurface; // On windows it is called VK_KHR_win32_surface; glfwGetRequiredInstanceExtensions
    VkQueue mPresentQueue; // Present Queue

    const std::vector<const char*> mDeviceExtensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME
    };

    VkSwapchainKHR mSwapChain;
    std::vector<VkImage> mSwapChainImages;
    VkFormat mSwapChainImageFormat;
    VkExtent2D mSwapChainExtent;
    std::vector<VkImageView> mSwapChainImageViews;

    VkRenderPass mRenderPass;
    VkPipelineLayout mPipelineLayout;
    VkPipeline mGraphicsPipeline;

    std::vector<VkFramebuffer> mSwapChainFramebuffers;
    VkCommandPool mCommandPool;
    VkCommandBuffer mCommandBuffer;

#ifdef NDEBUG
    bool mEnableValidationLayers = false;
#else
    bool mEnableValidationLayers = true;
#endif

    VkSemaphore mImageAvailableSemaphore;
    VkSemaphore mRenderFinishedSemaphore;
    VkFence mInFlightFence;
};

ING_NAMESPACE_END

#endif // ING_APP_VK_HELLO_H_