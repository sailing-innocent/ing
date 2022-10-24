#pragma once
/**
 * @file: include/ing/all/win_vk.hpp
 * @author: sailing-innocent
 * @create: 2022-10-23
 * @desp: The common headers for vulkan applications
*/

#ifndef ING_APP_VK_COMMON_H_
#define ING_APP_VK_COMMON_H_

#include <ing/common.h>
#include <ing/utils/vk/vk_utils.hpp>

ING_NAMESPACE_BEGIN 

struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    bool isComplete() {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

class VkCommonApp {
public:
    void init();
    void run();
    void terminate();
protected:
    // procedure
    void initWindow();
    void initVulkan();
    void mainLoop();
    void cleanup();


    void createInstance();
    void pickPhysicalDevice();
    void createLogicalDevice();
    void createSurface();
    void createSwapChain();
    void createImageViews();
    void createRenderPass();
    void createGraphicsPipeline();
    void createFramebuffers();
    void createCommandPool();
    void createVertexBuffer();
    void createCommandBuffer();
    void createSyncObjects();

    void drawFrame();
 
    void setupDebugMessenger();
    void createVertexInputInfo();
protected:
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

protected:
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

    VkBuffer mVertexBuffer;
    VkDeviceMemory mVertexBufferMemory;

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

#endif // ING_APP_VK_COMMON_H_