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
    virtual void init(GLFWwindow* _m_window = NULL);
    virtual void run();
    virtual void terminate();
    virtual bool tick(float delta_time);
protected:
    // procedure
    virtual void initWindow();
    virtual void initVulkan();
    virtual void mainLoop();
    virtual void cleanup();
    virtual void createInstance();
    virtual void pickPhysicalDevice();
    virtual void createLogicalDevice();
    virtual void createSurface();
    virtual void createSwapChain();
    virtual void createImageViews();
    virtual void createRenderPass();
    virtual void createGraphicsPipeline();
    virtual void createFramebuffers();
    virtual void createCommandPool();
    virtual void createVertexBuffer();
    virtual void createCommandBuffer();
    virtual void createSyncObjects();
    virtual void drawFrame();
    virtual void setupDebugMessenger();

protected:
// utils methods
    virtual bool checkValidationLayerSupport();
    virtual std::vector<const char*> getRequiredExtensions();
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
    virtual QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device);
    virtual bool isDeviceSuitable(VkPhysicalDevice device);
    virtual bool checkDeviceExtensionSupport(VkPhysicalDevice device);
    virtual SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device);
    virtual VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats);
    virtual VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes);
    virtual VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities);
    virtual VkShaderModule createShaderModule(const std::vector<char>& code);
    virtual void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex);
    virtual void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory);
    virtual uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
    virtual void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size0);

protected:
    GLFWwindow* mWindow = NULL;
    const uint32_t mWidth = 1600; // in pixel
    const uint32_t mHeight = 1200; // in pixel
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

#endif // ING_APP_VK_COMMON_H_