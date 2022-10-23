#include <ing/app/vk_hello.hpp>
#include <ing/utils/vk/vk_utils.hpp>
ING_NAMESPACE_BEGIN

void HelloTriangleApplication::initWindow()
{
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    mWindow = glfwCreateWindow(mWidth, mHeight, "VK_HELLO", nullptr, nullptr);
}

void HelloTriangleApplication::mainLoop()
{
    while (!glfwWindowShouldClose(mWindow))
    {
        glfwPollEvents();
    }
}

void HelloTriangleApplication::cleanup()
{
    vkDestroyDevice(mDevice, nullptr);
    if (mEnableValidationLayers) {
        DestroyDebugUtilsMessengerEXT(mInstance, mDebugMessenger, nullptr);
    }
    vkDestroyInstance(mInstance, nullptr);
    glfwDestroyWindow(mWindow);
    glfwTerminate();
}

void HelloTriangleApplication::initVulkan()
{
    createInstance();
    setupDebugMessenger();
    pickPhysicalDevice();
}

void HelloTriangleApplication::createInstance()
{
    if (mEnableValidationLayers && !checkValidationLayerSupport()) {
        throw std::runtime_error("validation layers requested but not available~");
    }
    // fill the table 
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Hello Triangle";
    appInfo.applicationVersion = VK_MAKE_VERSION(1,0,0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_0;
    // leave pNext nullptr
    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};

    if (mEnableValidationLayers) {
        createInfo.enabledLayerCount = static_cast<uint32_t>(mValidationLayers.size());
        createInfo.ppEnabledLayerNames = mValidationLayers.data();

        populateDebugMessengerCreateInfo(debugCreateInfo);
        debugCreateInfo.pfnUserCallback = debugCallback;
        createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*) &debugCreateInfo;
    } else {
        createInfo.enabledLayerCount = 0;
        createInfo.pNext = nullptr;
    }
    auto extensions = getRequiredExtensions();
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();
    
    // SUBMIT TABLE
    VkResult result = vkCreateInstance(&createInfo, nullptr, &mInstance);

    if (result != VK_SUCCESS) {
        throw std::runtime_error("failed to create instance!");
    }
}

bool HelloTriangleApplication::checkValidationLayerSupport()
{
    uint32_t layerCount;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

    std::vector<VkLayerProperties> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

    for (const char* layerName: mValidationLayers) {
        bool layerFound = false;
        for (const auto& layerProperties: availableLayers) {
            if (strcmp(layerName, layerProperties.layerName) == 0) {
                layerFound = true;
                break;
            }
        }
        if (!layerFound) {
            return false;
        }
    }
    return true;
}

std::vector<const char *> HelloTriangleApplication::getRequiredExtensions()
{
    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions;
    glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

    if (mEnableValidationLayers) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME); // equal to string VK_EXT_debug_utils
    }

    return extensions;
}

void HelloTriangleApplication::setupDebugMessenger()
{
    if (!mEnableValidationLayers) return;
    // now fill in a structure with details about the messager and its callback
    VkDebugUtilsMessengerCreateInfoEXT createInfo{};
    populateDebugMessengerCreateInfo(createInfo);
    createInfo.pfnUserCallback = debugCallback;
    if (CreateDebugUtilsMessengerEXT(mInstance, &createInfo, nullptr, &mDebugMessenger) != VK_SUCCESS) {
        throw std::runtime_error("failed to set up debug messenger!");
    }
}

void HelloTriangleApplication::pickPhysicalDevice()
{
    // listing the graphics cards is very similiar to listing extensions and starts with querying just the number
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(mInstance, &deviceCount, nullptr);
    if (deviceCount == 0) {
        throw std::runtime_error("failed to find GPUs with Vulkan support!");
    }
    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(mInstance, &deviceCount, devices.data());

    // now we only choose device 0
    for (const auto& device: devices) {
        if (isDeviceSuitable(device)) {
            mPhysicalDevice = device;
            break;
        }
    }
    // you can also give each device a score and check the highest one

    if (mPhysicalDevice == VK_NULL_HANDLE) {
        throw std::runtime_error("failed to find a suitable GPU!");
    }
}

void HelloTriangleApplication::createLogicalDevice()
{
    QueueFamilyIndices indices = findQueueFamilies(mPhysicalDevice);
    // now we are only interested in a queue with graphics capabilities
    VkDeviceQueueCreateInfo queueCreateInfo{};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = indices.graphicsFamily.value();
    queueCreateInfo.queueCount = 1;

    float queuePriority = 1.0f;
    queueCreateInfo.pQueuePriorities = &queuePriority;

    // set of device features we will be using
    VkPhysicalDeviceFeatures deviceFeatures{};
    // Creating the logical device
    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.pQueueCreateInfos = &queueCreateInfo;
    createInfo.queueCreateInfoCount = 1;

    createInfo.pEnabledFeatures = &deviceFeatures;
    createInfo.enabledExtensionCount = 0;

    if (mEnableValidationLayers) {
        createInfo.enabledLayerCount = static_cast<uint32_t>(mValidationLayers.size());
        createInfo.ppEnabledLayerNames = mValidationLayers.data();
    } else {
        createInfo.enabledLayerCount = 0;
    }

    if (vkCreateDevice(mPhysicalDevice, &createInfo, nullptr, &mDevice) != VK_SUCCESS) {
        throw std::runtime_error("failed to create logical device!");
    }

    vkGetDeviceQueue(mDevice, indices.graphicsFamily.value(), 0, &mGraphicsQueue);
}

ING_NAMESPACE_END
