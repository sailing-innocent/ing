#include <ing/app/vk_canvas.hpp>

ING_NAMESPACE_BEGIN

CanvasApp::CanvasApp(const std::string& _vertShaderPath, const std::string& _fragShaderPath):
    mVertShaderPath(_vertShaderPath),
    mFragShaderPath(_fragShaderPath)
    {}
void CanvasApp::init()
{
    VkCommonApp::initWindow();
    initVulkan();
}

void CanvasApp::terminate() {
    cleanup();
}

void CanvasApp::run() {
    mainLoop();
}

bool CanvasApp::setVertex(std::vector<float> vfloat, size_t size) {
    if (size % 8 != 0) { return false; }
    mVertices.resize(size);
    for (auto i = 0; i < size/8; i++) {
        mVertices[i] = {
            {vfloat[8*i+0],vfloat[8*i+1],vfloat[8*i+2],vfloat[8*i+3]},
            {vfloat[8*i+4],vfloat[8*i+5],vfloat[8*i+6],vfloat[8*i+7]}
        };
    }
    return true;
}

bool CanvasApp::setIndex(std::vector<uint16_t> vu16, size_t size)
{
    mIndices.resize(size);
    for (auto i = 0; i < size; i++) {
        mIndices[i] = vu16[i];
    }
    return true;
}

void CanvasApp::initVulkan()
{
    VkCommonApp::createInstance();
    VkCommonApp::setupDebugMessenger();
    VkCommonApp::createSurface();
    VkCommonApp::pickPhysicalDevice();
    VkCommonApp::createLogicalDevice();
    VkCommonApp::createSwapChain();
    VkCommonApp::createImageViews();
    VkCommonApp::createRenderPass();

    HelloTriangleApplication::createGraphicsPipeline();

    VkCommonApp::createFramebuffers();
    VkCommonApp::createCommandPool();

    createVertexBuffer();
    createIndexBuffer();

    VkCommonApp::createCommandBuffer();
    VkCommonApp::createSyncObjects();
}

void CanvasApp::createIndexBuffer()
{
    VkDeviceSize bufferSize = sizeof(mIndices[0]) * mIndices.size();

    // STAGING BUFFERS ON CPU
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    createBuffer(bufferSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        stagingBuffer,
        stagingBufferMemory);
    void * data;
    vkMapMemory(mDevice, stagingBufferMemory, 0, bufferSize, 0, &data);
    memcpy(data, mIndices.data(), (size_t)bufferSize);
    vkUnmapMemory(mDevice, stagingBufferMemory);

    // CREATE GPU BUFFER
    createBuffer(bufferSize,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        mIndexBuffer,
        mIndexBufferMemory);
    copyBuffer(stagingBuffer, mIndexBuffer, bufferSize);

    vkDestroyBuffer(mDevice, stagingBuffer, nullptr);
    vkFreeMemory(mDevice, stagingBufferMemory, nullptr);
}

void CanvasApp::mainLoop()
{
    while (!glfwWindowShouldClose(mWindow))
    {
        glfwPollEvents();
        drawFrame();
    }
    vkDeviceWaitIdle(mDevice);
}


void CanvasApp::drawFrame()
{
    vkWaitForFences(mDevice, 1, &mInFlightFence, VK_TRUE, UINT64_MAX);
    vkResetFences(mDevice, 1, &mInFlightFence);
    // acquire an image from swapchain
    uint32_t imageIndex;
    vkAcquireNextImageKHR(mDevice, mSwapChain, UINT64_MAX, mImageAvailableSemaphore, VK_NULL_HANDLE, &imageIndex);
    vkResetCommandBuffer(mCommandBuffer, 0);
    recordCommandBuffer(mCommandBuffer, imageIndex);
    // we can now submit it 
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    VkSemaphore waitSemaphores[] = {mImageAvailableSemaphore};
    VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &mCommandBuffer;

    VkSemaphore signalSemaphores[] = { mRenderFinishedSemaphore };
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores;

    // the last parameter is optional, let us know when is safe
    if (vkQueueSubmit(mGraphicsQueue, 1, &submitInfo, mInFlightFence)!= VK_SUCCESS) {
        std::runtime_error("failed to submit draw command buffer!");
    }

    // presentation
    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = signalSemaphores;
    
    VkSwapchainKHR swapChains[] = {mSwapChain};
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = swapChains;
    presentInfo.pImageIndices = &imageIndex;

    presentInfo.pResults = nullptr;
    vkQueuePresentKHR(mPresentQueue, &presentInfo);
}


void  CanvasApp::recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex)
{
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = 0;
    beginInfo.pInheritanceInfo = nullptr; // Optional

    if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
        throw std::runtime_error("failed to begin recording command buffer!");
    }
    
    VkRenderPassBeginInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = mRenderPass;
    renderPassInfo.framebuffer = mSwapChainFramebuffers[imageIndex];
    renderPassInfo.renderArea.offset = {0, 0};
    renderPassInfo.renderArea.extent = mSwapChainExtent;

    VkClearValue clearColor = {{{0.0f, 0.0f, 0.0f, 1.0f}}};
    renderPassInfo.clearValueCount = 1;
    renderPassInfo.pClearValues = &clearColor;

    vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, mGraphicsPipeline);
    
    // ADD BUFFER
    VkBuffer vertexBuffers[] = {mVertexBuffer};
    VkDeviceSize offsets[] = {0};
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
    vkCmdBindIndexBuffer(commandBuffer, mIndexBuffer, 0, VK_INDEX_TYPE_UINT16);

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(mSwapChainExtent.width);
    viewport.height = static_cast<float>(mSwapChainExtent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = mSwapChainExtent;
    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

    // vkCmdDraw(commandBuffer, static_cast<uint32_t>(mVertices.size()), 1, 0, 0); // vertex count, instance count, first vertex, fisrt instance
    vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(mIndices.size()), 1, 0, 0, 0);
    vkCmdEndRenderPass(commandBuffer);
    if (vkEndCommandBuffer(commandBuffer)!= VK_SUCCESS) {
        throw std::runtime_error("failed to record command buffer!");
    } 
}


void CanvasApp::createVertexBuffer()
{
    VkDeviceSize bufferSize = sizeof(mVertices[0]) * mVertices.size();
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    createBuffer(bufferSize, 
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT 
      | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 
        stagingBuffer, 
        stagingBufferMemory);
    // copy the data
    void* data;
    vkMapMemory(mDevice, stagingBufferMemory, 0, bufferSize, 0, &data);
    memcpy(data, mVertices.data(), (size_t) bufferSize);
    vkUnmapMemory(mDevice, stagingBufferMemory);
    
    createBuffer(bufferSize,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        mVertexBuffer,
        mVertexBufferMemory);
    
    copyBuffer(stagingBuffer, mVertexBuffer, bufferSize);

    vkDestroyBuffer(mDevice, stagingBuffer, nullptr);
    vkFreeMemory(mDevice, stagingBufferMemory, nullptr);
}


void CanvasApp::cleanup()
{
    vkDestroySemaphore(mDevice, mImageAvailableSemaphore, nullptr);
    vkDestroySemaphore(mDevice, mRenderFinishedSemaphore, nullptr);
    vkDestroyFence(mDevice, mInFlightFence, nullptr);
    vkDestroyCommandPool(mDevice, mCommandPool, nullptr);
    for (auto framebuffer : mSwapChainFramebuffers) {
        vkDestroyFramebuffer(mDevice, framebuffer, nullptr);
    }
    vkDestroyPipeline(mDevice, mGraphicsPipeline, nullptr);
    vkDestroyPipelineLayout(mDevice, mPipelineLayout, nullptr);
    vkDestroyRenderPass(mDevice, mRenderPass, nullptr);
    for (auto imageView: mSwapChainImageViews) {
        vkDestroyImageView(mDevice, imageView, nullptr);
    }
    vkDestroySwapchainKHR(mDevice, mSwapChain, nullptr);

    // DESTROY BUFFER
    vkDestroyBuffer(mDevice, mVertexBuffer, nullptr);
    vkFreeMemory(mDevice, mVertexBufferMemory, nullptr);
    vkDestroyBuffer(mDevice, mIndexBuffer, nullptr);
    vkFreeMemory(mDevice, mIndexBufferMemory, nullptr);

    vkDestroyDevice(mDevice, nullptr);
    if (mEnableValidationLayers) {
        DestroyDebugUtilsMessengerEXT(mInstance, mDebugMessenger, nullptr);
    }
    vkDestroySurfaceKHR(mInstance, mSurface, nullptr);
    vkDestroyInstance(mInstance, nullptr);
    glfwDestroyWindow(mWindow);
    glfwTerminate();
}



ING_NAMESPACE_END
