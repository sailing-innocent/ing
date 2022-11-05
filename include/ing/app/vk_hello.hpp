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

class HelloTriangleApplication: public VkCommonApp {
public:
    void init();
    void run();
    void terminate();
protected:
    void initVulkan();
    void cleanup();
    void mainLoop();
    void createGraphicsPipeline();
    void createVertexBuffer();
    void drawFrame();

protected:
    void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex);

protected:
    VkBuffer mVertexBuffer;
    VkDeviceMemory mVertexBufferMemory;
    
    const std::vector<VkOutVertex> mVertices = {
        {{0.0f, -0.5f, 0.0f, 1.0f}, {1.0f, 1.0f, 1.0f, 1.0f}},
        {{0.5f, 0.5f, 0.0f, 1.0f}, {0.0f, 1.0f, 0.0f, 1.0f}},
        {{-0.5f, 0.5f, 0.0f, 1.0f}, {0.0f, 0.0f, 1.0f, 1.0f}} 
    };
    std::string mVertShaderPath = "E:/assets/shaders/basic/vert.spv";
    std::string mFragShaderPath = "E:/assets/shaders/basic/frag.spv";
};

ING_NAMESPACE_END

#endif // ING_APP_VK_HELLO_H_