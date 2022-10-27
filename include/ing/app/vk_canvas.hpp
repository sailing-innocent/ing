#pragma once 
/**
 * @file: include/app/vk_canvas.hpp
 * @author: sailing-innocent
 * @create: 2022-10-24
 * @desp: the canvas app for multiple static
*/

#ifndef ING_APP_VK_CANVAS_H_
#define ING_APP_VK_CANVAS_H_

#include <ing/app/vk_hello.hpp>

ING_NAMESPACE_BEGIN

class CanvasApp: public HelloTriangleApplication
{
public:
    void init();
    void run();
    void terminate();
    // void setVertices(std::vector<INGVertex>& vertices);
    // void setIndices(std::vector<uint32_t>& indices);
protected:
    void initVulkan();
    void cleanup();
    void mainLoop();
    void createVertexBuffer();
    void createIndexBuffer();
    void drawFrame();
protected:
    void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex);
protected:
    VkBuffer mIndexBuffer;
    VkDeviceMemory mIndexBufferMemory;
    const std::vector<VkOutVertex> mVertices = {
        {{-0.5f, -0.5f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}},
        {{0.5f, -0.5f, 0.0f, 1.0f}, {0.0f, 1.0f, 0.0f, 1.0f}},
        {{0.5f, 0.5f, 0.0f, 1.0f}, {0.0f, 0.0f, 1.0f, 1.0f}},
        {{-0.5f, 0.5f, 0.0f, 1.0f}, {1.0f, 1.0f, 1.0f, 1.0f}}
    };
    const std::vector<uint16_t> mIndices = {
        0, 1, 2, 2, 3, 0
    };
};

ING_NAMESPACE_END

#endif // ING_APP_VK_CANVAS_H_