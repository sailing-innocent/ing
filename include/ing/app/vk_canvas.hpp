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


//________SHOULD_BE_GENERATED_BY_GEOMETRY_PART_LATER______

const std::vector<Vertex> tempVertices = {
    {{-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}},
    {{0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},
    {{0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},
    {{-0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}}
};

const std::vector<uint16_t> tempIndices = {
    0, 1, 2, 2, 3, 0
};

// ______________________________________________________

class CanvasApp: public HelloTriangleApplication
{
public:
    void init();
    void run();
    void terminate();
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
    const std::vector<Vertex> mVertices = {
        {{-0.9f, -0.9f}, {1.0f, 0.0f, 0.0f}},
        {{0.9f, -0.9f}, {0.0f, 1.0f, 0.0f}},
        {{0.9f, 0.9f}, {0.0f, 0.0f, 1.0f}},
        {{-0.9f, 0.9f}, {1.0f, 1.0f, 1.0f}}
    };
    const std::vector<uint16_t> mIndices = {
        0, 1, 2, 2, 3, 0
    };
};

ING_NAMESPACE_END

#endif // ING_APP_VK_CANVAS_H_