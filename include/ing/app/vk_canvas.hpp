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
    CanvasApp() = default;
    CanvasApp(const std::string& _vertShaderPath, const std::string& _fragShaderPath);
    bool setVertex(std::vector<float> vfloat, size_t size);
    bool setIndex(std::vector<uint16_t> vu16, size_t size);
protected:
    void cleanup();
    void createVertexBuffer();
    void createIndexBuffer();
    void createGraphicsPipeline();
    void drawFrame();
protected:
    void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex);
protected:
    VkBuffer mIndexBuffer;
    VkDeviceMemory mIndexBufferMemory;
    std::vector<VkOutVertex> mVertices = {};
    std::vector<uint16_t> mIndices = {};
    std::string mVertShaderPath = "E:/assets/shaders/canvas/vert.spv";
    std::string mFragShaderPath = "E:/assets/shaders/canvas/frag.spv";
};

ING_NAMESPACE_END

#endif // ING_APP_VK_CANVAS_H_