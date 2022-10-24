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


struct Vertex {
    glm::vec2 pos;
    glm::vec3 color;

    static VkVertexInputBindingDescription getBindingDescription() {
        VkVertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Vertex);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX; // INSTANCE
        return bindingDescription;
    }

    static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescriptions() {
        std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions{};
        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT; 
            // float: VK_FORMAT_R32_SFLOAT
            // vec2: VK_FORMAT_R32G32_SFLOAT
            // vec3: VK_FORMAT_R32G32B32_SFLOAT
            // vec4: VK_FORMAT_R32G32B32A32_SFLOAT
            // ivec2: VK_FORMAT_R32G32_SINT
            // uvec4: VK_FORMAT_R32G32B32A32_UINT
            // double: VK_FORMAT_R64_SFLOAT
        attributeDescriptions[0].offset = offsetof(Vertex, pos);

        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[1].offset = offsetof(Vertex, color);
        return attributeDescriptions;
    }
};

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
    
    const std::vector<Vertex> mVertices = {
        {{0.0f, -0.5f}, {1.0f, 1.0f, 1.0f}},
        {{0.5f, 0.5f}, {0.0f, 1.0f, 0.0f}},
        {{-0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}} 
    };
};

ING_NAMESPACE_END

#endif // ING_APP_VK_HELLO_H_