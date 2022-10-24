#pragma once
/**
 * @file: include/app/vk_scene.hpp
 * @author: sailing-innocent
 * @create: 2022-10-24
 * @desp: the scene app for 3D view
*/

#ifndef ING_APP_VK_SCENE_H_
#define ING_APP_VK_SCENE_H_

#include <ing/app/vk_canvas.hpp>

ING_NAMESPACE_BEGIN

const int MAX_FRAMES_IN_FLIGHT = 2;

struct UniformBufferObject {
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 proj;
};

class SceneApp: public CanvasApp
{
public:
    void init();
    void run();
    void terminate();
protected:
    void initVulkan();
    void cleanup();
    void mainLoop();
    void createDescriptorSetLayout();
    void createGraphicsPipeline();
    void createUniformBuffers();
    void createDescriptorPool();
    void createDescriptorSets();

    void drawFrame();

protected:
    void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex);
    void updateUniformBuffer(uint32_t currentImage);

protected:
    VkDescriptorSetLayout mDescriptorSetLayout;
    VkPipelineLayout mPipelineLayout;

    std::vector<VkBuffer> mUniformBuffers;
    std::vector<VkDeviceMemory> mUniformBuffersMemory;
    VkDescriptorPool mDescriptorPool;
    std::vector<VkDescriptorSet> mDescriptorSets;
    uint32_t currentFrame = 0;
};

ING_NAMESPACE_END

#endif // ING_APP_VK_SCENE_H_