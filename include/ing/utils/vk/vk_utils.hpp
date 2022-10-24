#pragma once 
/**
 * @file: include/utils/vk/vk_utils.hpp
 * @author: sailing-innocent
 * @create: 2022-10-23
 * @desp: the utility headers for vulkan applications
*/

#ifndef ING_UTILS_VULKAN_H_
#define ING_UTILS_VULKAN_H_

#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <vector>
#include <optional>
#include <set>
#include <cstdint> // Necessary for uint32_t
#include <limits> // Necessary for std::numeric_limits
#include <algorithm> // Necessary for std::clamp
#include <string>
#include <fstream>
#include <array>

#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan/vulkan.h>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>

#include <ing/common.h>

ING_NAMESPACE_BEGIN

VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger);
void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator);
void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo);
std::vector<char> readFile(const std::string& filename);

ING_NAMESPACE_END

#endif // ING_UTILS_VULKAN_H_