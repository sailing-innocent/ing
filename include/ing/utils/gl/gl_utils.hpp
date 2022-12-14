#pragma once
#ifndef ING_UTILS_GL_UTILS_HPP_
#define ING_UTILS_GL_UTILS_HPP_

#include <GLFW/glfw3.h>
#include <ing/common.h>
#include <ing/utils/common_utils.hpp>

ING_NAMESPACE_BEGIN

void processInput(GLFWwindow *window);
// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height);

const char* readSource(std::string& sourcePath);

ING_NAMESPACE_END

#endif // ING_UTILS_GL_UTILS_HPP_