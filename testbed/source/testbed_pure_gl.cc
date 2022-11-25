/**
 * @file: testbed_pure_gl.cc
 * @author: sailing-innocent
 * @create: 2022-11-25
 * @desp: The Common Testbed Implement For Pure OpenGL
*/

#include "testbed/pure/pure_gl.h"
#include <iostream>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <ing/utils/gl/gl_utils.hpp>

ING_NAMESPACE_BEGIN

TestbedPureGL::~TestbedPureGL() { destroy_window(); }

void TestbedPureGL::init_window(int resw, int resh) 
{
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    mWindow = glfwCreateWindow(resw, resh, "Testbed_PURE_GL", NULL, NULL);
    if (mWindow == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
    }
    glfwMakeContextCurrent(mWindow);
    glfwSetFramebufferSizeCallback(mWindow, framebuffer_size_callback);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initalize GLAD" << std::endl;
    }
}

void TestbedPureGL::destroy_window()
{
    glfwTerminate();
}

bool TestbedPureGL::frame()
{
    processInput(mWindow);
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glfwSwapBuffers(mWindow);
    glfwPollEvents();

    return glfwWindowShouldClose(mWindow);
}

bool TestbedPureGL::begin_frame()
{
    return true;
}

void TestbedPureGL::draw_gui()
{
}

ING_NAMESPACE_END
