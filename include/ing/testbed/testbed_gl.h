#pragma once
#ifndef ING_TESTBED_GL_H_
#define ING_TESTBED_GL_H_

/**
 * @file: include/ing/testbed/testbed_gl.h
 * @author: sailing-innocent
 * @create: 2022-11-20
 * @desp: The Testbed Definition using GLFW and GLAD
*/

#include <ing/testbed.h>

struct GLFWwindow;

ING_NAMESPACE_BEGIN

class Triangle;

class TestbedGL: public TestbedBase
{
public:
    TestbedGL() = default;
    TestbedGL(ITestbedMode _mode): mTestbedMode(_mode) {}
    ~TestbedGL();
    void init_window(int resw, int resh);
    void destroy_window();
    bool frame();
protected:
    void render();
    void redraw_gui_next_frame() { mGuiRedraw = true; }
protected:
    int mWindowRes[2] = { 0, 0 };
    ITestbedMode mTestbedMode;
    bool mRenderWindow = true;
    bool mGuiRedraw = true; // used to delay the render of GUI
    GLFWwindow* mGLFWWindow = nullptr;
};

ING_NAMESPACE_END

#endif // ING_TESTBED_GL_H_