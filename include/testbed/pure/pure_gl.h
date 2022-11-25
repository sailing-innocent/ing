#pragma once
#ifndef TESTBED_PURE_GL_H_
#define TESTBED_PURE_GL_H_

#include <testbed/testbed.h>

struct GLFWwindow;

ING_NAMESPACE_BEGIN

class TestbedPureGL: public TestbedBase
{
public:
    TestbedPureGL() = default;
    ~TestbedPureGL() {}
    explicit TestbedPureGL(ITestbedMode mode): TestbedBase(mode) {}
    void init_window(int resw, int resh) override;
    void destroy_window() override;
    bool frame() override;
    // class SphereTracer
    // redraw_gui_next_frame
protected:
    bool begin_frame();
    void draw_gui();

protected:
    GLFWwindow* m_glfw_window = nullptr;
    // m_render_texture
};

ING_NAMESPACE_END

#endif // TESTBED_PURE_GL_H_
