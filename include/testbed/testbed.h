#pragma once
#ifndef ING_TESTBED_H_
#define ING_TESTBED_H_

#include <ing/common.h>

ING_NAMESPACE_BEGIN

enum ITestbedMode {
    RaytraceMesh
};

class TestbedBase: public Base {
public:
    TestbedBase() = default;
    explicit TestbedBase(ITestbedMode mode): m_testbed_mode(mode) {}
    virtual ~TestbedBase() {}
    virtual void init() = 0;
    virtual void init_window(int resw, int resh) = 0;
    virtual void init_buffers() = 0;
    virtual void destroy_window() = 0;
    virtual void destroy_buffers() = 0;

    virtual bool frame() = 0;
protected:
    ITestbedMode m_testbed_mode;
    int m_window_res[2];
    bool m_render_window = true;
    bool m_gui_redraw = true;
};



ING_NAMESPACE_END

#endif // ING_TESTBED_H_
