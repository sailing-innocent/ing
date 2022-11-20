#pragma once
#ifndef ING_TESTBED_H_
#define ING_TESTBED_H_

#include <ing/common.h>

ING_NAMESPACE_BEGIN

class TestbedBase: public Base {
public:
    TestbedBase() = default;
    virtual ~TestbedBase() {}
    virtual void init_window(int resw, int resh) = 0;
    virtual void destroy_window() = 0;
    virtual bool frame() = 0;
};

enum ITestbedMode {
    RaytraceMesh
};

ING_NAMESPACE_END

#endif // ING_TESTBED_H_
