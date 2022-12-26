#pragma once
/**
 * @file: include/ing/app/win_pure.hpp
 * @author: sailing-innocent
 * @create: 2022-10-15
 * @desp: Windows Pure App Header
*/

#ifndef ING_APP_PURE_WIN_APP_H_
#define ING_APP_PURE_WIN_APP_H_

#include <ing/app.h>
#include <windows.h>

ING_NAMESPACE_BEGIN

class PureWinApp: public INGApp
{
public:
    PureWinApp(HINSTANCE instanceHandle, int show):
        mInstanceHandle(instanceHandle),
        mShow(show) {}
    virtual bool Init();
    virtual int Run();

public:
    virtual void init() override {};
    virtual bool tick(int count) override { return true; };
    virtual void terminate() override {};
private:
    HINSTANCE mInstanceHandle;
    int mShow;
    HWND mGhMainWnd = 0;
};

ING_NAMESPACE_END

#endif // ING_APP_PURE_WIN_APP_H_