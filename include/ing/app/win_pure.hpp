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

class PureWinApp: public BaseApp
{
public:
    PureWinApp(): BaseApp() {}
    virtual PureWinApp(PureWinApp& rhs) = delete;
    virtual PureWinApp& operator=(PureWinApp& rhs) = delete;
    virtual ~PureWinApp() override;
    virtual void Init() override;
    virtual void Run() override;
private:
    HWND mHWND;
};

ING_NAMESPACE_END

#endif // ING_APP_PURE_WIN_APP_H_