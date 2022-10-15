#pragma once

/**
 * @file: include/ing/app.h
 * @author: sailing-innocent
 * @create: 2022-10-15
 * @desp: The base app clase
*/

#ifndef ING_APP_H_
#define ING_APP_H_

#include <ing/common.h>

ING_NAMESPACE_BEGIN

class BaseApp {
public:
    BaseApp() {}
    virtual BaseApp(BaseApp& rhs) = 0;
    virtual BaseApp& operator=(BaseApp& rhs) = 0;
    virtual ~BaseApp() = 0;
    virtual void Init();
    virtual void Run();
};

ING_NAMESPACE_END



#endif // ING_APP_H_