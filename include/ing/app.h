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
#include <string>
#include <vector>
#include <iostream>

ING_NAMESPACE_BEGIN

class INGBaseApp {
public:
    virtual void init() = 0;
    virtual void run() = 0;
    virtual void terminate() = 0;
};

ING_NAMESPACE_END

#endif // ING_APP_H_