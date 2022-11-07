#pragma once

/**
 * @file: ing/app/gl_common.hpp
 * @author: sailing-innocent
 * @create: 2022-11-07
 * @desp: the common application for OpenGL Implementation
*/

#ifndef ING_APP_GL_COMMON_H_
#define ING_APP_GL_COMMON_H_

#include <iostream>
#include <ing/utils/gl/gl_utils.hpp>

ING_NAMESPACE_BEGIN

class GLCommonApp {
public:
    virtual void init();
    virtual void run();
    virtual void terminate();
protected:
    virtual void initWindow();
    virtual void initGL();
    virtual void tick();
    virtual void cleanup();

protected:
    GLFWwindow* mWindow = NULL;
    unsigned int mWidth = 800;
    unsigned int mHeight = 600;
};

ING_NAMESPACE_END

#endif // ING_APP_GL_COMMON_H_