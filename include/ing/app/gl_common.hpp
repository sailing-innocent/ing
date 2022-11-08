#pragma once

/**
 * @file: ing/app/gl_common.hpp
 * @author: sailing-innocent
 * @create: 2022-11-07
 * @desp: the common application for OpenGL Implementation
*/

#ifndef ING_APP_GL_COMMON_H_
#define ING_APP_GL_COMMON_H_

#include <ing/utils/gl/gl_utils.hpp>
#include <ing/app.h>

ING_NAMESPACE_BEGIN

class GLCommonApp: public INGBaseApp {
public:
    GLCommonApp() = default;
    GLCommonApp(std::string& _vertexShaderPath, std::string& _fragmentShaderPath):
        mVertexShaderPath(_vertexShaderPath),
        mFragmentShaderPath(_fragmentShaderPath) {}
    void init() override;
    void run() override;
    void terminate() override;
protected:
    virtual void initWindow();
    virtual void initGL();
    virtual void tick();
    virtual void cleanup();
    virtual void createShaderProgram();
    virtual void bindVertexBuffer();

protected:
    GLFWwindow* mWindow = NULL;
    unsigned int mWidth = 800;
    unsigned int mHeight = 600;
    std::string mVertexShaderPath = "";
    std::string mFragmentShaderPath = "";
    float mVertices[9] = { -0.5f, -0.5f, 0.0f, 0.5f, -0.5f, 0.0f, 0.0f, 0.5f, 0.0f };
    unsigned int mVertexBufferObject;
    unsigned int mShaderProgram;
    unsigned int mVertexArrayObject;
};

ING_NAMESPACE_END

#endif // ING_APP_GL_COMMON_H_