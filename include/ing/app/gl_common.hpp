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
#include <ing/utils/gl/gl_shader.h>
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
public:
    virtual void setVertices(std::vector<float>& _vertices);
    virtual void setIndices(std::vector<unsigned int>& _indices);
protected:
    virtual void initWindow();
    virtual void initGL();
    virtual void tick();
    virtual void cleanup();
    virtual void bindVertexBuffer();

protected:
    GLFWwindow* mWindow = NULL;
    unsigned int mWidth = 800;
    unsigned int mHeight = 600;
    std::string mVertexShaderPath; // = "D:/repos/inno/engine/shader/glsl/basic.vert";
    std::string mFragmentShaderPath; // = "D:/repos/inno/engine/shader/glsl/basic.frag";
    std::vector<float> mVertices; // = { -0.5f, -0.5f, 0.0f, 0.5f, -0.5f, 0.0f, 0.0f, 0.5f, 0.0f };
    std::vector<unsigned int> mIndices;
    GLShader mShader;
    unsigned int mVertexBufferObject;
    unsigned int mVertexArrayObject;
    unsigned int mElementBufferObject;
};

ING_NAMESPACE_END

#endif // ING_APP_GL_COMMON_H_