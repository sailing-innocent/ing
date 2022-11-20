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
#include <ing/utils/gl/gl_primitive.h>

#include <ing/app.h>

ING_NAMESPACE_BEGIN

class GLCommonApp: public INGApp {
public:
    GLCommonApp() = default;
    GLCommonApp(std::string& _vertexShaderPath, std::string& _fragmentShaderPath):
       mVertexShaderPath(_vertexShaderPath),
       mFragmentShaderPath(_fragmentShaderPath) {}
    void init() override;
    bool tick(int count) override;
    void terminate() override;
    bool shouldClose();
public:
    virtual void setVertices(std::vector<float>& _vertices);
    virtual void setIndices(std::vector<unsigned int>& _indices);
    virtual void addPrimitive(GLPrimitive _primitive) { 
        mPrimitiveRoot.appendPrimitive(_primitive);
    }
    virtual void addTriangles(GLTriangle _triangle) {
        mTriangles.appendPrimitive(_triangle);
    }
    virtual void addTriangles(GLTriangleList _triangles) {
        mTriangles.appendPrimitive(_triangles);
    }
    virtual void addPoints(GLPoint _point) {
        mPoints.appendPrimitive(_point);
    }
    virtual void addLines(GLLine _line) {
        mLines.appendPrimitive(_line);
    }
    virtual void addLines(GLLineList _linelist) {
        mLines.appendPrimitive(_linelist);
    }
protected:
    virtual void initWindow();
    virtual void initGL();
    virtual void cleanup();
    virtual void bindVertexBuffer();

protected:
    GLFWwindow* mWindow = NULL;
    unsigned int mWidth = 800;
    unsigned int mHeight = 600;
    std::string mVertexShaderPath; // = "D:/repos/inno/engine/shader/glsl/basic.vert";
    std::string mFragmentShaderPath; // = "D:/repos/inno/engine/shader/glsl/basic.frag";
    
    GLPrimitive mPrimitiveRoot;
    GLTriangleList mTriangles;
    GLPointList mPoints;
    GLLineList mLines;
    std::vector<float> mVertices; // = { -0.5f, -0.5f, 0.0f, 0.5f, -0.5f, 0.0f, 0.0f, 0.5f, 0.0f };
    std::vector<unsigned int> mIndices;

    GLShader mShader;
    GLShader mShader2;
    unsigned int mVertexBufferObject;
    unsigned int mVertexArrayObject;
    unsigned int mElementBufferObject;
    float mStartTime = 0.0f;
    size_t mLineOffsetStart = 0;
    size_t mLineOffsetEnd = 0;
    size_t mTriangleOffsetStart = 0;
    size_t mTriangleOffsetEnd = 0;
    size_t mPointOffsetStart = 0;
    size_t mPointOffsetEnd = 0;
};

ING_NAMESPACE_END

#endif // ING_APP_GL_COMMON_H_