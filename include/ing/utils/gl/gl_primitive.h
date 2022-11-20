#pragma once
#ifndef ING_UTILS_GL_PRIMITIVE_H_
#define ING_UTILS_GL_PRIMITIVE_H_

/**
 * @file: include/ing/utils/gl/gl_primitive.h
 * @author: sailing-innocent
 * @create: 2022-11-20
 * @desp: The GL Primitive Utility Header
*/

// init
// new primitive(vertex data type)
// primitive.bindShader()
// app.addPrimitve()

// draw
// primitive update

#include <ing/utils/gl/gl_utils.hpp>
#include <ing/utils/gl/gl_shader.h>

ING_NAMESPACE_BEGIN

enum GLPrimitiveType {
    GL_PRIMITIVE_TRIANGLES = 0,
    GL_PRIMITIVE_LINES,
    GL_PRIMITIVE_POINTS,
    GL_PRIMITIVE_TYPES_COUNT
};

class GLPrimitive {
public:
    GLPrimitive(GLPrimitiveType _type = GL_PRIMITIVE_TRIANGLES): mType(_type) {}
    virtual ~GLPrimitive() {}
    virtual void bindShader(GLShader& _shader) { mShader = _shader; }
    virtual void appendVertices(std::vector<float>& _vertices) { 
        size_t offset = mVertices.size();
        mVerticesCount = _vertices.size() / 8 + mVerticesCount;
        mVertices.resize(offset +_vertices.size());
        for (auto i = 0; i < _vertices.size(); i++) {
            mVertices[i + offset] = _vertices[i];
        }
    }
    virtual void appendIndicies(std::vector<unsigned int>& _indices) {
        size_t offset = mIndicies.size();
        mIndicies.resize(offset +_indices.size());
        for (auto i = 0; i < _indices.size(); i++) {
            mIndicies[i+offset] = _indices[i] + static_cast<unsigned int>(mVerticesCount);
        }
    }
    virtual void appendPrimitive(GLPrimitive& rhs) {
        appendIndicies(rhs.indicies());
        appendVertices(rhs.vertices());
    }
    virtual std::vector<float>& vertices() { return mVertices; }
    virtual std::vector<unsigned int>& indicies() { return mIndicies; }
    unsigned int & VBO() { return mVertexBufferObject; }
    unsigned int & VAO() { return mVertexArrayObject; }
    unsigned int & EBO() { return mElementBufferObject; }
protected:
    GLPrimitiveType mType = GL_PRIMITIVE_TRIANGLES;
    GLShader mShader;
    size_t mVerticesCount = 0;
    std::vector<float> mVertices;
    std::vector<unsigned int> mIndicies;
    unsigned int mVertexBufferObject;
    unsigned int mVertexArrayObject;
    unsigned int mElementBufferObject;
};

class GLPoint: public GLPrimitive {
public:
    GLPoint(float x = 0.0f, float y = 0.0f, float z = 0.0f): GLPrimitive(GL_PRIMITIVE_POINTS) {
        mVertices.resize(8);
        mIndicies.resize(1);
        mVerticesCount = 1;
        mVertices[0] = x;
        mVertices[1] = y;
        mVertices[2] = z;
        mVertices[3] = 1.0f;
        mVertices[4] = 1.0f;
        mVertices[5] = 0.0f;
        mVertices[6] = 0.0f;
        mVertices[7] = 1.0f;
        mIndicies[0] = 0u;
    }
    ~GLPoint() {}
    void setColor(std::vector<float>& _color) {
        mVertices[4] = _color[0];
        mVertices[5] = _color[1];
        mVertices[6] = _color[2];
        mVertices[7] = _color[3];
    }
protected:
};

class GLPointList: public GLPrimitive {
public:
    GLPointList(): GLPrimitive(GL_PRIMITIVE_POINTS) {}
    void appendPrimitive(GLPoint& p) {
        appendIndicies(p.indicies());
        appendVertices(p.vertices());
    }
    void appendPrimitive(GLPointList& p) {
        appendIndicies(p.indicies());
        appendVertices(p.vertices());
    }
};

class GLLine: public GLPrimitive {
public:
    GLLine(): GLPrimitive(GL_PRIMITIVE_LINES) {}
    GLLine(GLPoint& p1, GLPoint& p2) {
        appendPrimitive(p1);
        appendPrimitive(p2);
    }
};

class GLLineList: public GLPrimitive {
public:
    GLLineList(): GLPrimitive(GL_PRIMITIVE_LINES) {}
    void appendPrimitve(GLLine& rhs) {
        appendIndicies(rhs.indicies());
        appendVertices(rhs.vertices());
    }
    void appendPrimitve(GLLineList& rhs) {
        appendIndicies(rhs.indicies());
        appendVertices(rhs.vertices());
    }
};

class GLTriangle: public GLPrimitive {
public:
    GLTriangle() = default;
    GLTriangle(GLPoint& a, GLPoint& b, GLPoint& c): GLPrimitive() {
        appendPrimitive(a);
        appendPrimitive(b);
        appendPrimitive(c);
    }
};

class GLTriangleList: public GLPrimitive {
public:
    GLTriangleList() = default;
    GLTriangleList(std::vector<GLTriangle>& triangles) {
        for (auto tr: triangles) {
            appendPrimitive(tr);
        }
    }
    void appendPrimitive(GLTriangle& rhs) {
        appendIndicies(rhs.indicies());
        appendVertices(rhs.vertices());
    }
    void appendPrimitive(GLTriangleList& rhs) {
        appendIndicies(rhs.indicies());
        appendVertices(rhs.vertices());
    }
};

ING_NAMESPACE_END

#endif // ING_UTILS_GL_PRIMITIVE_H_