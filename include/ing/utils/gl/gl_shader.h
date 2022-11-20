#pragma once
#ifndef ING_UTILS_GL_SHADER_H_
#define ING_UTILS_GL_SHADER_H_

#include <ing/utils/gl/gl_utils.hpp>

ING_NAMESPACE_BEGIN

class GLShader
{
public:
    unsigned int ID; // the program id
    GLShader() = default;
    GLShader(std::string& vertexPath, std::string& fragmentPath);
    GLShader(const char* vertexPath, const char* fragmentPath);
    void use(); // use/activate the shader
    GLShader& operator=(const GLShader& rhs) { ID = rhs.ID; return *this; }
    // utility uniform functions
    void setBool(const std::string& name, bool value) const;
    void setInt(const std::string& name, int value) const;
    void setFloat(const std::string& name, float value) const;
    void setFloat4(const std::string& name, float v0, float v1, float v2, float v3) const;
    void setMat4(const std::string& name, float* value_ptr);
};

ING_NAMESPACE_END

#endif // ING_CORE_SHADER_H_