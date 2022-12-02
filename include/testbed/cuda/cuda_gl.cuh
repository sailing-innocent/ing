#pragma once
/**
 * @file: testbed/cuda/cuda_gl.h
 * @author: sailing-innocent
 * @create: 2022-11-25
 * @desp: The CUDA GL Binding Testbed 
*/

#include <testbed/pure/pure_gl.h>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <ing/utils/gl/gl_shader.h>

ING_NAMESPACE_BEGIN

class TestbedCudaGL: public TestbedPureGL {
public: 
    TestbedCudaGL();
    TestbedCudaGL(ITestbedMode mode): TestbedPureGL(mode) {}
    ~TestbedCudaGL();
    bool frame() override;
    void init_buffers() override;
    void destroy_buffers();
protected:
    virtual void gen_verticies();

protected:
    unsigned int m_cuda_device = 0;
    unsigned int m_positions_VBO;
    unsigned int m_VAO;
    struct cudaGraphicsResource* m_positionsVBO_CUDA;
    GLShader m_shader;
};

ING_NAMESPACE_END
