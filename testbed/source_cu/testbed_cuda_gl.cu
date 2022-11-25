#include <testbed/cuda/cuda_gl.cuh>

#include <cuda_gl_interop.h>
#include <ing/utils/gl/gl_utils.hpp>
#include <iostream>

// opengl interoperaily
// a buffer object is registered using cudaGraphicsGLRegisterBuffer() -> a device pointer that can be read and written using cudaMemcpy calls
// A texture or renderbuffer object is registered using cudaGraphicsGLRegisterImage() -> CUDA array
// Kernels can read from the array by binding it to a texture or surface reference
// they can also write to it via the surface write functions
// cudaGraphicsRegisterFlagsSurfaceLoadStore()
// cudaMemcpy2D() calls
// internal types of GL_RGBA_FLOAT32, GL_RGBA_8, GL_INTENSITY16 or GL_RGBA8UI

__global__ void createVertices(float4* positions, float time, unsigned int width, unsigned int height)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    // calculate uv coordinate
    float u = x / (float)width;
    float v = y / (float)height;

    u = u * 2.0f - 1.0f;
    v = v * 2.0f - 1.0f;

    // calculate simple sine wave pattern
    float freq = 8.0f;
    float w = sinf(u * freq + time) * cosf(v * freq + time) * 0.5f;

    // write position
    positions[y*width + x] = make_float4(u, v, w, 1.0f);
}

ING_NAMESPACE_BEGIN

TestbedCudaGL::TestbedCudaGL()
{}

TestbedCudaGL::~TestbedCudaGL()
{
    destroy_buffers();
    destroy_window();
}

void TestbedCudaGL::init_buffers()
{
    // init shaders
    std::string _vertPath = "E:/assets/shaders/testbed/shader.vert";
    std::string _fragPath = "E:/assets/shaders/testbed/shader.frag";
    GLShader newShader(_vertPath, _fragPath);
    m_shader = newShader;

    glGenVertexArrays(1, &m_VAO);
    glBindVertexArray(m_VAO);
    std::cout << "Is initializing buffers: " << m_window_res[0] << "," << m_window_res[1] << std::endl;

    glGenBuffers(1, &m_positions_VBO);
    glBindBuffer(GL_ARRAY_BUFFER, m_positions_VBO);
    unsigned int size = m_window_res[0] * m_window_res[1] * 4 * sizeof(float);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(0));
    glEnableVertexAttribArray(0);
    /*
    float vertices[] = {
        -0.5f, -0.5f, 0.0f, 1.0f,
        0.5f, -0.5f, 0.0f, 1.0f,
        0.0f, 0.5f, 0.0f, 1.0f
    };

    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    */
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    cudaSetDevice(0);
    cudaGraphicsGLRegisterBuffer(
        &m_positionsVBO_CUDA, 
        m_positions_VBO, 
        cudaGraphicsMapFlagsWriteDiscard
    );
    


}

bool TestbedCudaGL::frame()
{
    processInput(mWindow);

    float timeValue = static_cast<float>(glfwGetTime());
    float4* positions;
    cudaGraphicsMapResources(1, &m_positionsVBO_CUDA, 0);
    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer((void**)&positions, &num_bytes, m_positionsVBO_CUDA);
    dim3 dimBlock(16, 16, 1);
    dim3 dimGrid(m_window_res[0]/dimBlock.x, m_window_res[1]/dimBlock.y, 1);
    createVertices<<<dimGrid, dimBlock>>>(positions, timeValue, m_window_res[0], m_window_res[1]);
    cudaGraphicsUnmapResources(1, &m_positionsVBO_CUDA, 0);

    // Render from buffer object
    glClearColor(0.2f, 0.3f, 0.6f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glBindVertexArray(m_VAO);
    m_shader.use();
    glBindBuffer(GL_ARRAY_BUFFER, m_positions_VBO);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(0));
    glEnableVertexAttribArray(0);
    glDrawArrays(GL_POINTS, 0, m_window_res[0] * m_window_res[1]);
    // glDrawArrays(GL_TRIANGLES, 0, 3);
    // glDrawArrays(GL_POINTS, 0, 3);
    glfwSwapBuffers(mWindow);
    glfwPollEvents();

    return glfwWindowShouldClose(mWindow);
}

ING_NAMESPACE_END
