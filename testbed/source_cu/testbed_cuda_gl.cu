#include <testbed/cuda/cuda_gl.cuh>

#include <cuda_gl_interop.h>
#include <ing/utils/gl/gl_utils.hpp>
#include <iostream>
#include <testbed/cuda/utils/base.cuh>

// opengl interoperaily
// a buffer object is registered using cudaGraphicsGLRegisterBuffer() -> a device pointer that can be read and written using cudaMemcpy calls
// A texture or renderbuffer object is registered using cudaGraphicsGLRegisterImage() -> CUDA array
// Kernels can read from the array by binding it to a texture or surface reference
// they can also write to it via the surface write functions
// cudaGraphicsRegisterFlagsSurfaceLoadStore()
// cudaMemcpy2D() calls
// internal types of GL_RGBA_FLOAT32, GL_RGBA_8, GL_INTENSITY16 or GL_RGBA8UI

__device__ float trans(float u) {
    return u * 2.0f - 1.0f;
}

__global__ void createVertices(float* positions, float time, unsigned int width, unsigned int height)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    // calculate uv coordinate
    float u = x / (float)width;
    float v = y / (float)height;

    u = trans(u);
    v = trans(v);

    // calculate simple sine wave pattern
    float freq = 18.0f;
    // float w = sinf(u * freq + time) * cosf(v * freq + time) * 0.5f;
    float w = sinf(freq * sqrtf( u * u + v * v) - time * 6.0f);
    // write position
    // positions[y*width + x] = make_float4(u, v, w, 1.0f);
    positions[8*(y*width+x)+0] = u;
    positions[8*(y*width+x)+1] = v;
    positions[8*(y*width+x)+2] = w;
    positions[8*(y*width+x)+3] = 1.0f;
    // generate color
    positions[8*(y*width+x)+4] = w * 2.0f;
    positions[8*(y*width+x)+5] = 0.3f;
    positions[8*(y*width+x)+6] = 0.8f;
    positions[8*(y*width+x)+7] = 1.0f;
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
    unsigned int size = m_window_res[0] * m_window_res[1] * 8 * sizeof(float);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(0));
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(4 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    cudaSetDevice(0);
    cudaGraphicsGLRegisterBuffer(
        &m_positionsVBO_CUDA, 
        m_positions_VBO, 
        cudaGraphicsMapFlagsWriteDiscard
    );
}

void TestbedCudaGL::gen_verticies()
{
    float timeValue = static_cast<float>(glfwGetTime());
    float* positions;
    cudaGraphicsMapResources(1, &m_positionsVBO_CUDA, 0);
    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer((void**)&positions, &num_bytes, m_positionsVBO_CUDA);
    dim3 dimBlock(16, 16, 1);
    dim3 dimGrid(m_window_res[0]/dimBlock.x, m_window_res[1]/dimBlock.y, 1);
    createVertices<<<dimGrid, dimBlock>>>(positions, timeValue, m_window_res[0], m_window_res[1]);
    cudaGraphicsUnmapResources(1, &m_positionsVBO_CUDA, 0);
}

bool TestbedCudaGL::frame()
{
    processInput(mWindow);
    gen_verticies();

    // Render from buffer object
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glBindVertexArray(m_VAO);
    m_shader.use();
    glBindBuffer(GL_ARRAY_BUFFER, m_positions_VBO);
    glDrawArrays(GL_POINTS, 0, m_window_res[0] * m_window_res[1]);
    // glDrawArrays(GL_TRIANGLES, 0, 3);
    // glDrawArrays(GL_POINTS, 0, 3);
    glfwSwapBuffers(mWindow);
    glfwPollEvents();

    return glfwWindowShouldClose(mWindow);
}

void TestbedCudaGL::destroy_buffers()
{
    cudaGraphicsUnregisterResource(m_positionsVBO_CUDA);
    glDeleteBuffers(1, &m_positions_VBO);
}

ING_NAMESPACE_END
