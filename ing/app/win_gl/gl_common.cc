#include <glad/glad.h>
#include <ing/app/gl_common.hpp>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>


ING_NAMESPACE_BEGIN

void GLCommonApp::init() {
    // glfw: initialize and configure
    // ------------------------------
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // init window
    initWindow();
    // init gl
    initGL();
    // build shader
    mShader = *(new GLShader(mVertexShaderPath, mFragmentShaderPath));
    mStartTime = static_cast<float>(glfwGetTime());
    // mShader2 = *(new GLShader("D:/repos/inno/engine/shader/glsl/plain.vert", "D:/repos/inno/engine/shader/glsl/plain.frag"));    
    // bind Vertex Buffer
    bindVertexBuffer();
}

void GLCommonApp::initWindow() {
    // glfw window creation
    // --------------------
    mWindow = glfwCreateWindow(mWidth, mHeight, "LearnOpenGL", NULL, NULL);
    if (mWindow == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
    }
    glfwMakeContextCurrent(mWindow);
    glfwSetFramebufferSizeCallback(mWindow, framebuffer_size_callback);
}

void GLCommonApp::initGL() {
    // glad: load all OpenGL function pointers
    // ---------------------------------------
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
    }    
}
void GLCommonApp::bindVertexBuffer() {
    // construct triangles
    mTriangleOffsetStart = 0;
    mPrimitiveRoot.appendPrimitive(mTriangles);
    mTriangleOffsetEnd = mPrimitiveRoot.indicies().size();
    mLineOffsetStart = mTriangleOffsetEnd;
    mPrimitiveRoot.appendPrimitive(mLines);
    mLineOffsetEnd = mPrimitiveRoot.indicies().size();
    mPointOffsetStart = mLineOffsetEnd;
    mPrimitiveRoot.appendPrimitive(mPoints);
    mPointOffsetEnd = mPrimitiveRoot.indicies().size();
    // gen buffers
    glGenBuffers(1, &mPrimitiveRoot.VBO());
    glGenBuffers(1, &mPrimitiveRoot.EBO());
    glGenVertexArrays(1, &mPrimitiveRoot.VAO());
    glBindVertexArray(mPrimitiveRoot.VAO());
    glBindBuffer(GL_ARRAY_BUFFER, mPrimitiveRoot.VBO());
    float* vertices = new float[mPrimitiveRoot.vertices().size()];
    for (auto i = 0; i < mPrimitiveRoot.vertices().size(); i++) {
        vertices[i] = mPrimitiveRoot.vertices()[i];
    }
    glBufferData(
        GL_ARRAY_BUFFER,
        mPrimitiveRoot.vertices().size() * sizeof(float), 
        vertices,
        GL_STATIC_DRAW  
    );
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mPrimitiveRoot.EBO());
    unsigned int* indices = new unsigned int[mPrimitiveRoot.indicies().size()];
    for (auto i = 0; i < mPrimitiveRoot.indicies().size(); i++) {
        indices[i] = mPrimitiveRoot.indicies()[i];
        // std::cout << indices[i] << ",";
    }
    // std::cout << std::endl;
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, mPrimitiveRoot.indicies().size() * sizeof(unsigned int), indices, GL_STATIC_DRAW);

    glVertexAttribPointer(
        0,  // which vertex attribute .. something like "position = 0" in shader
        4,  // sizeof data attribute 
        GL_FLOAT, // typeof data
        GL_FALSE,  // normalized or not 
        8 * sizeof(float), // stride 
        (void*)0 // offset
    );
    glEnableVertexAttribArray(0); 
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)( 4 * sizeof(float)));
    glEnableVertexAttribArray(1); 
    
    delete(vertices);
    delete(indices);
    std::cout << mPrimitiveRoot.VAO() << " " << mPrimitiveRoot.EBO() << " " <<  mPrimitiveRoot.VBO() << std::endl;
}

bool GLCommonApp::shouldClose() {
    return glfwWindowShouldClose(mWindow);
}

bool GLCommonApp::tick(int count) {
    // std::cout << "IS TICKING" << std::endl;
    // input
    // -----
    processInput(mWindow);

    // render
    // ------
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    // draw
    float timeValue = static_cast<float>(glfwGetTime()) - mStartTime;
    float greenValue = (sin(timeValue) / 2.0f ) + 0.5f;
    // transform
    // glm::vec4 vec(1.0f, 0.0f, 0.0f, 1.0f);
    // glm::mat4 trans = glm::mat4(1.0f);
    // trans = glm::translate(trans, glm::vec3(1.0f, 1.0f, 0.0f));
    // vec = trans * vec;
    // int vertexColorLocation = glGetUniformLocation(mShaderProgram, "ourColor");
    // glUseProgram(mShaderProgram);
    // glm::mat4 trans = glm::mat4(1.0f);
    // trans = glm::rotate(trans, glm::radians(90.0f), glm::vec3(0.0, 0.0, 1.0));
    // trans = glm::scale(trans, glm::vec3(0.5, 0.5, 0.5));  
    mShader.use();
    // mShader.setFloat("ourColor", greenValue)
    mShader.setFloat4("ourColor", 0.0f, greenValue, 0.0f, 1.0f);
    mShader.setFloat("t", timeValue);
    // unsigned int transformLoc = glGetUniformLocation(mShader.ID, "transform");
    // glUniformMatrix4fv(transformLoc, 1, GL_FALSE, glm::value_ptr(trans));

    // glDrawArrays(GL_TRIANGLES, 0, mVertices.size());
    // glDrawElements(GL_TRIANGLES, mTriangleOffsetEnd - mTriangleOffsetStart + 1, GL_UNSIGNED_INT, (void*)(mTriangleOffsetStart * sizeof(unsigned int)));
    // mShader2.use();
    // glDrawElements(GL_LINES,  mLineOffsetEnd - mLineOffsetStart + 1, GL_UNSIGNED_INT, (void*)(mLineOffsetStart * sizeof(unsigned int)));
    
    // glBindVertexArray(mVertexArrayObject);
    // glDrawElements(GL_TRIANGLES, static_cast<unsigned int>(mIndices.size()) , GL_UNSIGNED_INT, (void*)(0));
    // glBindVertexArray(0);


    glBindVertexArray(mPrimitiveRoot.VAO());
    // draw triangles
    if (mTriangleOffsetEnd > mTriangleOffsetStart) {
        glDrawElements(GL_TRIANGLES, mTriangleOffsetEnd - mTriangleOffsetStart, GL_UNSIGNED_INT, (void*)(mTriangleOffsetStart * sizeof(unsigned int)));
    }
    if (mLineOffsetEnd > mLineOffsetStart) {
        glDrawElements(GL_LINES, mLineOffsetEnd - mLineOffsetStart, GL_UNSIGNED_INT, (void*)(mLineOffsetStart * sizeof(unsigned int)));
    }
    if (mPointOffsetEnd > mPointOffsetStart) {
        glDrawElements(GL_POINTS, mPointOffsetEnd - mPointOffsetStart, GL_UNSIGNED_INT, (void*)(mPointOffsetStart * sizeof(unsigned int)));
    }

    // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
    // -------------------------------------------------------------------------------
    glfwSwapBuffers(mWindow);
    glfwPollEvents(); 

    return true;
}

void GLCommonApp::terminate() {
    cleanup();
    // glfw: terminate, clearing all previously allocated GLFW resources.
    // ------------------------------------------------------------------
    glfwTerminate();
}

void GLCommonApp::cleanup() {

}

void GLCommonApp::setVertices(std::vector<float>& _vertices) {
    mVertices.resize(_vertices.size());
    for (auto i = 0; i < _vertices.size(); i++) {
        mVertices[i] = _vertices[i];
    }
}

void GLCommonApp::setIndices(std::vector<unsigned int>& _indices) {
    mIndices.resize(_indices.size());
    for (auto i = 0; i < _indices.size(); i++) {
        mIndices[i] = _indices[i];
    }
}

ING_NAMESPACE_END
