#include <ing/app/gl_common.hpp>

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
    // create Shader Program
    createShaderProgram();
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

void GLCommonApp::createShaderProgram() {
    const char* vertexShaderSource = readSource(mVertexShaderPath);
    const char* fragmentShaderSource = readSource(mFragmentShaderPath);
    unsigned int vertexShader;
    vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);
    int  success;
    char infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if(!success)
    {
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    unsigned int fragmentShader;
    fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if(!success)
    {
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    mShaderProgram = glCreateProgram();
    glAttachShader(mShaderProgram, vertexShader);
    glAttachShader(mShaderProgram, fragmentShader);
    glLinkProgram(mShaderProgram);

    glGetProgramiv(mShaderProgram, GL_LINK_STATUS, &success);
    if(!success) {
        glGetProgramInfoLog(mShaderProgram, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }
    glUseProgram(mShaderProgram);
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader); 
}

void GLCommonApp::bindVertexBuffer() {
    unsigned int VBO;
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &mElementBufferObject);
    glGenVertexArrays(1, &mVertexArrayObject);

    glBindVertexArray(mVertexArrayObject);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);

    float* vertices = new float[mVertices.size()];
    for (auto i = 0; i < mVertices.size(); i++) {
        vertices[i] = mVertices[i];
    }
    glBufferData(GL_ARRAY_BUFFER, mVertices.size() * sizeof(float) , vertices, GL_STATIC_DRAW);
    // first param: which vertex attribute we want to configure
    // size of vertex attribute
    // type of data
    // normalized or not
    // stride
    // offset

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mElementBufferObject);
    unsigned int* indices = new unsigned int[mIndices.size()];
    for (auto i = 0; i < mIndices.size(); i++) {
        indices[i] = mIndices[i];
    }
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, mIndices.size() * sizeof(unsigned int), indices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0); 
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)( 3 * sizeof(float)));
    glEnableVertexAttribArray(1); 

    delete(vertices);
    delete(indices);
}


void GLCommonApp::run() {
    while (!glfwWindowShouldClose(mWindow)) {
        tick();
    }
}

void GLCommonApp::tick() {
    // input
    // -----
    processInput(mWindow);

    // render
    // ------
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    // draw
    float timeValue = glfwGetTime();
    float greenValue = (sin(timeValue) / 2.0f ) + 0.5f;
    int vertexColorLocation = glGetUniformLocation(mShaderProgram, "ourColor");
    glUseProgram(mShaderProgram);
    glUniform4f(vertexColorLocation, 0.0f, greenValue, 0.0f, 1.0f);
    glBindVertexArray(mVertexArrayObject);
    // glDrawArrays(GL_TRIANGLES, 0, mVertices.size());
    glDrawElements(GL_TRIANGLES, mIndices.size(), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
    // GL PRIMITIVE
    // starting index
    // amount of vertices

    // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
    // -------------------------------------------------------------------------------
    glfwSwapBuffers(mWindow);
    glfwPollEvents(); 
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
