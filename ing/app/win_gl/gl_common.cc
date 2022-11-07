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

ING_NAMESPACE_END
