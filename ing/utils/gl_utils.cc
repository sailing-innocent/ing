#include <ing/utils/gl/gl_utils.hpp>

ING_NAMESPACE_BEGIN


void processInput(GLFWwindow *window)
{
    if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}

const char* readSource(std::string& sourcePath) {
    std::vector<char> sourceBuffer = readFile(sourcePath);
    char* source = new char[sourceBuffer.size()];
    for (auto i = 0; i < sourceBuffer.size(); i++) {
        source[i] = sourceBuffer[i];
    }
    // std::cout << "SOURCE:\n" << source << std::endl;
    return source;
}

ING_NAMESPACE_END