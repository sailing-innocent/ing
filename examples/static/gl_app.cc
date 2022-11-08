#include <ing/app/gl_common.hpp>

std::string _vertPath = "D:/repos/inno/engine/shader/glsl/basic.vert";
std::string _fragPath = "D:/repos/inno/engine/shader/glsl/basic.frag";

int main() {
    ing::GLCommonApp app(_vertPath, _fragPath);
    std::vector<float> _vertices = {
        // positions         // colors
        0.5f, -0.5f, 0.0f,  1.0f, 0.0f, 0.0f,   // bottom right
        -0.5f, -0.5f, 0.0f,  0.0f, 1.0f, 0.0f,   // bottom left
        0.0f,  0.5f, 0.0f,  0.0f, 0.0f, 1.0f    // top 
    }; 
    std::vector<unsigned int> _indices = {
        0, 1, 2
    };
    app.setVertices(_vertices);
    app.setIndices(_indices);
    app.init();
    int i = 0;
    while (!app.shouldClose()) {
        app.tick(i);
        i++;
    }
    app.terminate();
    return 0;
}