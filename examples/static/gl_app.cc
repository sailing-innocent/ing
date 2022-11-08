#include <ing/app/gl_common.hpp>

std::string _vertPath = "D:/repos/inno/engine/shader/glsl/basic.vert";
std::string _fragPath = "D:/repos/inno/engine/shader/glsl/basic.frag";

int main() {
    ing::GLCommonApp app(_vertPath, _fragPath);
    std::vector<float> _vertices = {
         //  -0.5f, -0.5f, 0.0f, 0.5f, -0.5f, 0.0f, 0.0f, 0.5f, 0.0f 
        // first triangle
        0.5f,  0.5f, 0.0f,  // top right
        0.5f, -0.5f, 0.0f,  // bottom right
        -0.5f, -0.5f, 0.0f,  // bottom left
        -0.5f,  0.5f, 0.0f   // top left 
    }; 
    std::vector<unsigned int> _indices = {
        0, 1, 3,
        1, 2, 3
    };
    app.setVertices(_vertices);
    app.setIndices(_indices);
    app.init();
    app.run();
    app.terminate();
    return 0;
}