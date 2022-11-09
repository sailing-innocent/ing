#include <ing/app/gl_common.hpp>
#include <cmath>

std::string _vertPath = "D:/repos/inno/engine/shader/glsl/basic.vert";
std::string _fragPath = "D:/repos/inno/engine/shader/glsl/basic.frag";

int main() {
    ing::GLCommonApp app(_vertPath, _fragPath);
    // samples through a line
    float gap = 0.02f;
    float start = -1.0f;
    float end = 1.0f;
    int steps = std::floor((end-start)/gap);
    std::vector<float> _vertices;
    std::vector<unsigned int> _indices;
    for (auto i = 0; i < steps; i++) {
        _vertices.push_back(start+gap*i);
        _vertices.push_back(0.0f);
        _vertices.push_back(0.0f);
        // color
        _vertices.push_back(1.0f); // red
        _vertices.push_back(0.0f);
        _vertices.push_back(0.0f);
        // index
        _indices.push_back(static_cast<unsigned int>(i));
        _indices.push_back(static_cast<unsigned int>(i+1));
    }
    _vertices.push_back(start+gap*steps);
    _vertices.push_back(0.0f);
    _vertices.push_back(0.0f);
    // color
    _vertices.push_back(1.0f); // red
    _vertices.push_back(0.0f);
    _vertices.push_back(0.0f);

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