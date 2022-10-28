#include <ing/app/vk_canvas.hpp>

using namespace ing;

bool genIndex(std::vector<uint16_t>& vu) {
    const size_t size = 6;
    vu.resize(size);
    uint16_t indi[size] = {
        0, 1, 2, 2, 3, 0
    };
    for (auto i = 0; i < size; i++) {
        vu[i] = indi[i];
    }
    return true;
}

bool genVertex(std::vector<float>& vf) {
    vf =  {
        -0.5f, -0.5f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
        0.5f, -0.5f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f,
        0.5f, 0.5f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f,
        -0.5f, 0.5f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f
    };
    return true;
}

int main() {
    CanvasApp app;

    std::vector<float> vertices;
    genVertex(vertices);

    std::vector<uint16_t> indices;
    genIndex(indices);

    if ( !app.setVertex(vertices, vertices.size())) {
        std::cout << "init verttices failed" << std::endl;
    }
    if (!app.setIndex(indices, indices.size())) {
        std::cout << "init indices failed" << std::endl;
    }
    try {
        app.init();
        app.run();
        app.terminate();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}