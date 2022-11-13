#include <ing/app/vk_scene.hpp>

using namespace ing;

const std::string vertShaderPath = "E:/assets/shaders/scene/vert.spv";
const std::string fragShaderPath = "E:/assets/shaders/scene/frag.spv";

int main() {
    SceneApp app(vertShaderPath, fragShaderPath);
    std::vector<float> vf = {
        -0.5f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f,
        0.5f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f,
        0.0f, 0.5f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f,
    };
    std::vector<uint16_t> iu = { 0, 1, 2 };
    if (!app.setVertex(vf,vf.size())) {
        std::cerr << "input vertex failed" << std::endl; 
    }
    if (!app.setIndex(iu, iu.size())) {
        std::cerr << "input index failed" << std::endl;
    };
    try {
        app.init();
        app.run();
        app.terminate();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}