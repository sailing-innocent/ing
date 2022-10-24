#include <ing/app/vk_scene.hpp>

using namespace ing;

int main() {
    SceneApp app;
    
    try {
        app.init();
        app.run();
        app.terminate();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}