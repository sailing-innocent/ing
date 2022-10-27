#include <ing/app/vk_canvas.hpp>

using namespace ing;

int main() {
    CanvasApp app;

    try {
        app.init();
        // app.getVertices(INGVertices) // Convert to its inner vertices
        
        app.run();
        app.terminate();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}