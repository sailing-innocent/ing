#include <ing/app/gl_common.hpp>

int main() {
    ing::GLCommonApp app;
    app.init();
    app.run();
    app.terminate();
    return 0;
}