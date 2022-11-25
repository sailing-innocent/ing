#include <ing/app/gl_common.hpp>

std::string _vertPath_canvas = "E:/assets/shaders/canvas/shader.vert";
std::string _fragPath_canvas = "E:/assets/shaders/canvas/shader.frag";
std::string _vertPath_scene = "E:/assets/shaders/glscene/shader.vert";
std::string _fragPath_scene = "E:/assets/shaders/glscene/shader.frag";

int main() {
    ing::GLPoint p1(0.5f, -0.5f, 0.0f);
    ing::GLPoint p2(-0.5f, -0.5f, 0.0f);
    ing::GLPoint p3(0.0f, 0.5f, 0.0f);
    ing::GLPoint p4(0.0f, -0.7f, 0.0f);
    ing::GLPoint p5(0.6f, 0.6f, 0.0f);
    std::vector<float> blue = { 0.0f, 0.0f, 1.0f, 1.0f }; 
    p4.setColor(blue);
    p1.setColor(blue);

    ing::GLTriangle tr1(p1, p2, p3);
    ing::GLTriangle tr2(p1, p4, p2);
    ing::GLTriangle tr3(p1, p3, p5);

    std::vector<ing::GLTriangle> _trlist = {tr1, tr2};
    ing::GLTriangleList trlist(_trlist);
    ing::GLPoint p6(-0.8f, 0.0f, 0.0f);
    ing::GLPoint p7(0.8f, 0.0f, 0.0f);
    ing::GLLine axis_x(p6, p7);
    ing::GLPoint p8(0.0f, -0.8f, 0.0f);
    ing::GLPoint p9(0.0f, 0.8f, 0.0f);
    ing::GLLine axis_y(p8, p9);

    ing::GLCommonApp app;
    app.addTriangles(trlist);
    app.addTriangles(tr3);
    // app.addPoints(p6);
    app.addLines(axis_x);
    app.addLines(axis_y);
    app.init();
    size_t count;
    count = app.addShader(_vertPath_scene, _fragPath_scene);
    count = app.addShader(_vertPath_canvas, _fragPath_canvas);
    std::cout << "Now we have " << count << " Shaders. " << std::endl;
    int i = 0;
    while (!app.shouldClose()) {
        app.tick(i);
        i++;
    }
    app.terminate();
    return 0;
}