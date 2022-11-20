/**
 * @file: ing/testbed/testbed.cc
 * @author: sailing-innocent
 * @create: 2022-11-20
 * @desp: The Testbed GL implementation
*/

#include <ing/testbed/testbed_gl.h>

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#undef min
#undef max
#undef near
#undef far

ING_NAMESPACE_BEGIN

void TestbedGL::init_window(int resw, int resh)
{

}

void TestbedGL::~TestbedGL()
{
    if (mRenderWindow) {
        destroy_window();
    }
}

void TestbedGL::init_window(int resw, int resh)
{
    // why not use my app directly?
}

bool TestbedGL::frame()
{
    if (mRenderWindow) {
        // begin frame
    }

    // update
    if (mRenderWindow) {
        if (mGuiRedraw) {
            // draw gui
        }
    }
}

ING_NAMESPACE_END
