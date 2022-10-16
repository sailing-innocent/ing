#include <windows.h>
#include <ing/app/win_pure.hpp>

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, PSTR pCmdLine, int nShowCmd)
{
    ing::PureWinApp app(hInstance, nShowCmd);
    if (!app.Init())
        return 0;
    return app.Run();
}
