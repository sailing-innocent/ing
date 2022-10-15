#include <windows.h>

HWND ghMainWnd = 0;

bool InitWindowsApp(HINSTANCE instanceHandle, int show);

int Run();

LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, PSTR pCmdLine, int nShowCmd)
{
    if (!InitWindowsApp(hInstance, nShowCmd))
        return 0;
    return Run();
}

bool InitWindowsApp(HINSTANCE instanceHandle, int show)
{
    WNDCLASS wc;
    wc.style = CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc = WndProc;
    wc.cbClsExtra = 0;
    wc.cbWndExtra = 0;
    wc.hInstance = instanceHandle;
    wc.hIcon = LoadIcon(0, IDI_APPLICATION);
    wc.hCursor = LoadCursor(0, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)GetStockObject(WHITE_BRUSH);
    wc.lpszMenuName = 0;
    wc.lpszClassName = L"BasicWndClass";

    if (!RegisterClass(&wc)) {
        MessageBox(0, L"RegisterClass FAILED", 0, 0);
        return false;
    }

    ghMainWnd = CreateWindow(
        L"BasicWndClass",
        L"Wnd32Basic",
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT,
        CW_USEDEFAULT,
        CW_USEDEFAULT,
        CW_USEDEFAULT,
        0,
        0,
        instanceHandle,
        0
    );

    if (ghMainWnd == 0) 
    {
        MessageBox(0, L"CreateWindow FAILED", 0, 0);
        return false;
    }

    ShowWindow(ghMainWnd, show);
    UpdateWindow(ghMainWnd);

    return true;
}

int Run()
{
    MSG msg = {0};

    BOOL bRet = 1;
    while ((bRet = GetMessage(&msg, 0, 0, 0)) != 0)
    {
        if (bRet == -1)
        {
            MessageBox(0, L"GetMessage FAILED", L"Error", MB_OK);
            break;
        }
        else
        {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
    }

    return (int)msg.wParam;
}

LRESULT CALLBACK WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    switch(msg)
    {
        case WM_KEYDOWN:
            if (wParam == VK_ESCAPE)
                DestroyWindow(hWnd);
            return 0;
        case WM_DESTROY:
            PostQuitMessage(0);
            return 0;
    }

    return DefWindowProc(hWnd, msg, wParam, lParam);
}