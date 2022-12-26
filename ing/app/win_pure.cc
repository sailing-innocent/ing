/**
 * @file: ing/app/win_pure.cc
 * @desp: The implementation of pure windows application
*/

#include <ing/app/win_pure.hpp>
#include "tchar.h"

LRESULT CALLBACK WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    switch(msg)
    {
        case WM_LBUTTONDOWN:
            MessageBox(0, _T("Hello World!"), _T("Hello"), MB_OK);
            return 0;
        case WM_KEYDOWN:
            if (wParam == VK_ESCAPE)
                DestroyWindow(hWnd);
            return 0;
        case WM_DESTROY:
            PostQuitMessage(0);
            return 0;
        case WM_PAINT:
            {
                PAINTSTRUCT ps;
                HDC hdc = BeginPaint(hWnd, &ps);

                // All painting occurs here, between BeginPaint and EndPaint.
                FillRect(hdc, &ps.rcPaint, (HBRUSH)GetStockObject(WHITE_BRUSH));
                EndPaint(hWnd, &ps);
                return 0;
            }

    }

    return DefWindowProc(hWnd, msg, wParam, lParam);
}


ING_NAMESPACE_BEGIN

bool PureWinApp::Init() {
    WNDCLASS wc;
    wc.style = CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc = WndProc;
    wc.cbClsExtra = 0;
    wc.cbWndExtra = 0;
    wc.hInstance = mInstanceHandle;
    wc.hIcon = LoadIcon(0, IDI_APPLICATION);
    wc.hCursor = LoadCursor(0, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)GetStockObject(WHITE_BRUSH);
    wc.lpszMenuName = 0;
    wc.lpszClassName = _T("BasicWndClass");

    if (!RegisterClass(&wc)) {
        MessageBox(0, _T("RegisterClass FAILED"), 0, 0);
        return false;
    }

    mGhMainWnd = CreateWindow(
        _T("BasicWndClass"),
        _T("Wnd32Basic"),
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT,
        CW_USEDEFAULT,
        CW_USEDEFAULT,
        CW_USEDEFAULT,
        0,
        0,
        mInstanceHandle,
        0
    );

    if (mGhMainWnd == 0) 
    {
        MessageBox(0, _T("CreateWindow FAILED"), 0, 0);
        return false;
    }

    ShowWindow(mGhMainWnd, mShow);
    UpdateWindow(mGhMainWnd);

    return true;
}

int PureWinApp::Run() {
    MSG msg = {0};

    BOOL bRet = 1;
    while ((bRet = GetMessage(&msg, 0, 0, 0)) != 0)
    {
        if (bRet == -1)
        {
            MessageBox(0, _T("GetMessage FAILED"), _T("Error"), MB_OK);
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

ING_NAMESPACE_END
