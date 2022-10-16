#pragma once
/**
 * @file: include/app/win_drx12.hpp
 * @author: sailing-innocent
*/

#ifndef ING_APP_WIN_DRX12_H_
#define ING_APP_WIN_DRX12_H_

#include <ing/common.h>

#include <utils/d3dapp_utils.hpp>
#include <utils/drx/timer.hpp>

ING_NAMESPACE_BEGIN

class D3DApp
{
protected:
    D3DApp(HINSTANCE hInstance);
    D3DApp(const D3DApp& rhs) = delete;
    D3DApp& operator=(const D3DApp& rhs) = delete;
    virtual ~D3DApp();
public:
    static D3DApp* GetApp();
    HINSTANCE AppInst() const;
    HWND MainWnd() const;
    float AspectRatio() const;
    bool Get4xMsaaState() const;
    void Set4xMsaaState(bool value);

    int Run();

    virtual bool Initialize();
    virtual LRESULT MsgProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);

protected:
    virtual void CreateRtvAndDsvDescriptorHeaps();
    virtual void OnResize();
    virtual void Update(const Timer& timer) = 0;
    virtual void Draw(const Timer& timer) = 0;

    // Convience overrides for handling mouse input
    virtual void OnMouseDown(WPARAM btnState, int x, int y) { }
    virtual void OnMouseUp(WPARAM btnState, int x, int y) { }
    virtual void OnMouseMove(WPARAM btnState, int x, int y) { }

protected:
    bool InitMainWindow();
    bool InitDirect3D();
    void CreateCommandObjects();
    void CreateSwapChain();
    void FlushCommandQueue();

    ID3D12Resource* CurrentBackBuffer() const;

    D3D12_CPU_DESCRIPTOR_HANDLE CurrentBackBufferView() const;
    
    D3D12_CPU_DESCRIPTOR_HANDLE DepthStencilView() const;

    void CalculateFrameStats();

    void LogAdapters();
    void LogAdapterOutputs(IDXGIAdapter* adapter);
    void LogOutputDisplayModes(IDXGIOutput* output, DXGI_FORMAT format);

protected:
    static D3DApp* mApp;
    HINSTANCE mhAppInst = nullptr; // application insatance handle
    HWND mhMainWnd = nullptr; //  main window handle
    bool mAppPaused = false; // is application paused?
    bool mMinimized = false; // is application minimized?
    bool mMaximized = false; // is application maximzied?
    bool mResizing = false; // are the resize bars being dragged?
    bool mFullscreenState = false; // fullscreen enabled

    bool m4xMsaaState = false; // true if use 4x Multisampling
    UINT m4xMsaaQuality = 0; // quality level of 4X Multisampling

    Timer mTimer;

    Microsoft::WRL::ComPtr<IDXGIFactory4> mdxgiFactory;
    Microsoft::WRL::ComPtr<IDXGISwapChain> mSwapChain;
    Microsoft::WRL::ComPtr<ID3D12Device> md3dDevice;

    Microsoft::WRL::ComPtr<ID3D12Fence> mFence;
    UINT64 mCurrentFence = 0;

    Microsoft::WRL::ComPtr<ID3D12CommandQueue> mCommandQueue;
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> mDirectCmdListAlloc;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> mCommandList;

    static const int SwapChainBufferCount = 2;

    int mCurrBackBuffer = 0;
    Microsoft::WRL::ComPtr<ID3D12Resource> mSwapChainBuffer[SwapChainBufferCount];
    Microsoft::WRL::ComPtr<ID3D12Resource> mDepthStencilBuffer;

    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> mRtvHeap;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> mDsvHeap;

    D3D12_VIEWPORT mScreenViewport;
    D3D12_RECT mScissorRect;
    
    UINT mRtvDescriptorSize = 0;
    UINT mDsvDescriptorSize = 0;
    UINT mCbvSrvUavDescriptorSize = 0;

    std::wstring mMainWndCaption = L"testbed";
    D3D_DRIVER_TYPE md3dDriverType = D3D_DRIVER_TYPE_HARDWARE;
    DXGI_FORMAT mBackBufferFormat = DXGI_FORMAT_R8G8B8A8_UNORM;
    DXGI_FORMAT mDepthStencilFormat = DXGI_FORMAT_D24_UNORM_S8_UINT;

    int mClientWidth = 800;
    int mClientHeight = 600;
};


ING_NAMESPACE_END

#endif // ING_APP_WIN_DRX12_H_
