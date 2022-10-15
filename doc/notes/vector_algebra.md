# Chapter 1: Vector Algebra

Vectors play a crucial role in computer graphics, collition detection, and physical simulation.

## 1.1 Vectors
## 1.2 Length and Unit Vectors
## 1.3 Dot Product
## 1.4 Cross Product
## 1.5 Points
## 1.6 DirectX Math Vectors

For Windows 8 and above, DirectXMath is a 3D math library for Direct3D application that is a part of Windows SDK. The library uses `SSE2` (Streaming SIMD Extension 2) instruction set. With 128-bit wide SIMD (single instruction multiple data) registers. SIMD instructions can operate on four 32-bit `float`s or `int`s with one instruction.

To use DirectX Math Library:
- `#include <DirectXMath.h>`
- `#include <DirectXPackedVector.h>`
- for x86 platform, you should enable SSE2: http://en.wikipedia.org/wiki/SSE2

### 1.6.1 Vector Types

`typedef __m128 XMVECTOR;`

```cpp
struct XMFLOAT2
{
    float x;
    float y;

    XMFLOAT2() {}
    XMFLOAT2(float _x, float _y): x(_x), y(_y) {}
    explicit XMFLOAT(_In_reads_(2) const float *pArray): x(pArray[0], y(pArray[1]) {}
    XMFLOAT2& operator=(const XMFLOAT2& Float2) {
        x = Float2.x; y = Float2.y; return *this;
    }
}
```

similiarly `XMFLOAT3` `XMFLOAT4`. However, if we use these types directly for calculations, then we will not take advantage of SIMD. Thus we need to convert these instances to `XMVECTOR` type

### Loading and Storage Methods
```cpp
XMVECTOR XM_CALLCONV XMLoadFloat2(const XMFLOAT2 *pSource);

void XM_CALLCONV XMStoreFloat2(XMFLOAT2 *pDestination, FXMVECTOR V);

float XM_CALLCONV XMVectorGetX(FXMVECTOR V);

```

Can refer to DirectX Math Documentation, by Calling Conventions 