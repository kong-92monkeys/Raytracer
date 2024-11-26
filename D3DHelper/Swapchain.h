#pragma once

#include <d3d11.h>

namespace D3D
{
	class Swapchain
	{
	public:
		Swapchain(
			HWND const hWnd,
			UINT const width,
			UINT const height);

		virtual ~Swapchain() noexcept;

		void resize(
			UINT const width,
			UINT const height);

	private:
		ID3D11Device *__pDevice{ };
		ID3D11DeviceContext *__pContext{ };
		IDXGISwapChain *__pSwapChain{ };
	};
}