#pragma once

#include <d3d11.h>
#include <dxgi1_6.h>
#include <memory>

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

		[[nodiscard]]
		UINT getNextImageIndex() noexcept;

		[[nodiscard]]
		std::shared_ptr<ID3D11Texture2D> acquireImageOf(
			UINT const index);

		void present();

	private:
		ID3D11Device *__pDevice{ };
		ID3D11DeviceContext *__pContext{ };
		IDXGISwapChain3 *__pSwapChain{ };

		template <typename $T>
		static void __customDeleter(
			$T *pResource);
	};

	template <typename $T>
	static void Swapchain::__customDeleter(
		$T *const pResource)
	{
		pResource->Release();
	}
}