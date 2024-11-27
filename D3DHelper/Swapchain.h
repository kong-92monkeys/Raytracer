#pragma once

#include <d3d11.h>
#include <dxgi1_6.h>
#include <memory>
#include <vector>

namespace D3D
{
	class Swapchain
	{
	public:
		Swapchain(
			HWND hWnd,
			UINT width,
			UINT height,
			UINT imageCount);

		virtual ~Swapchain() noexcept;

		void resize(
			UINT width,
			UINT height);

		[[nodiscard]]
		UINT getNextImageIndex() noexcept;

		[[nodiscard]]
		std::shared_ptr<ID3D11Texture2D> getImageOf(
			UINT const index);

		void present();

	private:
		ID3D11Device *__pDevice{ };
		ID3D11DeviceContext *__pContext{ };
		IDXGISwapChain3 *__pSwapChain{ };

		std::vector<ID3D11Texture2D *> __images;

		void __initSwapchain(
			HWND hWnd,
			UINT width,
			UINT height,
			UINT imageCount);

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