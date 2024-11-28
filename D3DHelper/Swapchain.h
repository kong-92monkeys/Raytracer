#pragma once

#include "../Infra/Unique.h"
#include <d3d11.h>
#include <dxgi1_6.h>
#include <memory>
#include <vector>

namespace D3D
{
	class Swapchain : public Infra::Unique
	{
	public:
		Swapchain(
			HWND hWnd,
			UINT width,
			UINT height,
			UINT imageCount);

		virtual ~Swapchain() noexcept override;

		[[nodiscard]]
		constexpr UINT getWidth() const noexcept;

		[[nodiscard]]
		constexpr UINT getHeight() const noexcept;

		[[nodiscard]]
		constexpr UINT getImageCount() const noexcept;

		void resize(
			UINT width,
			UINT height);

		[[nodiscard]]
		UINT getBackBufferIndex() noexcept;

		[[nodiscard]]
		constexpr ID3D11Texture2D *getBufferOf(
			UINT const index) noexcept;

		void present();

	private:
		UINT const __imageCount;

		ID3D11Device *__pDevice{ };
		ID3D11DeviceContext *__pContext{ };
		IDXGISwapChain3 *__pSwapchain{ };

		std::vector<ID3D11Texture2D *> __buffers;
		
		UINT __width{ };
		UINT __height{ };

		void __initSwapchain(
			HWND hWnd,
			UINT width,
			UINT height,
			UINT imageCount);

		void __resolveBuffers();
		void __clearBuffers();
	};

	constexpr UINT Swapchain::getWidth() const noexcept
	{
		return __width;
	}

	constexpr UINT Swapchain::getHeight() const noexcept
	{
		return __height;
	}

	constexpr UINT Swapchain::getImageCount() const noexcept
	{
		return __imageCount;
	}

	constexpr ID3D11Texture2D *Swapchain::getBufferOf(
		UINT const index) noexcept
	{
		return __buffers[index];
	}
}