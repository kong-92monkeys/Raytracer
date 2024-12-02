#pragma once

#include <d3d11.h>
#include <dxgi.h>
#include "../Infra/Stateful.h"
#include "../Cuda/Stream.h"
#include "../Cuda/Surface.h"
#include "KernelLauncher.h"

namespace Render
{
	class RenderTarget : public Infra::Stateful<RenderTarget>
	{
	public:
		RenderTarget(
			ID3D11Device *pDevice,
			IDXGIFactory *pDXGIFactory,
			Cuda::Stream &renderStream,
			HWND hwnd,
			UINT width,
			UINT height,
			UINT swapBufferCount);

		virtual ~RenderTarget() noexcept override;

		[[nodiscard]]
		constexpr UINT getWidth() const noexcept;

		[[nodiscard]]
		constexpr UINT getHeight() const noexcept;

		[[nodiscard]]
		constexpr bool isPresentable() const noexcept;

		void setViewport(
			Kernel::Viewport const &viewport) noexcept;

		void resize(
			UINT width,
			UINT height);

		void draw();
		void present();

		[[nodiscard]]
		constexpr Infra::Event<RenderTarget *> &getResizeEvent() noexcept;

	protected:
		virtual void _onValidate() override;

	private:
		static constexpr UINT __SWAPCHAIN_IMAGE_COUNT{ 3U };

		ID3D11Device *const __pDevice;
		IDXGIFactory *const __pDXGIFactory;

		Cuda::Stream &__renderStream;
		UINT const __swapBufferCount;

		IDXGISwapChain *__pSwapchain{ };
		std::unique_ptr<Cuda::InteropSurface> __pSwapchainSurface;

		bool __swapBufferCreated{ };
		std::vector<Cuda::ArraySurface *> __swapBuffers;

		UINT __frontBufferIdx{ };
		UINT __backBufferIdx{ };

		UINT __width{ };
		UINT __height{ };

		KernelLauncher __kernelLauncher;
		std::vector<std::shared_ptr<Cuda::Event>> __launchEvents;

		Infra::Event<RenderTarget *> __resizeEvent;

		void __createSwapchain(
			HWND hwnd);

		void __resolveBackSurface();

		void __clearSwapBuffers();
		void __createSwapBuffers();

		[[nodiscard]]
		constexpr UINT __getNextFrontBufferIdx() const noexcept;

		[[nodiscard]]
		constexpr UINT __getNextBackBufferIdx() const noexcept;

		[[nodiscard]]
		constexpr bool __isSwapBufferEmpty() const noexcept;

		[[nodiscard]]
		constexpr bool __isSwapBufferFull() const noexcept;

		constexpr void __rotateFrontBuffer() noexcept;
		constexpr void __rotateBackBuffer() noexcept;
	};

	constexpr UINT RenderTarget::getWidth() const noexcept
	{
		return __width;
	}

	constexpr UINT RenderTarget::getHeight() const noexcept
	{
		return __height;
	}

	constexpr bool RenderTarget::isPresentable() const noexcept
	{
		return (getWidth() && getHeight());
	}

	constexpr Infra::Event<RenderTarget *> &RenderTarget::getResizeEvent() noexcept
	{
		return __resizeEvent;
	}

	constexpr UINT RenderTarget::__getNextFrontBufferIdx() const noexcept
	{
		return ((__frontBufferIdx + 1U) % __swapBufferCount);
	}

	constexpr UINT RenderTarget::__getNextBackBufferIdx() const noexcept
	{
		return ((__backBufferIdx + 1U) % __swapBufferCount);
	}

	constexpr bool RenderTarget::__isSwapBufferEmpty() const noexcept
	{
		return (__frontBufferIdx == __backBufferIdx);
	}

	constexpr bool RenderTarget::__isSwapBufferFull() const noexcept
	{
		return (__frontBufferIdx == __getNextBackBufferIdx());
	}

	constexpr void RenderTarget::__rotateFrontBuffer() noexcept
	{
		__frontBufferIdx = __getNextFrontBufferIdx();
	}

	constexpr void RenderTarget::__rotateBackBuffer() noexcept
	{
		__backBufferIdx = __getNextBackBufferIdx();
	}
}