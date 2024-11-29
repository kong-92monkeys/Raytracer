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
			HWND hWnd,
			UINT width,
			UINT height,
			UINT swapchainImageCount);

		virtual ~RenderTarget() noexcept override;

		[[nodiscard]]
		constexpr UINT getWidth() const noexcept;

		[[nodiscard]]
		constexpr UINT getHeight() const noexcept;

		[[nodiscard]]
		constexpr bool isPresentable() const noexcept;

		void resize(
			UINT width,
			UINT height);

		void draw();
		void present();

		[[nodiscard]]
		constexpr Infra::Event<RenderTarget const *> &getNeedRedrawEvent() const noexcept;

	protected:
		virtual void _onValidate() override;

	private:
		ID3D11Device *const __pDevice;
		IDXGIFactory *const __pDXGIFactory;

		Cuda::Stream &__renderStream;

		IDXGISwapChain *__pSwapchain{ };
		ID3D11Texture2D *__pBackBuffer{ };

		std::unique_ptr<Cuda::Surface> __pBackSurface;

		UINT __width{ };
		UINT __height{ };

		std::vector<std::shared_ptr<Cuda::Event>> __launchEvents;

		KernelLauncher __kernelLauncher;

		mutable Infra::Event<RenderTarget const *> __needRedrawEvent;

		void __createSwapchain(
			HWND hWnd,
			UINT width,
			UINT height,
			UINT imageCount);

		void __resolveBackBuffer();
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

	constexpr Infra::Event<RenderTarget const *> &RenderTarget::getNeedRedrawEvent() const noexcept
	{
		return __needRedrawEvent;
	}
}