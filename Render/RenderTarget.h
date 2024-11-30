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
		static constexpr UINT __SWAPCHAIN_IMAGE_COUNT{ 3U };

		ID3D11Device *const __pDevice;
		IDXGIFactory *const __pDXGIFactory;

		Cuda::Stream &__renderStream;
		UINT const __swapBufferCount;

		IDXGISwapChain *__pSwapchain{ };
		std::unique_ptr<Cuda::InteropSurface> __pBackSurface;

		bool __swapBufferCreated{ };
		std::vector<Cuda::ArraySurface *> __swapBuffers;

		UINT __width{ };
		UINT __height{ };

		std::vector<std::shared_ptr<Cuda::Event>> __launchEvents;

		KernelLauncher __kernelLauncher;

		mutable Infra::Event<RenderTarget const *> __needRedrawEvent;

		void __createSwapchain(
			HWND hwnd);

		void __resolveBackSurface();

		void __clearSwapBuffers();
		void __createSwapBuffers();
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