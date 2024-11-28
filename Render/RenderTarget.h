#pragma once

#include "../Infra/Stateful.h"
#include "../Cuda/Stream.h"
#include "../Cuda/Swapchain.h"
#include "KernelLauncher.h"

namespace Render
{
	class RenderTarget : public Infra::Stateful<RenderTarget>
	{
	public:
		RenderTarget(
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
		Cuda::Stream &__renderStream;

		std::unique_ptr<Cuda::Swapchain> __pSwapchain;
		KernelLauncher __kernelLauncher;

		mutable Infra::Event<RenderTarget const *> __needRedrawEvent;
	};

	constexpr UINT RenderTarget::getWidth() const noexcept
	{
		return __pSwapchain->getWidth();
	}

	constexpr UINT RenderTarget::getHeight() const noexcept
	{
		return __pSwapchain->getHeight();
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