#pragma once

#include "../Infra/Stateful.h"
#include "../Cuda/Swapchain.h"
#include "Kernel.h"

namespace Render
{
	class RenderTarget : public Infra::Stateful<RenderTarget>
	{
	public:
		RenderTarget(
			HWND hWnd,
			UINT width,
			UINT height,
			UINT swapchainImageCount);

		virtual ~RenderTarget() noexcept override;

		void resize(
			UINT width,
			UINT height);

		void draw(
			Kernel::EngineContext const &engineContext);

	protected:
		virtual void _onValidate() override;

	private:
		std::unique_ptr<Cuda::Swapchain> __pSwapchain;

		dim3 __kernelGridSize;
		dim3 __kernelBlockSize;
		Kernel::RenderTargetContext __renderTargetContext;

		constexpr void __resolveKernelBlockSize(
			UINT width,
			UINT height) noexcept;
	};

	constexpr void RenderTarget::__resolveKernelBlockSize(
		UINT const width,
		UINT const height) noexcept
	{
		__kernelBlockSize.x = (width / __kernelGridSize.x);
		__kernelBlockSize.x += ((width % __kernelGridSize.x) ? 1 : 0);

		__kernelBlockSize.y = (height / __kernelGridSize.y);
		__kernelBlockSize.y += ((height % __kernelGridSize.y) ? 1 : 0);
	}
}