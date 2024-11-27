#include "RenderTarget.h"

namespace Render
{
	RenderTarget::RenderTarget(
		HWND const hWnd,
		UINT const width,
		UINT const height,
		UINT const swapchainImageCount)
	{
		__pSwapchain = std::make_unique<Cuda::Swapchain>(hWnd, width, height, 3U);
		__resolveKernelBlockSize(width, height);
	}

	RenderTarget::~RenderTarget() noexcept
	{
		__pSwapchain = nullptr;
	}

	void RenderTarget::resize(
		UINT const width,
		UINT const height)
	{
		__pSwapchain->resize(width, height);
		__resolveKernelBlockSize(width, height);
	}

	void RenderTarget::draw(
		Kernel::EngineContext const &engineContext)
	{
		// draw
		Kernel::launch(
			engineContext, __renderTargetContext,
			__kernelGridSize, __kernelBlockSize);

		cudaDeviceSynchronize();
		__pSwapchain->present();
	}

	void RenderTarget::_onValidate()
	{
		// TODO
	}
}