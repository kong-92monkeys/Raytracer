#include "RenderTarget.h"

namespace Render
{
	RenderTarget::RenderTarget(
		Cuda::Stream &renderStream,
		HWND const hWnd,
		UINT const width,
		UINT const height,
		UINT const swapchainImageCount) :
		__renderStream{ renderStream }
	{
		__pSwapchain = std::make_unique<Cuda::Swapchain>(hWnd, width, height, swapchainImageCount);

		for (UINT iter{ }; iter < swapchainImageCount; ++iter)
			__launchEvents.emplace_back(std::make_shared<Cuda::Event>());

		__kernelLauncher.setStream(renderStream.getHandle());
		__kernelLauncher.setGridSize(16U, 16U);
		__kernelLauncher.setSurfaceExtent(width, height);
	}

	RenderTarget::~RenderTarget() noexcept
	{
		__launchEvents.clear();
		__pSwapchain = nullptr;
	}

	void RenderTarget::resize(
		UINT const width,
		UINT const height)
	{
		__pSwapchain->resize(width, height);
		__kernelLauncher.setSurfaceExtent(width, height);
	}

	void RenderTarget::draw()
	{
		UINT const backSurfIdx{ __pSwapchain->getBackSurfaceIndex() };

		auto &backSurface{ __pSwapchain->getSurfaceOf(backSurfIdx) };
		backSurface.map();

		__kernelLauncher.launch(backSurface.getHandle());
		__renderStream.recordEvent(*(__launchEvents[backSurfIdx]));
	}

	void RenderTarget::present()
	{
		UINT const nextFrontSurfIdx{ __pSwapchain->getNextFrontIndex() };

		auto &nextFrontSurface{ __pSwapchain->getSurfaceOf(nextFrontSurfIdx) };
		nextFrontSurface.unmap();

		__renderStream.syncEvent(*(__launchEvents[nextFrontSurfIdx]));
		__pSwapchain->present();
	}

	void RenderTarget::_onValidate()
	{
		// TODO
	}
}