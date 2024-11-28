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
		__pSwapchain = std::make_unique<Cuda::Swapchain>(hWnd, width, height, 3U);

		__kernelLauncher.setStream(renderStream.getHandle());
		__kernelLauncher.setGridSize(16U, 16U);
		__kernelLauncher.setSurfaceExtent(width, height);
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
		__kernelLauncher.setSurfaceExtent(width, height);
	}

	void RenderTarget::requestRedraw() const
	{
		__needRedrawEvent.invoke(this);
	}

	void RenderTarget::draw()
	{
		UINT const surfaceIdx{ __pSwapchain->getBackSurfaceIndex() };

		auto &surface{ __pSwapchain->getSurfaceOf(surfaceIdx) };
		surface.map();

		__kernelLauncher.launch(surface.getHandle());
	}

	void RenderTarget::present()
	{
		__pSwapchain->present();
	}

	void RenderTarget::_onValidate()
	{
		// TODO
	}
}