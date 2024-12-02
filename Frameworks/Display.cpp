#include "Display.h"

namespace Frx
{
	Display::Display(
		Infra::Executor &rcmdExecutor,
		Render::Engine &renderEngine,
		HWND const hwnd,
		UINT const width,
		UINT const height,
		UINT const swapchainImageCount) :
		__rcmdExecutor	{ rcmdExecutor },
		__renderEngine	{ renderEngine }
	{
		__rcmdExecutor.run([this, hwnd, width, height, swapchainImageCount]
		{
			__rcmd_pRenderTarget = std::unique_ptr<Render::RenderTarget>
			{
				__renderEngine.createRenderTarget(
					hwnd, width, height, swapchainImageCount)
			};
		}).wait();

		__width		= width;
		__height	= height;
	}

	Display::~Display() noexcept
	{
		__rcmdExecutor.run([this]
		{
			__renderEngine.cancelRender(__rcmd_pRenderTarget.get());
			__rcmd_pRenderTarget = nullptr;
		}).wait();
	}

	void Display::setViewport(
		Render::Kernel::Viewport const &viewport) noexcept
	{
		__rcmdExecutor.silentRun([this, viewport]
		{
			__rcmd_pRenderTarget->setViewport(viewport);
		});
	}

	void Display::resize(
		UINT const width,
		UINT const height)
	{
		__rcmdExecutor.run([this, width, height]
		{
			__rcmd_pRenderTarget->resize(width, height);
		}).wait();
		
		__width		= width;
		__height	= height;
		__resizeEvent.invoke(this);
	}

	void Display::requestRedraw()
	{
		__rcmdExecutor.silentRun([this]
		{
			__renderEngine.reserveRender(__rcmd_pRenderTarget.get());
		});
	}
}