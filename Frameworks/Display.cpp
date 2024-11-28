#include "Display.h"

namespace Frx
{
	Display::Display(
		Infra::Executor &rcmdExecutor,
		Render::Engine &renderEngine,
		HWND const hWnd,
		UINT const width,
		UINT const height,
		UINT const swapchainImageCount) :
		__rcmdExecutor	{ rcmdExecutor },
		__renderEngine	{ renderEngine }
	{
		__rcmdExecutor.run([this, hWnd, width, height, swapchainImageCount]
		{
			__rcmd_pRenderTargetNeedRedrawListener =
				Infra::EventListener<Render::RenderTarget const *>::bind(
				&Display::__rcmd_onRenderTargetNeedRedraw, this);

			__pRenderTarget = std::unique_ptr<Render::RenderTarget>
			{
				__renderEngine.createRenderTarget(
					hWnd, width, height, swapchainImageCount)
			};

			__pRenderTarget->getNeedRedrawEvent() += __rcmd_pRenderTargetNeedRedrawListener;
		}).wait();
	}

	Display::~Display() noexcept
	{
		__rcmdExecutor.run([this]
		{
			__renderEngine.cancelRender(__pRenderTarget.get());
			__pRenderTarget = nullptr;
		}).wait();
	}

	void Display::resize(
		UINT const width,
		UINT const height)
	{
		__rcmdExecutor.run([this, width, height]
		{
			__pRenderTarget->resize(width, height);
		}).wait();

		__syncEvent.invoke(this);
	}

	void Display::requestRedraw() const
	{
		__rcmdExecutor.silentRun([this]
		{
			__pRenderTarget->requestRedraw();
		});
	}

	void Display::__rcmd_onRenderTargetNeedRedraw()
	{
		__renderEngine.reserveRender(__pRenderTarget.get());
	}
}