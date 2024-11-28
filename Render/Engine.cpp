#include "Engine.h"

namespace Render
{
	Engine::Engine()
	{
		__pRenderStream = std::make_unique<Cuda::Stream>();
	}

	Engine::~Engine() noexcept
	{
		__pRenderStream = nullptr;
	}

	RenderTarget *Engine::createRenderTarget(
		HWND const hWnd,
		UINT const width,
		UINT const height,
		UINT const swapchainImageCount)
	{
		return new RenderTarget
		{
			*__pRenderStream, hWnd,
			width, height, swapchainImageCount
		};
	}

	void Engine::reserveRender(
		RenderTarget *const pRenderTarget) noexcept
	{
		__reservedRenderTargets.emplace(pRenderTarget);
	}

	void Engine::cancelRender(
		RenderTarget *const pRenderTarget) noexcept
	{
		__reservedRenderTargets.erase(pRenderTarget);
	}

	void Engine::render()
	{
		if (__reservedRenderTargets.empty())
			return;

		//__pGlobalDescriptorManager->validate();
		__validateReservedRenderTargets();

		//__pDescriptorUpdater->update();

		for (auto const pRenderTarget : __reservedRenderTargets)
			pRenderTarget->draw();

		//auto &submissionFence{ __getNextSubmissionFence() };
		//__pDevice->vkResetFences(1U, &(submissionFence.getHandle()));
		//__pCommandSubmitter->submit(submissionFence);

		//__pCommandSubmitter->present();

		//__deferredDeleter.advance();
		__reservedRenderTargets.clear();
	}

	void Engine::__validateReservedRenderTargets()
	{
		for (auto it{ __reservedRenderTargets.begin() }; it != __reservedRenderTargets.end(); )
		{
			auto const pRenderTarget{ *it };

			if (pRenderTarget->isPresentable())
			{
				pRenderTarget->validate();
				++it;
			}
			else
				it = __reservedRenderTargets.erase(it);
		}
	}
}