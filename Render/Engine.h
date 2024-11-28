#pragma once

#include "RenderTarget.h"
#include <unordered_set>

namespace Render
{
	class Engine : public Infra::Unique
	{
	public:
		Engine();
		virtual ~Engine() noexcept override;

		[[nodiscard]]
		RenderTarget *createRenderTarget(
			HWND hWnd,
			UINT width,
			UINT height,
			UINT swapchainImageCount);

		void reserveRender(
			RenderTarget *pRenderTarget) noexcept;

		void cancelRender(
			RenderTarget *pRenderTarget) noexcept;

		void render();

	private:
		std::unique_ptr<Cuda::Stream> __pRenderStream;
		std::unordered_set<RenderTarget *> __reservedRenderTargets;

		void __validateReservedRenderTargets();
	};
}