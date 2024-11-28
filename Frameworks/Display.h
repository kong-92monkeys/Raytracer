#pragma once

#include "../Infra/Executor.h"
#include "../Render/Engine.h"

namespace Frx
{
	class Display : public Infra::Unique
	{
	public:
		Display(
			Infra::Executor &rcmdExecutor,
			Render::Engine &renderEngine,
			HWND hWnd,
			UINT width,
			UINT height,
			UINT swapchainImageCount);

		virtual ~Display() noexcept override;

		void resize(
			UINT width,
			UINT height);

		void requestRedraw() const;

		[[nodiscard]]
		constexpr Render::RenderTarget &rcmd_getRenderTarget() noexcept;

		[[nodiscard]]
		constexpr Infra::EventView<Display const *> &getSyncEvent() const noexcept;

	private:
		Infra::Executor &__rcmdExecutor;
		Render::Engine &__renderEngine;

		std::unique_ptr<Render::RenderTarget> __pRenderTarget;

		Infra::EventListenerPtr<Render::RenderTarget const *> __rcmd_pRenderTargetNeedRedrawListener;

		mutable Infra::Event<Display const *> __syncEvent;

		void __rcmd_onRenderTargetNeedRedraw();
	};

	constexpr Render::RenderTarget &Display::rcmd_getRenderTarget() noexcept
	{
		return *__pRenderTarget;
	}

	constexpr Infra::EventView<Display const *> &Display::getSyncEvent() const noexcept
	{
		return __syncEvent;
	}
}