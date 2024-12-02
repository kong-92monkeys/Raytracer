#pragma once

#include "../Render/Engine.h"
#include "../Infra/Executor.h"

namespace Frx
{
	class Display : public Infra::Unique
	{
	public:
		Display(
			Infra::Executor &rcmdExecutor,
			Render::Engine &renderEngine,
			HWND hwnd,
			UINT width,
			UINT height,
			UINT swapchainImageCount);

		virtual ~Display() noexcept override;

		[[nodiscard]]
		constexpr UINT getWidth() const noexcept;

		[[nodiscard]]
		constexpr UINT getHeight() const noexcept;

		[[nodiscard]]
		constexpr bool isPresentable() const noexcept;

		void setViewport(
			Render::Kernel::Viewport const &viewport) noexcept;

		void resize(
			UINT width,
			UINT height);

		void requestRedraw();

		[[nodiscard]]
		constexpr Infra::EventView<Display *> &getResizeEvent() noexcept;

	private:
		Infra::Executor &__rcmdExecutor;
		Render::Engine &__renderEngine;

		std::unique_ptr<Render::RenderTarget> __rcmd_pRenderTarget;

		UINT __width{ };
		UINT __height{ };

		Infra::Event<Display *> __resizeEvent;
	};

	constexpr UINT Display::getWidth() const noexcept
	{
		return __width;
	}

	constexpr UINT Display::getHeight() const noexcept
	{
		return __height;
	}

	constexpr bool Display::isPresentable() const noexcept
	{
		return (__width && __height);
	}

	constexpr Infra::EventView<Display *> &Display::getResizeEvent() noexcept
	{
		return __resizeEvent;
	}
}