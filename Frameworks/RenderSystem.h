#pragma once

#include "../Infra/Looper.h"
#include "Display.h"
#include <array>
#include <cstddef>

namespace Frx
{
	class RenderSystem : public Infra::Unique
	{
	public:
		RenderSystem();
		virtual ~RenderSystem() noexcept override;

		[[nodiscard]]
		Display *createDisplay(
			HWND hwnd,
			UINT width,
			UINT height,
			UINT swapchainImageCount);

	private:
		Infra::Looper __rcmdExecutor;

		std::array<std::byte, sizeof(Render::Engine)> __enginePlaceholder{ };

		Infra::EventListenerPtr<Infra::Looper *> __rcmd_pIdleListener;

		void __createEngine();

		[[nodiscard]]
		Render::Engine &__getRenderEngine() noexcept;

		void __rcmd_onIdle();
	};
}