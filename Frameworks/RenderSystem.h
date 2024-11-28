#pragma once

#include "../Infra/SingleThreadPool.h"
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
			HWND hWnd,
			UINT width,
			UINT height,
			UINT swapchainImageCount);

	private:
		Infra::SingleThreadPool __rcmdExecutor;

		std::array<std::byte, sizeof(Render::Engine)> __enginePlaceholder{ };

		void __createEngine();

		[[nodiscard]]
		Render::Engine &__getRenderEngine() noexcept;
	};
}