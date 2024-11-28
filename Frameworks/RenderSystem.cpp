#include "RenderSystem.h"

namespace Frx
{
	RenderSystem::RenderSystem()
	{
		__rcmdExecutor.run([this]
		{
			__createEngine();
		}).wait();
	}

	RenderSystem::~RenderSystem() noexcept
	{
		__rcmdExecutor.run([this]
		{
			__getRenderEngine().~Engine();
		}).wait();
	}

	Display *RenderSystem::createDisplay(
		HWND const hWnd,
		UINT const width,
		UINT const height,
		UINT const swapchainImageCount)
	{
		return new Display
		{
			__rcmdExecutor, __getRenderEngine(),
			hWnd, width, height, swapchainImageCount
		};
	}

	void RenderSystem::__createEngine()
	{
		new (__enginePlaceholder.data()) Render::Engine{ };
	}

	Render::Engine &RenderSystem::__getRenderEngine() noexcept
	{
		return *(reinterpret_cast<Render::Engine *>(__enginePlaceholder.data()));
	}
}