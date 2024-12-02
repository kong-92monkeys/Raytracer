#include "RenderSystem.h"
#include <new>

namespace Frx
{
	RenderSystem::RenderSystem()
	{
		__rcmdExecutor.run([this]
		{
			__rcmd_pIdleListener =
				Infra::EventListener<Infra::Looper *>::bind(
					&RenderSystem::__rcmd_onIdle, this);

			__createEngine();

			__rcmdExecutor.in_getIdleEvent() += __rcmd_pIdleListener;
		}).wait();
	}

	RenderSystem::~RenderSystem() noexcept
	{
		__rcmdExecutor.run([this]
		{
			__rcmd_pIdleListener = nullptr;
			__getRenderEngine().~Engine();
		}).wait();
	}

	Display *RenderSystem::createDisplay(
		HWND const hwnd,
		UINT const width,
		UINT const height,
		UINT const swapchainImageCount)
	{
		return new Display
		{
			__rcmdExecutor, __getRenderEngine(),
			hwnd, width, height, swapchainImageCount
		};
	}
	
	void RenderSystem::__createEngine()
	{
		new (__enginePlaceholder.data()) Render::Engine;
	}

	Render::Engine &RenderSystem::__getRenderEngine() noexcept
	{
		return *(reinterpret_cast<Render::Engine *>(__enginePlaceholder.data()));
	}

	void RenderSystem::__rcmd_onIdle()
	{
		__getRenderEngine().render();
	}
}