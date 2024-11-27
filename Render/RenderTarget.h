#pragma once

#include "../Infra/Stateful.h"
#include "../Cuda/Swapchain.h"

namespace Render
{
	class RenderTarget : public Infra::Stateful<RenderTarget>
	{
	public:
		RenderTarget(
			HWND hWnd,
			UINT width,
			UINT height,
			UINT swapchainImageCount);

		void sync();

	protected:
		virtual void _onValidate() override;

	};
}