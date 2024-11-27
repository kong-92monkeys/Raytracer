#pragma once

#include "../D3DHelper/Swapchain.h"
#include "Surface.h"

namespace Cuda
{
	class Swapchain : public Infra::Unique
	{
	public:
		Swapchain(
			HWND hWnd,
			UINT width,
			UINT height,
			UINT imageCount);

		virtual ~Swapchain() noexcept;

		void resize(
			UINT width,
			UINT height);

		[[nodiscard]]
		Surface &getNextImage();

		void present();

	private:
		std::unique_ptr<D3D::Swapchain> __pD3DSwapchain;
		std::vector<Surface *> __surfaces;

		void __clearSurfaces();
	};
}