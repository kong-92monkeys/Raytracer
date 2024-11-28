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

		[[nodiscard]]
		constexpr UINT getWidth() const noexcept;

		[[nodiscard]]
		constexpr UINT getHeight() const noexcept;

		void resize(
			UINT width,
			UINT height);

		[[nodiscard]]
		UINT getBackSurfaceIndex() noexcept;

		[[nodiscard]]
		constexpr Surface &getSurfaceOf(
			UINT const index) noexcept;

		void present();

	private:
		std::unique_ptr<D3D::Swapchain> __pD3DSwapchain;
		std::vector<Surface *> __surfaces;

		void __createSurfaces();
		void __clearSurfaces();
	};

	constexpr UINT Swapchain::getWidth() const noexcept
	{
		return __pD3DSwapchain->getWidth();
	}

	constexpr UINT Swapchain::getHeight() const noexcept
	{
		return __pD3DSwapchain->getHeight();
	}

	constexpr Surface &Swapchain::getSurfaceOf(
		UINT const index) noexcept
	{
		return *(__surfaces[index]);
	}
}