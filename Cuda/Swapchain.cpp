#include "Swapchain.h"

namespace Cuda
{
	Swapchain::Swapchain(
		HWND const hWnd,
		UINT const width,
		UINT const height,
		UINT const imageCount)
	{
		__pD3DSwapchain = std::make_unique<D3D::Swapchain>(hWnd, width, height, imageCount);
		__surfaces.resize(imageCount);
	}

	Swapchain::~Swapchain() noexcept
	{
		__clearSurfaces();
		__pD3DSwapchain = nullptr;
	}

	void Swapchain::resize(
		UINT const width,
		UINT const height)
	{
		__clearSurfaces();
		__pD3DSwapchain->resize(width, height);
	}

	Surface &Swapchain::getNextImage()
	{
		UINT const nextImageIdx{ __pD3DSwapchain->getNextImageIndex() };

		auto &pRetVal{ __surfaces[nextImageIdx] };
		if (!pRetVal)
		{
			auto const pImage{ __pD3DSwapchain->getImageOf(nextImageIdx) };

			int flags{ };
			flags |= cudaGraphicsRegisterFlags::cudaGraphicsRegisterFlagsWriteDiscard;
			flags |= cudaGraphicsRegisterFlags::cudaGraphicsRegisterFlagsSurfaceLoadStore;

			pRetVal = new Surface{ pImage.get(), static_cast<cudaGraphicsRegisterFlags>(flags) };
		}

		return *pRetVal;
	}

	void Swapchain::present()
	{
		__pD3DSwapchain->present();
	}

	void Swapchain::__clearSurfaces()
	{
		for (auto const pSurface : __surfaces)
			delete pSurface;

		ZeroMemory(__surfaces.data(), __surfaces.size() * sizeof(Surface *));
	}
}