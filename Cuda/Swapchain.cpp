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
		__createSurfaces();
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
		__createSurfaces();
	}

	UINT Swapchain::getBackSurfaceIndex() noexcept
	{
		return __pD3DSwapchain->getBackBufferIndex();
	}

	void Swapchain::present()
	{
		__pD3DSwapchain->present();
	}

	void Swapchain::__createSurfaces()
	{
		auto const flags{ cudaGraphicsRegisterFlags::cudaGraphicsRegisterFlagsSurfaceLoadStore };

		UINT const surfaceCount{ static_cast<UINT>(__surfaces.size()) };
		for (UINT idx{ }; idx < surfaceCount; ++idx)
		{
			auto const pBuffer{ __pD3DSwapchain->getBufferOf(idx) };
			__surfaces[idx] = new Surface{ pBuffer, flags };
		}
	}

	void Swapchain::__clearSurfaces()
	{
		for (auto const pSurface : __surfaces)
			delete pSurface;

		ZeroMemory(__surfaces.data(), __surfaces.size() * sizeof(Surface *));
	}
}