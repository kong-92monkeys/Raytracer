#include "RenderTarget.h"

namespace Render
{
	RenderTarget::RenderTarget(
		ID3D11Device *const pDevice,
		IDXGIFactory *const pDXGIFactory,
		Cuda::Stream &renderStream,
		HWND const hWnd,
		UINT const width,
		UINT const height,
		UINT const swapchainImageCount) :
		__pDevice		{ pDevice },
		__pDXGIFactory	{ pDXGIFactory },
		__renderStream	{ renderStream }
	{
		__createSwapchain(hWnd, width, height, swapchainImageCount);
		__resolveBackBuffer();

		__width		= width;
		__height	= height;

		for (UINT iter{ }; iter < swapchainImageCount; ++iter)
			__launchEvents.emplace_back(std::make_shared<Cuda::Event>());

		__kernelLauncher.setStream(renderStream.getHandle());
		__kernelLauncher.setGridSize(16U, 16U);
		__kernelLauncher.setSurfaceExtent(width, height);
	}

	RenderTarget::~RenderTarget() noexcept
	{
		__pBackSurface = nullptr;

		__pSwapchain->Release();
		__launchEvents.clear();
	}

	void RenderTarget::resize(
		UINT const width,
		UINT const height)
	{
		__pBackSurface = nullptr;

		HRESULT result{ };
		result = __pSwapchain->ResizeBuffers(4U, width, height, DXGI_FORMAT_R8G8B8A8_UNORM, 0U);

		if (FAILED(result))
			throw std::runtime_error{ "Cannot resize the swapchain." };

		__resolveBackBuffer();

		__width		= width;
		__height	= height;

		__kernelLauncher.setSurfaceExtent(width, height);
	}

	void RenderTarget::draw()
	{
		//UINT const backSurfIdx{ __pSwapchain->getBackSurfaceIndex() };

		//auto &backSurface{ __pSwapchain->getSurfaceOf(backSurfIdx) };
		__pBackSurface->map();
		__kernelLauncher.launch(__pBackSurface->getHandle());

		cudaDeviceSynchronize();
		__pBackSurface->unmap();
		//__renderStream.recordEvent(*(__launchEvents[backSurfIdx]));
	}

	void RenderTarget::present()
	{
		//UINT const nextFrontSurfIdx{ __pSwapchain->getNextFrontIndex() };

		//auto &nextFrontSurface{ __pSwapchain->getSurfaceOf(nextFrontSurfIdx) };

		//__renderStream.syncEvent(*(__launchEvents[nextFrontSurfIdx]));
		HRESULT result{ };
		result = __pSwapchain->Present(0U, 0U);

		if (FAILED(result))
			throw std::runtime_error{ "Failed to present the swapchain image." };
	}

	void RenderTarget::_onValidate()
	{
		// TODO
	}

	void RenderTarget::__createSwapchain(
		HWND const hWnd,
		UINT const width,
		UINT const height,
		UINT const imageCount)
	{
		DXGI_SWAP_CHAIN_DESC swapchainDesc	{ };

		swapchainDesc.BufferDesc.Format		= DXGI_FORMAT::DXGI_FORMAT_R8G8B8A8_UNORM;
		swapchainDesc.BufferDesc.Width		= width;
		swapchainDesc.BufferDesc.Height		= height;
		swapchainDesc.SampleDesc.Count		= 1U;
		swapchainDesc.BufferUsage			= DXGI_USAGE_RENDER_TARGET_OUTPUT;
		swapchainDesc.BufferCount			= imageCount;
		swapchainDesc.OutputWindow			= hWnd;
		swapchainDesc.Windowed				= TRUE;
		swapchainDesc.SwapEffect			= DXGI_SWAP_EFFECT::DXGI_SWAP_EFFECT_DISCARD;

		HRESULT result{ };
		result = __pDXGIFactory->CreateSwapChain(__pDevice, &swapchainDesc, &__pSwapchain);

		if (FAILED(result))
			throw std::runtime_error{ "Cannot create a swapchain." };
	}

	void RenderTarget::__resolveBackBuffer()
	{
		ID3D11Texture2D *pBackBuffer{ };

		HRESULT result{ };
		result = __pSwapchain->GetBuffer(
			0U, __uuidof(ID3D11Texture2D), reinterpret_cast<void **>(&pBackBuffer));

		if (FAILED(result))
			throw std::runtime_error{ "Cannot resolve a swapchain buffer." };

		int iFlags{ };
		iFlags |= cudaGraphicsRegisterFlags::cudaGraphicsRegisterFlagsSurfaceLoadStore;

		__pBackSurface = std::make_unique<Cuda::Surface>(
			pBackBuffer, static_cast<cudaGraphicsRegisterFlags>(iFlags));

		pBackBuffer->Release();
	}
}