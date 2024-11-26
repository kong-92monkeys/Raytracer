#include "Swapchain.h"
#include <stdexcept>

namespace D3D
{
	Swapchain::Swapchain(
		HWND const hWnd,
		UINT const width,
		UINT const height)
	{
		DXGI_SWAP_CHAIN_DESC swapchainDesc	{ };

		swapchainDesc.BufferCount			= 3U;
		swapchainDesc.BufferUsage			= DXGI_USAGE_RENDER_TARGET_OUTPUT;
		swapchainDesc.BufferDesc.Format		= DXGI_FORMAT::DXGI_FORMAT_R8G8B8A8_UNORM;
		swapchainDesc.BufferDesc.Width		= width;
		swapchainDesc.BufferDesc.Height		= height;

		swapchainDesc.OutputWindow			= hWnd;
		swapchainDesc.SampleDesc.Count		= 1;
		swapchainDesc.Windowed				= TRUE;

		HRESULT result{ };

		IDXGISwapChain *pTempSwapchain{ };

		result = D3D11CreateDeviceAndSwapChain(
			nullptr,
			D3D_DRIVER_TYPE::D3D_DRIVER_TYPE_HARDWARE,
			nullptr, 0U, nullptr, 0U,
			D3D11_SDK_VERSION, &swapchainDesc,
			&pTempSwapchain, &__pDevice, nullptr, &__pContext);

		if (FAILED(result))
			throw std::runtime_error{ "Cannot create a swapchain." };

		result = pTempSwapchain->QueryInterface(
			__uuidof(IDXGISwapChain3), reinterpret_cast<void **>(&__pSwapChain));

		if (FAILED(result))
			throw std::runtime_error{ "Cannot resolve a IDXGISwapChain3." };

		pTempSwapchain->Release();
	}

	Swapchain::~Swapchain() noexcept
	{
		__pContext->Release();
		__pDevice->Release();
		__pSwapChain->Release();
	}

	void Swapchain::resize(
		UINT const width,
		UINT const height)
	{
		HRESULT result{ };

		result = __pSwapChain->ResizeBuffers(
			0U, width, height, DXGI_FORMAT::DXGI_FORMAT_UNKNOWN, 0U);

		if (FAILED(result))
			throw std::runtime_error{ "Cannot resize the swapchain." };
	}

	UINT Swapchain::getNextImageIndex() noexcept
	{
		return __pSwapChain->GetCurrentBackBufferIndex();
	}

	std::shared_ptr<ID3D11Texture2D> Swapchain::acquireImageOf(
		UINT const index)
	{
		ID3D11Texture2D *pRetVal{ };
		HRESULT result{ };

		result = __pSwapChain->GetBuffer(
			index, __uuidof(ID3D11Texture2D), reinterpret_cast<void **>(&pRetVal));

		return std::shared_ptr<ID3D11Texture2D>
		{
			pRetVal,
			Swapchain::__customDeleter<ID3D11Texture2D>
		};
	}

	void Swapchain::present()
	{
		HRESULT result{ };

		result = __pSwapChain->Present(0U, 0U);

		if (FAILED(result))
			throw std::runtime_error{ "Cannot present the next image." };
	}
}