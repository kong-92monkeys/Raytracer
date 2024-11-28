#include "Swapchain.h"
#include <stdexcept>

namespace D3D
{
	Swapchain::Swapchain(
		HWND const hWnd,
		UINT const width,
		UINT const height,
		UINT const imageCount) :
		__imageCount{ imageCount }
	{
		__initSwapchain(hWnd, width, height, imageCount);

		__width = width;
		__height = height;
		__buffers.resize(imageCount);

		__resolveBuffers();
	}

	Swapchain::~Swapchain() noexcept
	{
		__clearBuffers();

		__pContext->Release();
		__pDevice->Release();
		__pSwapchain->Release();
	}

	void Swapchain::resize(
		UINT const width,
		UINT const height)
	{
		HRESULT result{ };

		__clearBuffers();

		result = __pSwapchain->ResizeBuffers(
			0U, width, height, DXGI_FORMAT::DXGI_FORMAT_UNKNOWN, 0U);

		if (FAILED(result))
			throw std::runtime_error{ "Cannot resize the swapchain." };

		__width		= width;
		__height	= height;

		__resolveBuffers();
	}

	UINT Swapchain::getBackBufferIndex() noexcept
	{
		return __pSwapchain->GetCurrentBackBufferIndex();
	}

	void Swapchain::present()
	{
		HRESULT result{ };
		result = __pSwapchain->Present(0U, 0U);

		if (FAILED(result))
			throw std::runtime_error{ "Cannot present the next image." };
	}

	void Swapchain::__initSwapchain(
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
		swapchainDesc.SwapEffect			= DXGI_SWAP_EFFECT::DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;

		HRESULT result{ };

		IDXGISwapChain *pTempSwapchain{ };
		auto const featureLevel{ D3D_FEATURE_LEVEL::D3D_FEATURE_LEVEL_11_1 };

		result = D3D11CreateDeviceAndSwapChain(
			nullptr,
			D3D_DRIVER_TYPE::D3D_DRIVER_TYPE_HARDWARE,
			nullptr, 0U, &featureLevel, 1U,
			D3D11_SDK_VERSION, &swapchainDesc,
			&pTempSwapchain, &__pDevice, nullptr, &__pContext);

		if (FAILED(result))
			throw std::runtime_error{ "Cannot create a swapchain." };

		result = pTempSwapchain->QueryInterface(
			__uuidof(IDXGISwapChain3), reinterpret_cast<void **>(&__pSwapchain));

		if (FAILED(result))
			throw std::runtime_error{ "Cannot resolve a IDXGISwapChain3." };

		pTempSwapchain->Release();
	}

	void Swapchain::__resolveBuffers()
	{
		HRESULT result{ };

		UINT const bufferCount{ static_cast<UINT>(__buffers.size()) };
		for (UINT idx{ }; idx < bufferCount; ++idx)
		{
			auto &pBuffer{ __buffers[idx] };

			result = __pSwapchain->GetBuffer(
				idx, __uuidof(ID3D11Texture2D), reinterpret_cast<void **>(&pBuffer));

			if (FAILED(result))
				throw std::runtime_error{ "Cannot resolve a swapchain buffer." };
		}
	}

	void Swapchain::__clearBuffers()
	{
		for (auto const pBuffer : __buffers)
			pBuffer->Release();

		ZeroMemory(__buffers.data(), __buffers.size() * sizeof(ID3D11Texture2D *));
	}
}