#include "RenderTarget.h"

namespace Render
{
	RenderTarget::RenderTarget(
		ID3D11Device *const pDevice,
		IDXGIFactory *const pDXGIFactory,
		Cuda::Stream &renderStream,
		HWND const hwnd,
		UINT const width,
		UINT const height,
		UINT const swapBufferCount) :
		__pDevice			{ pDevice },
		__pDXGIFactory		{ pDXGIFactory },
		__renderStream		{ renderStream },
		__swapBufferCount	{ swapBufferCount }
	{
		__swapBuffers.resize(__swapBufferCount);
		__width		= width;
		__height	= height;

		for (UINT iter{ }; iter < swapBufferCount; ++iter)
			__launchEvents.emplace_back(std::make_shared<Cuda::Event>());

		__kernelLauncher.setStream(renderStream.getHandle());
		__kernelLauncher.setSurfaceExtent(width, height);
		
		__createSwapchain(hwnd);
		__resolveBackSurface();

		if (isPresentable())
			__createSwapBuffers();
	}

	RenderTarget::~RenderTarget() noexcept
	{
		__clearSwapBuffers();
		__pSwapchainSurface = nullptr;
		__pSwapchain->Release();
		__launchEvents.clear();
	}

	void RenderTarget::resize(
		UINT const width,
		UINT const height)
	{
		__renderStream.sync();
		__backBufferIdx		= 0U;
		__frontBufferIdx	= 0U;

		__width		= width;
		__height	= height;
		__kernelLauncher.setSurfaceExtent(width, height);

		__clearSwapBuffers();
		__pSwapchainSurface = nullptr;

		HRESULT result{ };
		result = __pSwapchain->ResizeBuffers(0U, width, height, DXGI_FORMAT_R8G8B8A8_UNORM, 0U);

		if (FAILED(result))
			throw std::runtime_error{ "Cannot resize the swapchain." };

		__resolveBackSurface();

		if (isPresentable())
			__createSwapBuffers();
	}

	void RenderTarget::draw()
	{
		if (__isSwapBufferFull())
			return;

		auto &backBuffer	{ *(__swapBuffers[__backBufferIdx]) };
		auto &backEvent		{ *(__launchEvents[__backBufferIdx]) };

		__kernelLauncher.launch(backBuffer.getHandle());
		__renderStream.recordEvent(backEvent);
		
		__rotateBackBuffer();
	}

	void RenderTarget::present()
	{
		if (__isSwapBufferEmpty())
			return;

		auto &frontBuffer	{ *(__swapBuffers[__frontBufferIdx]) };
		auto &frontEvent	{ *(__launchEvents[__frontBufferIdx]) };

		if (__renderStream.queryEvent(frontEvent) != cudaError_t::cudaSuccess)
			return;

		__pSwapchainSurface->map();
		__pSwapchainSurface->copy(
			frontBuffer, 0ULL, 0ULL, 0ULL, 0ULL,
			__width * sizeof(uchar4), __height);

		__pSwapchainSurface->unmap();

		HRESULT result{ };
		result = __pSwapchain->Present(0U, 0U);

		if (FAILED(result))
			throw std::runtime_error{ "Failed to present the swapchain image." };

		__rotateFrontBuffer();
	}

	void RenderTarget::_onValidate()
	{
		// TODO
	}

	void RenderTarget::__createSwapchain(
		HWND const hwnd)
	{
		DXGI_SWAP_CHAIN_DESC swapchainDesc	{ };

		swapchainDesc.BufferDesc.Format		= DXGI_FORMAT::DXGI_FORMAT_R8G8B8A8_UNORM;
		swapchainDesc.BufferDesc.Width		= __width;
		swapchainDesc.BufferDesc.Height		= __height;
		swapchainDesc.SampleDesc.Count		= 1U;
		swapchainDesc.BufferUsage			= DXGI_USAGE_RENDER_TARGET_OUTPUT;
		swapchainDesc.BufferCount			= __SWAPCHAIN_IMAGE_COUNT;
		swapchainDesc.OutputWindow			= hwnd;
		swapchainDesc.Windowed				= TRUE;
		swapchainDesc.SwapEffect			= DXGI_SWAP_EFFECT::DXGI_SWAP_EFFECT_DISCARD;

		HRESULT result{ };
		result = __pDXGIFactory->CreateSwapChain(__pDevice, &swapchainDesc, &__pSwapchain);

		if (FAILED(result))
			throw std::runtime_error{ "Cannot create a swapchain." };
	}

	void RenderTarget::__resolveBackSurface()
	{
		ID3D11Texture2D *pBackBuffer{ };

		HRESULT result{ };
		result = __pSwapchain->GetBuffer(
			0U, __uuidof(ID3D11Texture2D), reinterpret_cast<void **>(&pBackBuffer));

		if (FAILED(result))
			throw std::runtime_error{ "Cannot resolve a swapchain buffer." };

		int iFlags{ };
		iFlags |= cudaGraphicsRegisterFlags::cudaGraphicsRegisterFlagsSurfaceLoadStore;

		__pSwapchainSurface = std::make_unique<Cuda::InteropSurface>(
			pBackBuffer, static_cast<cudaGraphicsRegisterFlags>(iFlags));

		pBackBuffer->Release();
	}

	void RenderTarget::__clearSwapBuffers()
	{
		for (auto const pSwapBuffer : __swapBuffers)
			delete pSwapBuffer;

		__swapBufferCreated = false;
	}

	void RenderTarget::__createSwapBuffers()
	{
		auto const channelDesc{ cudaCreateChannelDesc<uchar4>() };

		int flags{ };
		flags |= cudaArraySurfaceLoadStore;

		for (auto &pSwapBuffer : __swapBuffers)
			pSwapBuffer = new Cuda::ArraySurface{ __width, __height, channelDesc, flags };

		__swapBufferCreated = true;
	}
}