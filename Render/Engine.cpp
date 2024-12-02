#include "Engine.h"

namespace Render
{
	Engine::Engine()
	{
		__createDevice();
		__resolveDXGIReferences();
		__pRenderStream = std::make_unique<Cuda::Stream>();
	}

	Engine::~Engine() noexcept
	{
		__pRenderStream = nullptr;

		__pDXGIFactory->Release();
		__pDXGIAdapter->Release();
		__pDXGIDevice->Release();
		__pContext->Release();
		__pDevice->Release();
	}

	RenderTarget *Engine::createRenderTarget(
		HWND const hwnd,
		UINT const width,
		UINT const height,
		UINT const swapchainImageCount)
	{
		return new RenderTarget
		{
			__pDevice, __pDXGIFactory, *__pRenderStream,
			hwnd, width, height, swapchainImageCount
		};
	}

	void Engine::reserveRender(
		RenderTarget *const pRenderTarget) noexcept
	{
		__reservedRenderTargets.emplace(pRenderTarget);
	}

	void Engine::cancelRender(
		RenderTarget *const pRenderTarget) noexcept
	{
		__reservedRenderTargets.erase(pRenderTarget);
	}

	void Engine::render()
	{
		if (__reservedRenderTargets.empty())
			return;

		//__pGlobalDescriptorManager->validate();
		__validateReservedRenderTargets();

		//__pDescriptorUpdater->update();

		for (auto const pRenderTarget : __reservedRenderTargets)
		{
			if (!(pRenderTarget->isPresentable()))
				continue;

			pRenderTarget->draw();
			pRenderTarget->present();
		}

		//__pCommandSubmitter->present();

		//__deferredDeleter.advance();
		__reservedRenderTargets.clear();
	}

	void Engine::__createDevice()
	{
		auto const featureLevel{ D3D_FEATURE_LEVEL::D3D_FEATURE_LEVEL_11_1 };

		HRESULT result{ };
		result = D3D11CreateDevice(
			nullptr,
			D3D_DRIVER_TYPE::D3D_DRIVER_TYPE_HARDWARE,
			nullptr,
			0U,
			&featureLevel, 1U,
			D3D11_SDK_VERSION,
			&__pDevice, nullptr, &__pContext);

		if (FAILED(result))
			throw std::runtime_error{ "Cannot create a device." };
	}

	void Engine::__resolveDXGIReferences()
	{
		HRESULT result{ };

		// DXGI Device
		result = __pDevice->QueryInterface(
			__uuidof(IDXGIDevice), reinterpret_cast<void **>(&__pDXGIDevice));

		if (FAILED(result))
			throw std::runtime_error{ "Cannot resolve the DXGI references." };

		// DXGI Adapter
		result = __pDXGIDevice->GetAdapter(&__pDXGIAdapter);
		if (FAILED(result))
			throw std::runtime_error{ "Cannot resolve the DXGI references." };

		// DXGI Factory
		result = __pDXGIAdapter->GetParent(
			__uuidof(IDXGIFactory), reinterpret_cast<void **>(&__pDXGIFactory));

		if (FAILED(result))
			throw std::runtime_error{ "Cannot resolve the DXGI references." };
	}

	void Engine::__validateReservedRenderTargets()
	{
		for (auto it{ __reservedRenderTargets.begin() }; it != __reservedRenderTargets.end(); )
		{
			auto const pRenderTarget{ *it };

			if (pRenderTarget->isPresentable())
			{
				pRenderTarget->validate();
				++it;
			}
			else
				it = __reservedRenderTargets.erase(it);
		}
	}
}