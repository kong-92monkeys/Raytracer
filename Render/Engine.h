#pragma once

#include <d3d11.h>
#include <dxgi.h>
#include "RenderTarget.h"
#include <unordered_set>

namespace Render
{
	class Engine : public Infra::Unique
	{
	public:
		Engine();
		virtual ~Engine() noexcept override;

		[[nodiscard]]
		RenderTarget *createRenderTarget(
			HWND hWnd,
			UINT width,
			UINT height,
			UINT swapchainImageCount);

		void reserveRender(
			RenderTarget *pRenderTarget) noexcept;

		void cancelRender(
			RenderTarget *pRenderTarget) noexcept;

		void render();

	private:
		ID3D11Device *__pDevice{ };
		ID3D11DeviceContext *__pContext{ };

		IDXGIDevice *__pDXGIDevice{ };
		IDXGIAdapter *__pDXGIAdapter{ };
		IDXGIFactory *__pDXGIFactory{ };

		std::unique_ptr<Cuda::Stream> __pRenderStream;
		std::unordered_set<RenderTarget *> __reservedRenderTargets;

		void __createDevice();
		void __resolveDXGIReferences();

		void __validateReservedRenderTargets();
	};
}