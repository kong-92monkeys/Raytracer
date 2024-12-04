#pragma once

#include "RenderTarget.h"
#include "Sphere.h"
#include <unordered_set>

namespace Render
{
	class Engine : public Infra::Unique
	{
	public:
		Engine();
		virtual ~Engine() noexcept override;

		[[nodiscard]]
		Sphere *createSphere(
			glm::vec3 const &center,
			float radius) noexcept;

		[[nodiscard]]
		RenderTarget *createRenderTarget(
			HWND hwnd,
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

		HittableManager __hittableManager;

		void __createDevice();
		void __resolveDXGIReferences();

		void __validateReservedRenderTargets();
		void __onHittableManagerUpdated();
	};
}