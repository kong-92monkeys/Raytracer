#pragma once

#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>

namespace Cuda
{
	class Surface
	{
	public:
		Surface(
			ID3D11Texture2D *pImage,
			cudaGraphicsRegisterFlags registerFlags);

		virtual ~Surface() noexcept;

		void map();
		void unmap();

		[[nodiscard]]
		constexpr cudaSurfaceObject_t getHandle() noexcept;

	private:
		cudaGraphicsResource_t __interopHandle{ };
		cudaSurfaceObject_t __handle{ };

		void __registerInteropHandle(
			ID3D11Texture2D *pImage,
			cudaGraphicsRegisterFlags registerFlags);

		void __createHandle();
	};

	constexpr cudaSurfaceObject_t Surface::getHandle() noexcept
	{
		return __handle;
	}
}