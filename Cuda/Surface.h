#pragma once

#include "../Infra/Handle.h"
#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>

namespace Cuda
{
	class Surface : public Infra::Handle<cudaSurfaceObject_t>
	{ };

	class InteropSurface : public Surface
	{
	public:
		InteropSurface(
			ID3D11Texture2D *pBuffer,
			cudaGraphicsRegisterFlags registerFlags);

		virtual ~InteropSurface() noexcept;

		void map();
		void unmap();

	private:
		cudaGraphicsResource_t __interopHandle{ };
		bool __mapped{ };

		void __registerInteropHandle(
			ID3D11Texture2D *pBuffer,
			cudaGraphicsRegisterFlags registerFlags);

		void __createHandle();
	};

	class ArraySurface : public Surface
	{
	public:
		ArraySurface(
			size_t width,
			size_t height,
			cudaChannelFormatDesc const &formatDesc,
			int arrayFlags);

		virtual ~ArraySurface() noexcept;

	private:
		cudaArray_t __arrHandle{ };

		void __createArray(
			size_t width,
			size_t height,
			cudaChannelFormatDesc const &formatDesc,
			int arrayFlags);

		void __createHandle();
	};
}