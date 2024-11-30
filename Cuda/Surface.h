#pragma once

#include "../Infra/Handle.h"
#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>

namespace Cuda
{
	class Surface : public Infra::Handle<cudaSurfaceObject_t>
	{
	public:
		void copy(
			Surface const &src,
			size_t wOffsetDst, size_t hOffsetDst,
			size_t wOffsetSrc, size_t hOffsetSrc,
			size_t width, size_t height);

		[[nodiscard]]
		constexpr cudaArray_t const &getArrayHandle() const noexcept;

	protected:
		constexpr void _setArrayHandle(
			cudaArray_t handle) noexcept;

	private:
		cudaArray_t __arrHandle{ };
	};

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
		void __createArray(
			size_t width,
			size_t height,
			cudaChannelFormatDesc const &formatDesc,
			int arrayFlags);

		void __createHandle();
	};

	constexpr cudaArray_t const &Surface::getArrayHandle() const noexcept
	{
		return __arrHandle;
	}

	constexpr void Surface::_setArrayHandle(
		cudaArray_t const handle) noexcept
	{
		__arrHandle = handle;
	}
}