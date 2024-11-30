#include "Surface.h"
#include <stdexcept>

namespace Cuda
{
	void Surface::copy(
		Surface const &src,
		size_t const wOffsetDst, size_t const hOffsetDst,
		size_t const wOffsetSrc, size_t const hOffsetSrc,
		size_t const width, size_t const height)
	{
		cudaError_t result{ };

		result = cudaMemcpy2DArrayToArray(
			getArrayHandle(),
			wOffsetDst, hOffsetDst,
			src.getArrayHandle(),
			wOffsetSrc, hOffsetSrc,
			width, height, cudaMemcpyKind::cudaMemcpyDeviceToDevice);

		if (result != cudaError_t::cudaSuccess)
			throw std::runtime_error{ "Cannot copy from the surface." };
	}

	InteropSurface::InteropSurface(
		ID3D11Texture2D *const pBuffer,
		cudaGraphicsRegisterFlags const registerFlags)
	{
		__registerInteropHandle(pBuffer, registerFlags);
		__createHandle();
	}

	InteropSurface::~InteropSurface() noexcept
	{
		cudaDestroySurfaceObject(getHandle());
		cudaGraphicsUnregisterResource(__interopHandle);
	}

	void InteropSurface::map()
	{
		if (__mapped)
			return;

		if (cudaGraphicsMapResources(1, &__interopHandle) != cudaError_t::cudaSuccess)
			throw std::runtime_error{ "Cannot map a resource." };

		__mapped = true;
	}

	void InteropSurface::unmap()
	{
		if (!__mapped)
			return;

		if (cudaGraphicsUnmapResources(1, &__interopHandle) != cudaError_t::cudaSuccess)
			throw std::runtime_error{ "Cannot unmap a resource." };

		__mapped = false;
	}

	void InteropSurface::__registerInteropHandle(
		ID3D11Texture2D *const pBuffer,
		cudaGraphicsRegisterFlags const registerFlags)
	{
		cudaError_t result{ };

		result = cudaGraphicsD3D11RegisterResource(&__interopHandle, pBuffer, registerFlags);
		if (result != cudaError_t::cudaSuccess)
			throw std::runtime_error{ "Cannot register a resource." };
	}

	void InteropSurface::__createHandle()
	{
		cudaError_t result{ };

		map();

		cudaArray_t arrHandle{ };
		result = cudaGraphicsSubResourceGetMappedArray(&arrHandle, __interopHandle, 0U, 0U);

		if (result != cudaError_t::cudaSuccess)
			throw std::runtime_error{ "Cannot resolve a cudaArray." };

		cudaResourceDesc resDesc{ };
		resDesc.resType				= cudaResourceType::cudaResourceTypeArray;
		resDesc.res.array.array		= arrHandle;

		cudaSurfaceObject_t handle{ };
		result = cudaCreateSurfaceObject(&handle, &resDesc);

		if (result != cudaError_t::cudaSuccess)
			throw std::runtime_error{ "Cannot create a surface object." };

		_setArrayHandle(arrHandle);
		_setHandle(handle);

		unmap();
	}

	ArraySurface::ArraySurface(
		size_t const width,
		size_t const height,
		cudaChannelFormatDesc const &formatDesc,
		int const arrayFlags)
	{
		__createArray(width, height, formatDesc, arrayFlags);
		__createHandle();
	}

	ArraySurface::~ArraySurface() noexcept
	{
		cudaDestroySurfaceObject(getHandle());
		cudaFreeArray(getArrayHandle());
	}

	void ArraySurface::__createArray(
		size_t const width,
		size_t const height,
		cudaChannelFormatDesc const &formatDesc,
		int const arrayFlags)
	{
		cudaError_t result{ };

		cudaArray_t arrHandle{ };
		result = cudaMallocArray(&arrHandle, &formatDesc, width, height, arrayFlags);

		if (result != cudaError_t::cudaSuccess)
			throw std::runtime_error{ "Cannot create a cudaArray." };

		_setArrayHandle(arrHandle);
	}

	void ArraySurface::__createHandle()
	{
		cudaError_t result{ };

		cudaResourceDesc resDesc{ };
		resDesc.resType = cudaResourceType::cudaResourceTypeArray;
		resDesc.res.array.array = getArrayHandle();

		cudaSurfaceObject_t handle{ };
		result = cudaCreateSurfaceObject(&handle, &resDesc);

		if (result != cudaError_t::cudaSuccess)
			throw std::runtime_error{ "Cannot create a surface object." };

		_setHandle(handle);
	}
}