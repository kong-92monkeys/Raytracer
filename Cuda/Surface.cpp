#include "Surface.h"
#include <stdexcept>

namespace Cuda
{
	Surface::Surface(
		ID3D11Texture2D *const pBuffer,
		cudaGraphicsRegisterFlags const registerFlags)
	{
		__registerInteropHandle(pBuffer, registerFlags);
		__createHandle();
	}

	Surface::~Surface() noexcept
	{
		cudaDestroySurfaceObject(getHandle());
		cudaGraphicsUnregisterResource(__interopHandle);
	}

	void Surface::map()
	{
		if (__mapped)
			return;

		if (cudaGraphicsMapResources(1, &__interopHandle) != cudaError_t::cudaSuccess)
			throw std::runtime_error{ "Cannot map a resource." };

		__mapped = true;
	}

	void Surface::unmap()
	{
		if (!__mapped)
			return;

		if (cudaGraphicsUnmapResources(1, &__interopHandle) != cudaError_t::cudaSuccess)
			throw std::runtime_error{ "Cannot unmap a resource." };

		__mapped = false;
	}

	void Surface::__registerInteropHandle(
		ID3D11Texture2D *const pBuffer,
		cudaGraphicsRegisterFlags const registerFlags)
	{
		cudaError_t result{ };

		result = cudaGraphicsD3D11RegisterResource(&__interopHandle, pBuffer, registerFlags);
		if (result != cudaError_t::cudaSuccess)
			throw std::runtime_error{ "Cannot register a resource." };
	}

	void Surface::__createHandle()
	{
		cudaError_t result{ };

		map();

		cudaArray_t cuArr{ };
		result = cudaGraphicsSubResourceGetMappedArray(&cuArr, __interopHandle, 0U, 0U);

		if (result != cudaError_t::cudaSuccess)
			throw std::runtime_error{ "Cannot resolve a cudaArray." };

		cudaResourceDesc resDesc{ };
		resDesc.resType				= cudaResourceType::cudaResourceTypeArray;
		resDesc.res.array.array		= cuArr;

		cudaSurfaceObject_t handle{ };
		result = cudaCreateSurfaceObject(&handle, &resDesc);

		if (result != cudaError_t::cudaSuccess)
			throw std::runtime_error{ "Cannot create a surface object." };

		_setHandle(handle);
		unmap();
	}
}