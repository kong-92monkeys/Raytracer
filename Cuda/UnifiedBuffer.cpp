#include "UnifiedBuffer.h"
#include <stdexcept>

namespace Cuda
{
	UnifiedBuffer::~UnifiedBuffer() noexcept
	{
		if (!__pBuffer)
			return;

		cudaFree(__pBuffer);
	}

	void UnifiedBuffer::reserve(
		size_t const size)
	{
		if (__capacity >= size)
			return;

		__grow(size);
	}

	void UnifiedBuffer::resize(
		size_t const size)
	{
		if (__capacity >= size)
		{
			__size = size;
			return;
		}

		__grow(size);
		__size = size;
	}

	uint8_t *UnifiedBuffer::append(
		size_t const amount)
	{
		resize(getSize() + amount);
		return __pBuffer;
	}

	void UnifiedBuffer::__grow(
		size_t const reqSize)
	{
		if (__pBuffer)
			cudaFree(__pBuffer);

		__capacity = std::max(reqSize, __capacity << 1ULL);
		if (cudaMallocManaged(&__pBuffer, __capacity) != cudaError_t::cudaSuccess)
			throw std::runtime_error{ "Cannot allocate a unified memory." };
	}
}