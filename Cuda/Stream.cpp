#include "Stream.h"
#include <stdexcept>

namespace Cuda
{
	Stream::Stream()
	{
		cudaError_t result{ };

		cudaStream_t handle{ };
		result = cudaStreamCreate(&handle);

		if (result != cudaError_t::cudaSuccess)
			throw std::runtime_error{ "Cannot create a stream." };

		_setHandle(handle);
	}

	Stream::~Stream() noexcept
	{
		cudaStreamDestroy(getHandle());
	}

	void Stream::memcpy(
		void *const dst,
		const void *const src,
		size_t const count,
		cudaMemcpyKind const kind)
	{
		cudaError_t result{ };
		result = cudaMemcpyAsync(dst, src, count, kind, getHandle());

		if (result != cudaError_t::cudaSuccess)
			throw std::runtime_error{ "Failed to copy the data." };
	}

	void Stream::recordEvent(
		Event &event)
	{
		cudaError_t result{ };
		result = cudaEventRecord(event.getHandle(), getHandle());

		if (result != cudaError_t::cudaSuccess)
			throw std::runtime_error{ "Failed to record the given event with the stream." };
	}

	cudaError_t Stream::queryEvent(
		Event &event)
	{
		return cudaEventQuery(event.getHandle());
	}

	void Stream::syncEvent(
		Event &event)
	{
		cudaError_t result{ };
		result = cudaEventSynchronize(event.getHandle());

		if (result != cudaError_t::cudaSuccess)
			throw std::runtime_error{ "Failed to sync the event." };
	}

	void Stream::prefetch(
		UnifiedBuffer const &buffer,
		size_t const from,
		size_t const count)
	{
		auto const ptr{ buffer.getData() + from };

		if (cudaMemPrefetchAsync(ptr, count, 0, getHandle()) != cudaError_t::cudaSuccess)
			throw std::runtime_error{ "Cannot prefetch the memory." };
	}

	void Stream::prefetch(
		UnifiedBuffer const &buffer)
	{
		prefetch(buffer, 0ULL, buffer.getSize());
	}

	void Stream::sync()
	{
		cudaError_t result{ };
		cudaStreamSynchronize(getHandle());

		if (result != cudaError_t::cudaSuccess)
			throw std::runtime_error{ "Failed to sync with the stream." };
	}
}