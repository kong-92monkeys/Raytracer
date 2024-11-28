#include "Event.h"
#include <stdexcept>

namespace Cuda
{
	Event::Event()
	{
		cudaError_t result{ };

		cudaEvent_t handle{ };
		result = cudaEventCreate(&handle, cudaEventBlockingSync);

		if (result != cudaError_t::cudaSuccess)
			throw std::runtime_error{ "Cannot create a event." };

		_setHandle(handle);
	}

	Event::~Event() noexcept
	{
		cudaEventDestroy(getHandle());
	}
}