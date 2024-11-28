#pragma once

#include "../Infra/Handle.h"
#include <cuda_runtime.h>

namespace Cuda
{
	class Event : public Infra::Handle<cudaEvent_t>
	{
	public:
		Event();
		virtual ~Event() noexcept override;
	};
}