#pragma once

#include <cuda_runtime.h>

namespace Render
{
	namespace Kernel
	{
		struct ResourceContext
		{
		public:

		};

		struct SurfaceContext
		{
		public:
			cudaSurfaceObject_t surface{ };
			uint32_t width{ };
			uint32_t height{ };
		};

		struct LaunchContext
		{
		public:
			dim3 gridSize{ };
			dim3 blockSize{ };
			cudaStream_t stream{ };
		};
	}
}
