#pragma once

#include <cuda_runtime.h>

namespace Render
{
	namespace Kernel
	{
		struct ResourceContext
		{
		public:
			float3 sphereCenter	{ 0.0f, 0.0f, 0.0f };
			float sphereRadius	{ 1.0f };
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
			dim3 gridDim{ };
			dim3 blockDim{ };
			cudaStream_t stream{ };
		};
	}
}
