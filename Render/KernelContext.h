#pragma once

#include "Viewport.h"
#include "Hittable.h"

namespace Render
{
	namespace Kernel
	{
		struct RenderContext
		{
		public:
			Hittable hittable{ };
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
