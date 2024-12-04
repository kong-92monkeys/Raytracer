#pragma once

#include "Viewport.h"
#include "HittableContext.h"

namespace Render
{
	namespace Kernel
	{
		struct RenderContext
		{
		public:
			HittableContext hittableContext;
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
