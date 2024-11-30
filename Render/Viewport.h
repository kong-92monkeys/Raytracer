#pragma once

#include <cuda_runtime.h>

namespace Render
{
	namespace Kernel
	{
		struct Viewport
		{
		public:
			float3 rayOrigin{ };
			float3 viewportOrigin{ };
			float3 right{ };
			float3 down{ };
			float width{ };
			float height{ };
		};
	}
}