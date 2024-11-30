#pragma once

#include <cuda_runtime.h>

namespace Render
{
	namespace Kernel
	{
		struct Viewport
		{
		public:
			float3 eye{ };
			float3 origin{ };
			float3 right{ };
			float3 down{ };
			float width{ };
			float height{ };
		};
	}
}