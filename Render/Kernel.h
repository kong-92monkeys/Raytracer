#pragma once

#include <cuda_runtime.h>

namespace Render
{
	namespace Kernel
	{
		struct EngineContext
		{
		public:

		};

		struct RenderTargetContext
		{
		public:
			cudaSurfaceObject_t surface{ };
			uint32_t width{ };
			uint32_t height{ };
		};

		void launch(
			EngineContext const &engineContext,
			RenderTargetContext const &renderTargetContext,
			dim3 const &gridSize,
			dim3 const &blockSize);
	}
}
