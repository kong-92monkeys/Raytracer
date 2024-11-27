#include "Kernel.h"
#include "PixelHandler.h"
#include <device_launch_parameters.h>

namespace Render
{
	namespace Kernel
	{
		__global__ void launch_device(
			EngineContext const engineContext,
			RenderTargetContext const renderTargetContext)
		{
			auto const gidX{ (blockIdx.x * blockDim.x) + threadIdx.x };
			auto const gidY{ (blockIdx.y * blockDim.y) + threadIdx.y };

			PixelHandler pixelHandler{ gidX, gidY, renderTargetContext.surface };

			if (!((pixelHandler.isValid(renderTargetContext.width, renderTargetContext.height))))
				return;

			float4 color{ 1.0f, 0.0f, 1.0f, 1.0f };
			pixelHandler.set(color);
		}

		void launch(
			EngineContext const &engineContext,
			RenderTargetContext const &renderTargetContext,
			dim3 const &gridSize,
			dim3 const &blockSize)
		{
			launch_device<<<gridSize, blockSize>>>(engineContext, renderTargetContext);
		}
	}
}