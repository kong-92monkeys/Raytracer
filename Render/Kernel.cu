#include "Kernel.h"
#include "PixelHandler.h"
#include <device_launch_parameters.h>

namespace Render
{
	namespace Kernel
	{
		__global__ void launch_device(
			ResourceContext const resourceContext,
			SurfaceContext const surfaceContext)
		{
			auto const gidX{ (blockIdx.x * blockDim.x) + threadIdx.x };
			auto const gidY{ (blockIdx.y * blockDim.y) + threadIdx.y };

			PixelHandler pixelHandler{ gidX, gidY, surfaceContext };

			uchar4 color{ 255, 0, 255, 255 };
			surf2Dwrite(color, surfaceContext.surface, gidX * sizeof(uchar4), gidY);

			/*if (!(pixelHandler.isValid()))
				return;

			uchar4 color{ 255, 0, 255, 255 };
			pixelHandler.set(color);*/
		}

		void launch(
			ResourceContext const &resourceContext,
			SurfaceContext const &surfaceContext,
			LaunchContext const &launchContext)
		{
			launch_device<<<
				launchContext.gridSize,
				launchContext.blockSize,
				0U,
				launchContext.stream>>>
				(resourceContext, surfaceContext);
		}
	}
}