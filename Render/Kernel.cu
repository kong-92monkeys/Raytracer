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

			if (!(pixelHandler.isValid()))
				return;

			float4 color{ 0.0f, 0.0f, 0.0f, 1.0f };

			float const rayX{ gidX * 1.0f };
			float const rayY{ gidY * 1.0f };

			float const deltaX{ rayX - resourceContext.sphereCenter.x };
			float const deltaY{ rayY - resourceContext.sphereCenter.y };
			float const radius{ resourceContext.sphereRadius };

			if ((radius * radius) > ((deltaX * deltaX) + (deltaY * deltaY)))
				color.x = 1.0f;

			pixelHandler.set(color);
		}

		void launch(
			ResourceContext const &resourceContext,
			SurfaceContext const &surfaceContext,
			LaunchContext const &launchContext)
		{
			launch_device<<<
				launchContext.gridDim,
				launchContext.blockDim,
				0U,
				launchContext.stream>>>
				(resourceContext, surfaceContext);
		}
	}
}