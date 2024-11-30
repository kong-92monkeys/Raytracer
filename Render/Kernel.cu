#include "Kernel.h"
#include "PixelHandler.h"
#include "RayCaster.h"
#include <device_launch_parameters.h>

namespace Render
{
	namespace Kernel
	{
		__global__ void launch_device(
			RenderContext const renderContext,
			SurfaceContext const surfaceContext,
			RayCaster const rayCaster)
		{
			auto const gidX{ (blockIdx.x * blockDim.x) + threadIdx.x };
			auto const gidY{ (blockIdx.y * blockDim.y) + threadIdx.y };

			PixelHandler pixelHandler{ gidX, gidY, surfaceContext };

			if (!(pixelHandler.isValid()))
				return;

			auto const ray{ rayCaster.cast(gidX, gidY) };

			float4 color{ 0.0f, 0.0f, 0.0f, 1.0f };

			auto const result{ renderContext.hittable.hit(ray) };
			if (result.hit)
				color.x = 1.0f;

			pixelHandler.set(color);
		}

		void launch(
			Kernel::Viewport const &viewport,
			RenderContext const &renderContext,
			SurfaceContext const &surfaceContext,
			LaunchContext const &launchContext)
		{
			RayCaster const rayCaster
			{
				viewport,
				surfaceContext.width,
				surfaceContext.height
			};

			launch_device<<<
				launchContext.gridDim,
				launchContext.blockDim,
				0U,
				launchContext.stream>>>
				(renderContext, surfaceContext, rayCaster);
		}
	}
}