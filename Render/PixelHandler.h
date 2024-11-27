#pragma once

#include <cuda_runtime.h>

namespace Render
{
	namespace Kernel
	{
		class PixelHandler
		{
		public:
			__device__ PixelHandler(
				uint32_t gidX,
				uint32_t gidY,
				cudaSurfaceObject_t surface) noexcept;

			__device__ bool isValid(
				uint32_t width,
				uint32_t height) const noexcept;

			__device__ void set(
				uchar4 const &value);

			__device__ void set(
				float4 const &value);

		private:
			uint32_t const __gidX;
			uint32_t const __gidY;
			cudaSurfaceObject_t const __surface;
		};
	}
}
