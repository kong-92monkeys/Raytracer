#pragma once

#include "KernelContext.h"

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
				SurfaceContext const &surfaceContext) noexcept;

			__device__ bool isValid() const noexcept;

			__device__ void set(
				uchar4 const &value);

			__device__ void set(
				float4 const &value);

		private:
			uint32_t const __gidX;
			uint32_t const __gidY;
			SurfaceContext const &__surfaceContext;
		};
	}
}
