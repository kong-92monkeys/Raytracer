#pragma once

#include "Viewport.h"
#include "Ray.h"

namespace Render
{
	namespace Kernel
	{
		class RayCaster
		{
		public:
			__host__ RayCaster(
				Viewport const &viewport,
				uint32_t surfaceWidth,
				uint32_t surfaceHeight) noexcept;

			__device__ Ray cast(
				uint32_t x, uint32_t y) const noexcept;

		private:
			Viewport const __viewport;
			float const __stepX;
			float const __stepY;
		};
	}
}