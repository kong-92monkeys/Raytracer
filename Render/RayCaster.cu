#include "RayCaster.h"
#include "CudaMath.h"

namespace Render
{
	namespace Kernel
	{
		__host__ RayCaster::RayCaster(
			Viewport const &viewport,
			uint32_t const surfaceWidth,
			uint32_t const surfaceHeight) noexcept :
			__viewport	{ viewport },
			__stepX		{ viewport.width / surfaceWidth },
			__stepY		{ viewport.height / surfaceHeight }
		{}

		__device__ Ray RayCaster::cast(
			uint32_t const x,
			uint32_t const y) const noexcept
		{
			float3 rayDst{ __viewport.viewportOrigin };
			rayDst += (__viewport.right * (__stepX * (x + 0.5f)));
			rayDst += (__viewport.down * (__stepY * (y + 0.5f)));

			auto const dir{ normalize(rayDst - __viewport.rayOrigin) };
			return { __viewport.rayOrigin, dir };
		}
	}
}