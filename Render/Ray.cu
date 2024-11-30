#include "Ray.h"
#include "CudaMath.h"

namespace Render
{
	namespace Kernel
	{
		__device__ Ray::Ray(
			float3 const &origin,
			float3 const &dir) noexcept :
			__origin	{ origin },
			__dir		{ dir }
		{}

		__device__ float3 Ray::at(
			float const length) const noexcept
		{
			return (__origin + (__dir * length));
		}

		__device__ float3 const &Ray::getOrigin() const noexcept
		{
			return __origin;
		}

		__device__ float3 const &Ray::getDirection() const noexcept
		{
			return __dir;
		}
	}
}