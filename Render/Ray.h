#pragma once

#include <cuda_runtime.h>

namespace Render
{
	namespace Kernel
	{
		class Ray
		{
		public:
			__device__ Ray(
				float3 const &origin,
				float3 const &dir) noexcept;

			__device__ float3 at(
				float length) const noexcept;

			__device__ float3 const &getOrigin() const noexcept;
			__device__ float3 const &getDirection() const noexcept;

		private:
			float3 __origin;
			float3 __dir;
		};
	}
}
