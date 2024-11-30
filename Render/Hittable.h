#pragma once

#include "Ray.h"

namespace Render
{
	namespace Kernel
	{
		enum class HittableType
		{
			UNKNOWN = -1,
			SPHERE
		};

		struct HitResult
		{
		public:
			bool hit{ };
			float3 spot{ };
		};

		class Hittable
		{
		public:
			HittableType type{ HittableType::UNKNOWN };
			uint8_t context[32];

			__host__ void asSphere(
				float3 const &center,
				float const radius) noexcept;

			__device__ HitResult hit(
				Ray const &ray) const noexcept;

		private:
			__device__ HitResult __hit_sphere(
				Ray const &ray) const noexcept;
		};
	}
}