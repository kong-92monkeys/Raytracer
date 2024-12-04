#pragma once

#include "Ray.h"
#include "HittableContext.h"

namespace Render
{
	namespace Kernel
	{
		struct HitResult
		{
		public:
			bool hit{ };
			float3 spot{ };
			float3 normal{ };
		};

		class HitTester
		{
		public:
			__device__ HitTester(
				HittableContext const &context) noexcept;

			__device__ HitResult hit(
				size_t hittableHandle,
				Ray const &ray) const noexcept;

		private:
			HittableContext const &__context;

			__device__ static HitResult __hit(
				SphereContent const &content,
				Ray const &ray) noexcept;
		};
	}
}
