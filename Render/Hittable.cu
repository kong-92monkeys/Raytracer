#include "Hittable.h"
#include "CudaMath.h"

namespace Render
{
	namespace Kernel
	{
		__host__ void Hittable::asSphere(
			float3 const &center,
			float const radius) noexcept
		{
			type = HittableType::SPHERE;

			auto const pFloats{ reinterpret_cast<float *>(context) };
			pFloats[0] = center.x;
			pFloats[1] = center.y;
			pFloats[2] = center.z;
			pFloats[3] = radius;
		}

		__device__ HitResult Hittable::hit(
			Ray const &ray) const noexcept
		{
			switch (type)
			{
				case HittableType::SPHERE:
					return __hit_sphere(ray);
			}

			return { };
		}

		__device__ HitResult Hittable::__hit_sphere(
			Ray const &ray) const noexcept
		{
			auto const pFloats			{ reinterpret_cast<float const *>(context) };
			float3 const sphereCenter	{ pFloats[0], pFloats[1], pFloats[2] };
			float const sphereRadius	{ pFloats[3] };

			float3 const &rayOrigin		{ ray.getOrigin() };
			float3 const &rayDir		{ ray.getDirection() };

			float3 const rayOrigin_center{ rayOrigin - sphereCenter };

			float const A{ 1.0f /*lengthSq(rayDir)*/ };
			float const B{ 2.0f * dot(rayDir, rayOrigin_center) };
			float const C{ lengthSq(rayOrigin_center) - (sphereRadius * sphereRadius) };

			HitResult retVal{ };

			// Discriminant
			float const D{ (B * B) - (4.0f * A * C) };
			if (D >= 0.0f)
			{
				float const Dsqrt{ sqrtf(D) };

				float rayLength{ -(B + Dsqrt) / (2.0f * A) };
				if (rayLength < 0.0f)
					rayLength += Dsqrt;

				if (rayLength >= 0.0f)
				{
					retVal.hit = true;
					retVal.spot = ray.at(rayLength);
					retVal.normal = normalize(retVal.spot - sphereCenter);
				}
			}

			return retVal;
		}
	}
}