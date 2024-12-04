#include "HitTester.h"
#include "CudaMath.h"

namespace Render
{
	namespace Kernel
	{
		__device__ HitTester::HitTester(
			HittableContext const &context) noexcept :
			__context{ context }
		{ }

		__device__ HitResult HitTester::hit(
			size_t const hittableHandle,
			Ray const &ray) const noexcept
		{
			auto const &header{ __context.pHeaders[hittableHandle] };
			switch (header.type)
			{
				case HittableType::SPHERE:
				{
					auto const pSphere
					{
						reinterpret_cast<SphereContent const *>(
							__context.pContents + header.offset)
					};

					return __hit(*pSphere, ray);
				}
			}

			return { };
		}

		__device__ HitResult HitTester::__hit(
			SphereContent const &content,
			Ray const &ray) noexcept
		{
			float3 const &rayOrigin		{ ray.getOrigin() };
			float3 const &rayDir		{ ray.getDirection() };

			float3 const rayOrigin_center{ rayOrigin - content.center };

			float const A{ 1.0f /*lengthSq(rayDir)*/ };
			float const B{ 2.0f * dot(rayDir, rayOrigin_center) };
			float const C{ lengthSq(rayOrigin_center) - (content.radius * content.radius) };

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
					retVal.normal = normalize(retVal.spot - content.center);
				}
			}

			return retVal;
		}
	}
}