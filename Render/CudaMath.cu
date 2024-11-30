#include "CudaMath.h"

namespace Render
{
	namespace Kernel
	{
		__device__ float3 operator+(
			float3 const &lhs, float3 const &rhs) noexcept
		{
			return float3{ lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z };
		}

		__device__ float3 operator-(
			float3 const &lhs, float3 const &rhs) noexcept
		{
			return float3{ lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z };
		}

		__device__ float3 operator*(
			float3 const &lhs, float const rhs) noexcept
		{
			return float3{ lhs.x * rhs, lhs.y * rhs, lhs.z * rhs };
		}

		__device__ float3 operator*(
			float const lhs, float3 const &rhs) noexcept
		{
			return (rhs * lhs);
		}

		__device__ float3 operator/(
			float3 const &lhs, float const rhs) noexcept
		{
			float const rhsInv{ 1.0f / rhs };
			return (lhs * rhsInv);
		}

		__device__ float3 &operator+=(
			float3 &lhs, float3 const &rhs) noexcept
		{
			lhs.x += rhs.x;
			lhs.y += rhs.y;
			lhs.z += rhs.z;
			return lhs;
		}

		__device__ float3 &operator-=(
			float3 &lhs, float3 const &rhs) noexcept
		{
			lhs.x -= rhs.x;
			lhs.y -= rhs.y;
			lhs.z -= rhs.z;
			return lhs;
		}

		__device__ float3 &operator*=(
			float3 &lhs, float const rhs) noexcept
		{
			lhs.x *= rhs;
			lhs.y *= rhs;
			lhs.z *= rhs;
			return lhs;
		}

		__device__ float3 &operator/=(
			float3 &lhs, float const rhs) noexcept
		{
			float const rhsInv{ 1.0f / rhs };
			return (lhs *= rhsInv);
		}

		__device__ float dot(
			float3 const &lhs, float3 const &rhs) noexcept
		{
			return ((lhs.x * rhs.x) + (lhs.y * rhs.y) + (lhs.z * rhs.z));
		}

		__device__ float3 cross(
			float3 const &lhs, float3 const &rhs) noexcept
		{
			float3 retVal{ };
			retVal.x = ((lhs.y * rhs.z) - (lhs.z * rhs.y));
			retVal.y = ((lhs.z * rhs.x) - (lhs.x * rhs.z));
			retVal.z = ((lhs.x * rhs.y) - (lhs.y * rhs.x));

			return retVal;
		}

		__device__ float3 normalize(
			float3 const &vec) noexcept
		{
			return (vec / length(vec));
		}

		__device__ float length(
			float3 const &vec) noexcept
		{
			return sqrtf(lengthSq(vec));
		}

		__device__ float lengthSq(
			float3 const &vec) noexcept
		{
			return dot(vec, vec);
		}
	}
}