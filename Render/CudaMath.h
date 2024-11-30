#pragma once

#include <cuda_runtime.h>

namespace Render
{
	namespace Kernel
	{
		__device__ float3 operator+(
			float3 const &lhs, float3 const &rhs) noexcept;

		__device__ float3 operator-(
			float3 const &lhs, float3 const &rhs) noexcept;

		__device__ float3 operator*(
			float3 const &lhs, float const rhs) noexcept;

		__device__ float3 operator*(
			float const lhs, float3 const &rhs) noexcept;

		__device__ float3 operator/(
			float3 const &lhs, float const rhs) noexcept;

		__device__ float3 &operator+=(
			float3 &lhs, float3 const &rhs) noexcept;

		__device__ float3 &operator-=(
			float3 &lhs, float3 const &rhs) noexcept;

		__device__ float3 &operator*=(
			float3 &lhs, float const rhs) noexcept;

		__device__ float3 &operator/=(
			float3 &lhs, float const rhs) noexcept;

		__device__ float dot(
			float3 const &lhs, float3 const &rhs) noexcept;

		__device__ float3 cross(
			float3 const &lhs, float3 const &rhs) noexcept;

		__device__ float3 normalize(
			float3 const &vec) noexcept;

		__device__ float length(
			float3 const &vec) noexcept;

		__device__ float lengthSq(
			float3 const &vec) noexcept;
	}
}