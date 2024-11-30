#pragma once

#include "Kernel.h"

namespace Render
{
	class KernelLauncher
	{
	public:
		constexpr void setSurfaceExtent(
			uint32_t width,
			uint32_t height) noexcept;

		constexpr void setStream(
			cudaStream_t stream) noexcept;

		void launch(
			cudaSurfaceObject_t surface) const;

	private:
		Kernel::ResourceContext __resourceContext;
		Kernel::LaunchContext __launchContext;

		uint32_t __surfaceWidth		{ 1U };
		uint32_t __surfaceHeight	{ 1U };

		constexpr void __resolveBlockSize() noexcept;
	};

	constexpr void KernelLauncher::setSurfaceExtent(
		uint32_t const width,
		uint32_t const height) noexcept
	{
		__surfaceWidth		= width;
		__surfaceHeight		= height;
		__resolveBlockSize();
	}

	constexpr void KernelLauncher::setStream(
		cudaStream_t const stream) noexcept
	{
		__launchContext.stream = stream;
	}

	constexpr void KernelLauncher::__resolveBlockSize() noexcept
	{
		auto &gridDim	{ __launchContext.gridDim };
		auto &blockDim	{ __launchContext.blockDim };

		blockDim.x = 16U;
		blockDim.y = 16U;

		gridDim.x = (__surfaceWidth / blockDim.x);
		gridDim.x += ((__surfaceWidth % blockDim.x) ? 1 : 0);

		gridDim.y = (__surfaceHeight / blockDim.y);
		gridDim.y += ((__surfaceHeight % blockDim.y) ? 1 : 0);
	}
}