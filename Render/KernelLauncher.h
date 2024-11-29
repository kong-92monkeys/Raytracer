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

		constexpr void setGridSize(
			uint32_t x,
			uint32_t y) noexcept;

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

	constexpr void KernelLauncher::setGridSize(
		uint32_t const x,
		uint32_t const y) noexcept
	{
		__launchContext.gridSize.x = x;
		__launchContext.gridSize.y = y;
		__resolveBlockSize();
	}

	constexpr void KernelLauncher::setStream(
		cudaStream_t const stream) noexcept
	{
		__launchContext.stream = stream;
	}

	constexpr void KernelLauncher::__resolveBlockSize() noexcept
	{
		auto const &gridSize	{ __launchContext.gridSize };
		auto &blockSize			{ __launchContext.blockSize };

		blockSize.x = (__surfaceWidth / gridSize.x);
		blockSize.x += ((__surfaceWidth % gridSize.x) ? 1 : 0);
		blockSize.x = 4;

		blockSize.y = (__surfaceHeight / gridSize.y);
		blockSize.y += ((__surfaceHeight % gridSize.y) ? 1 : 0);
		blockSize.y = 4;
	}
}