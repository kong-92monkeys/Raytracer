#pragma once

#include "Kernel.h"

namespace Render
{
	class KernelLauncher
	{
	public:
		KernelLauncher();

		void temp_setSphere(
			float3 const &center,
			float const radius) noexcept;

		constexpr void setViewport(
			Kernel::Viewport const &viewport) noexcept;

		constexpr void setSurface(
			cudaSurfaceObject_t surface) noexcept;

		constexpr void setSurfaceExtent(
			uint32_t width,
			uint32_t height) noexcept;

		constexpr void setStream(
			cudaStream_t stream) noexcept;

		void launch() const;

	private:
		Kernel::Viewport __viewport;
		Kernel::RenderContext __resourceContext;
		Kernel::SurfaceContext __surfaceContext;
		Kernel::LaunchContext __launchContext;

		constexpr void __resolveBlockSize() noexcept;
	};

	constexpr void KernelLauncher::setViewport(
		Kernel::Viewport const &viewport) noexcept
	{
		__viewport = viewport;
	}

	constexpr void KernelLauncher::setSurface(
		cudaSurfaceObject_t const surface) noexcept
	{
		__surfaceContext.surface = surface;
	}

	constexpr void KernelLauncher::setSurfaceExtent(
		uint32_t const width,
		uint32_t const height) noexcept
	{
		__surfaceContext.width		= width;
		__surfaceContext.height		= height;
		__resolveBlockSize();
	}

	constexpr void KernelLauncher::setStream(
		cudaStream_t const stream) noexcept
	{
		__launchContext.stream = stream;
	}

	constexpr void KernelLauncher::__resolveBlockSize() noexcept
	{
		auto &gridDim			{ __launchContext.gridDim };
		auto const &blockDim	{ __launchContext.blockDim };

		gridDim.x = (__surfaceContext.width / blockDim.x);
		gridDim.x += ((__surfaceContext.width % blockDim.x) ? 1 : 0);

		gridDim.y = (__surfaceContext.height / blockDim.y);
		gridDim.y += ((__surfaceContext.height % blockDim.y) ? 1 : 0);
	}
}