#include "KernelLauncher.h"

namespace Render
{
	KernelLauncher::KernelLauncher()
	{
		auto &blockDim{ __launchContext.blockDim };
		blockDim.x = 16U;
		blockDim.y = 16U;
	}

	void KernelLauncher::temp_setSphere(
		float3 const &center,
		float const radius) noexcept
	{
		__resourceContext.hittable.asSphere(center, radius);
	}

	void KernelLauncher::launch() const
	{
		Kernel::launch(
			__viewport, __resourceContext,
			__surfaceContext, __launchContext);
	}
}