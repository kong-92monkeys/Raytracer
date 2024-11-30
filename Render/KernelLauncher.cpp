#include "KernelLauncher.h"

namespace Render
{
	KernelLauncher::KernelLauncher()
	{
		auto &blockDim{ __launchContext.blockDim };
		blockDim.x = 16U;
		blockDim.y = 16U;
	}

	void KernelLauncher::launch(
		cudaSurfaceObject_t const surface) const
	{
		Kernel::SurfaceContext const surfaceContext
		{
			surface,
			__surfaceWidth,
			__surfaceHeight
		};

		Kernel::launch(__resourceContext, surfaceContext, __launchContext);
	}
}