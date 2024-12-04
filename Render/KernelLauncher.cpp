#include "KernelLauncher.h"

namespace Render
{
	KernelLauncher::KernelLauncher()
	{
		auto &blockDim{ __launchContext.blockDim };
		blockDim.x = 16U;
		blockDim.y = 16U;
	}

	void KernelLauncher::launch() const
	{
		Kernel::launch(
			__viewport, __renderContext,
			__surfaceContext, __launchContext);
	}
}