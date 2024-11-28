#include "KernelLauncher.h"

namespace Render
{
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