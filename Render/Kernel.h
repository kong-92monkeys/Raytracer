#pragma once

#include "KernelContext.h"

namespace Render
{
	namespace Kernel
	{
		void launch(
			ResourceContext const &resourceContext,
			SurfaceContext const &surfaceContext,
			LaunchContext const &launchContext);
	}
}