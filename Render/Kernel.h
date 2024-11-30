#pragma once

#include "KernelContext.h"

namespace Render
{
	namespace Kernel
	{
		void launch(
			Viewport const &viewport,
			RenderContext const &resourceContext,
			SurfaceContext const &surfaceContext,
			LaunchContext const &launchContext);
	}
}