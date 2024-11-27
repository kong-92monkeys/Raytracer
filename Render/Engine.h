#pragma once

#include "RenderTarget.h"

namespace Render
{
	class Engine
	{
	public:
		void render(
			RenderTarget &renderTarget);

	private:
		Kernel::EngineContext __engineContext;
	};
}
