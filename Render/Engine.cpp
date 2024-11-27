#include "Engine.h"

namespace Render
{
	void Engine::render(
		RenderTarget &renderTarget)
	{
		renderTarget.draw(__engineContext);
	}
}