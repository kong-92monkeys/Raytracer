#pragma once

#include "../D3DHelper/Swapchain.h"

namespace Cuda
{
	class Swapchain
	{
	public:
		Swapchain(
			HWND const hWnd,
			UINT const width,
			UINT const height);

		virtual ~Swapchain() noexcept;

		void resize(
			UINT const width,
			UINT const height);
	};
}