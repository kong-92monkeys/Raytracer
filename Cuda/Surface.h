#pragma once

#include "../Infra/Handle.h"
#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>

namespace Cuda
{
	class Surface : public Infra::Handle<cudaSurfaceObject_t>
	{
	public:
		Surface(
			ID3D11Texture2D *pImage,
			cudaGraphicsRegisterFlags registerFlags);

		virtual ~Surface() noexcept;

		void map();
		void unmap();

	private:
		cudaGraphicsResource_t __interopHandle{ };

		void __registerInteropHandle(
			ID3D11Texture2D *pImage,
			cudaGraphicsRegisterFlags registerFlags);

		void __createHandle();
	};
}