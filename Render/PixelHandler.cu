#include "PixelHandler.h"

namespace Render
{
	namespace Kernel
	{
		__device__ PixelHandler::PixelHandler(
			uint32_t const gidX,
			uint32_t const gidY,
			SurfaceContext const &surfaceContext) noexcept :
			__gidX				{ gidX },
			__gidY				{ gidY },
			__surfaceContext	{ surfaceContext }
		{}

		__device__ bool PixelHandler::isValid() const noexcept
		{
			if (!__surfaceContext.surface)
				return false;

			if (__gidX >= __surfaceContext.width)
				return false;

			if (__gidY >= __surfaceContext.height)
				return false;

			return true;
		}

		__device__ void PixelHandler::set(
			uchar4 const &value)
		{
			surf2Dwrite(value, __surfaceContext.surface, __gidX * sizeof(uchar4), __gidY);
		}

		__device__ void PixelHandler::set(
			float4 const &value)
		{
			uchar4 uValue{ };
			uValue.x = static_cast<uint8_t>(value.x * 255.0f);
			uValue.y = static_cast<uint8_t>(value.y * 255.0f);
			uValue.z = static_cast<uint8_t>(value.z * 255.0f);
			uValue.w = static_cast<uint8_t>(value.w * 255.0f);

			set(uValue);
		}
	}
}