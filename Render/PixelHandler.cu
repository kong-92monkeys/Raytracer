#include "PixelHandler.h"

namespace Render
{
	namespace Kernel
	{
		__device__ PixelHandler::PixelHandler(
			uint32_t const gidX,
			uint32_t const gidY,
			cudaSurfaceObject_t const surface) noexcept :
			__gidX		{ gidX },
			__gidY		{ gidY },
			__surface	{ surface }
		{}

		__device__ bool PixelHandler::isValid(
			uint32_t const width,
			uint32_t const height) const noexcept
		{
			if (!__surface)
				return false;

			if (__gidX >= width)
				return false;

			if (__gidY >= height)
				return false;

			return true;
		}

		__device__ void PixelHandler::set(
			uchar4 const &value)
		{
			surf2Dwrite(value, __surface, __gidX * sizeof(uchar4), __gidY);
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