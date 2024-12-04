#pragma once

#include <cstdint>
#include "Ray.h"

namespace Render
{
	namespace Kernel
	{
		enum class HittableType
		{
			UNKNOWN = -1,
			SPHERE
		};

		struct HittableHeader
		{
		public:
			HittableType type{ HittableType::UNKNOWN };
			size_t offset{ };
		};

		struct SphereContent
		{
		public:
			float3 center;
			float radius;
		};

		struct HittableContext
		{
		public:
			HittableHeader *pHeaders{ };
			uint8_t *pContents{ };
		};
	}
}