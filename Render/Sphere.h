#pragma once

#include "Hittable.h"

namespace Render
{
	class Sphere : public Hittable
	{
	public:
		Sphere(
			HittableManager &manager,
			glm::vec3 const &center,
			float radius) noexcept;

		virtual ~Sphere() noexcept override = default;
	};
}
