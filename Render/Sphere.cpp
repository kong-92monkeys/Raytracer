#include "Sphere.h"

namespace Render
{
	Sphere::Sphere(
		HittableManager &manager,
		glm::vec3 const &center,
		float const radius) noexcept :
		Hittable{ manager, manager.createSphere(center, radius) }
	{}
}