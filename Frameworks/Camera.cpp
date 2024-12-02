#include "Camera.h"

namespace Frx
{
	Camera::Camera() noexcept
	{
		__pTransformInvalidateListener =
			Infra::EventListener<Transform *>::bind(
				&Camera::__onTransformInvalidated, this);

		__transform.getInvalidateEvent() += __pTransformInvalidateListener;
	}

	void Camera::setFocalLength(
		float const length)
	{
		__focalLength = length;
		_invalidate();
	}

	void Camera::setAspectRatio(
		float const ratio)
	{
		__aspectRatio = ratio;
		_invalidate();
	}

	void Camera::setFovY(
		float const fovY)
	{
		__fovY = fovY;
		__tanHalfFovY = glm::tan(fovY * 0.5f);
		_invalidate();
	}

	void Camera::_onValidate()
	{
		__transform.validate();

		auto const &rayOrigin			{ __transform.getPosition().get() };

		auto const &basis				{ __transform.getOrientation().getMatrix() };
		glm::vec3 const basisX			{ basis[0] };
		glm::vec3 const basisY			{ basis[1] };
		glm::vec3 const basisZ			{ basis[2] };

		glm::vec3 const right			{ basisX };
		glm::vec3 const down			{ -basisY };
		glm::vec3 const forward			{ -basisZ };

		float const width				{ __focalLength * __tanHalfFovY };
		float const height				{ width / __aspectRatio };

		glm::vec3 const viewportCenter	{ rayOrigin + (__focalLength * forward) };
		glm::vec3 const viewportOrigin	{ viewportCenter - (((width * 0.5f) * right) + ((height * 0.5f) * down)) };

		std::memcpy(&(__viewport.rayOrigin), &rayOrigin, sizeof(float3));
		std::memcpy(&(__viewport.viewportOrigin), &viewportOrigin, sizeof(float3));
		std::memcpy(&(__viewport.right), &right, sizeof(float3));
		std::memcpy(&(__viewport.down), &down, sizeof(float3));

		__viewport.width = width;
		__viewport.height = height;
	}

	void Camera::__onTransformInvalidated() noexcept
	{
		_invalidate();
	}
}