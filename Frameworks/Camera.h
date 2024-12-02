#pragma once

#include "../Infra/Stateful.h"
#include "../Render/Viewport.h"
#include "Transform.h"

namespace Frx
{
	class Camera : public Infra::Stateful<Camera>
	{
	public:
		Camera() noexcept;

		[[nodiscard]]
		constexpr float getFocalLength() const noexcept;
		void setFocalLength(
			float length);
		
		[[nodiscard]]
		constexpr float getAspectRatio() const noexcept;
		void setAspectRatio(
			float ratio);

		[[nodiscard]]
		constexpr float getFovY() const noexcept;
		void setFovY(
			float fovY);

		[[nodiscard]]
		constexpr Render::Kernel::Viewport const &getViewport() const noexcept;

		[[nodiscard]]
		constexpr Transform &getTransform() noexcept;

		[[nodiscard]]
		constexpr Transform const &getTransform() const noexcept;

	protected:
		virtual void _onValidate() override;

	private:
		static constexpr float __DEFAULT_FOV_Y{ glm::pi<float>() / 4.0f };

		Transform __transform;

		float __focalLength	{ 1.0f };
		float __aspectRatio	{ 1.0f };

		float __fovY		{ __DEFAULT_FOV_Y };
		float __tanHalfFovY	{ glm::tan(__DEFAULT_FOV_Y * 0.5f) };

		Render::Kernel::Viewport __viewport;

		Infra::EventListenerPtr<Transform *> __pTransformInvalidateListener;

		void __onTransformInvalidated() noexcept;
	};

	constexpr float Camera::getFocalLength() const noexcept
	{
		return __focalLength;
	}

	constexpr float Camera::getAspectRatio() const noexcept
	{
		return __aspectRatio;
	}

	constexpr float Camera::getFovY() const noexcept
	{
		return __fovY;
	}

	constexpr Render::Kernel::Viewport const &Camera::getViewport() const noexcept
	{
		return __viewport;
	}

	constexpr Transform &Camera::getTransform() noexcept
	{
		return __transform;
	}

	constexpr Transform const &Camera::getTransform() const noexcept
	{
		return __transform;
	}
}