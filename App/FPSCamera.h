#pragma once

#include "../Frameworks/Camera.h"

class FPSCamera : public Infra::Stateful<FPSCamera>
{
public:
	[[nodiscard]]
	constexpr glm::vec3 const &getPosition() const noexcept;
	void setPosition(
		float x,
		float y,
		float z) noexcept;

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

	void moveLocalX(
		float amount) noexcept;

	void moveLocalY(
		float amount) noexcept;

	void moveLocalZ(
		float amount) noexcept;

	void pitch(
		float amount) noexcept;

	void yaw(
		float amount) noexcept;

protected:
	virtual void _onValidate() override;

private:
	float __cameraPitch{ 0.0f };
	float __cameraYaw{ 0.0f };

	Frx::Camera __camera;

	void __setOrientation() noexcept;
};

constexpr glm::vec3 const &FPSCamera::getPosition() const noexcept
{
	return __camera.getTransform().getPosition().get();
}

constexpr float FPSCamera::getFocalLength() const noexcept
{
	return __camera.getFocalLength();
}

constexpr float FPSCamera::getAspectRatio() const noexcept
{
	return __camera.getAspectRatio();
}

constexpr float FPSCamera::getFovY() const noexcept
{
	return __camera.getFovY();
}

constexpr Render::Kernel::Viewport const &FPSCamera::getViewport() const noexcept
{
	return __camera.getViewport();
}