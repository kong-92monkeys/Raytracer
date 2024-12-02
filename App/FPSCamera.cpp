#include "pch.h"
#include "FPSCamera.h"

void FPSCamera::setPosition(
	float const x,
	float const y,
	float const z) noexcept
{
	auto &position{ __camera.getTransform().getPosition() };
	position.set(x, y, z);
	_invalidate();
}

void FPSCamera::moveLocalX(
	float const amount) noexcept
{
	auto &transform		{ __camera.getTransform() };
	auto &position		{ transform.getPosition() };
	auto &orientation	{ transform.getOrientation() };

	orientation.validate();
	glm::vec3 const cameraRight{ orientation.getMatrix()[0] };

	position.add(cameraRight * amount);
	_invalidate();
}

void FPSCamera::moveLocalY(
	float const amount) noexcept
{
	auto &transform		{ __camera.getTransform() };
	auto &position		{ transform.getPosition() };
	auto &orientation	{ transform.getOrientation() };

	orientation.validate();
	glm::vec3 const cameraUp{ orientation.getMatrix()[1] };

	position.add(cameraUp * amount);
	_invalidate();
}

void FPSCamera::moveLocalZ(
	float const amount) noexcept
{
	auto &transform		{ __camera.getTransform() };
	auto &position		{ transform.getPosition() };
	auto &orientation	{ transform.getOrientation() };

	orientation.validate();
	glm::vec3 const cameraForward{ orientation.getMatrix()[2] };

	position.add(cameraForward * amount);
	_invalidate();
}

void FPSCamera::pitch(
	float const amount) noexcept
{
	__cameraPitch += amount;
	__setOrientation();
	_invalidate();
}

void FPSCamera::yaw(
	float const amount) noexcept
{
	__cameraYaw += amount;
	__setOrientation();
	_invalidate();
}

void FPSCamera::setFocalLength(
	float const length)
{
	__camera.setFocalLength(length);
	_invalidate();
}

void FPSCamera::setAspectRatio(
	float const ratio)
{
	__camera.setAspectRatio(ratio);
	_invalidate();
}

void FPSCamera::setFovY(
	float const fovY)
{
	__camera.setFovY(fovY);
	_invalidate();
}

void FPSCamera::_onValidate()
{
	__camera.validate();
}

void FPSCamera::__setOrientation() noexcept
{
	auto &orientation{ __camera.getTransform().getOrientation() };
	
	auto const yaw				{ glm::angleAxis(__cameraYaw, glm::vec3{ 0.0f, 1.0f, 0.0f }) };
	auto const yawMatrix		{ glm::mat4_cast(yaw) };

	glm::vec3 const cameraRight	{ yawMatrix[0] };
	auto const pitch			{ glm::angleAxis(__cameraPitch, cameraRight) };

	orientation.set(pitch * yaw);
}