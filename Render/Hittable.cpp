#include "Hittable.h"

namespace Render
{
	size_t HittableManager::createSphere(
		glm::vec3 const &center,
		float const radius) noexcept
	{
		size_t const handle	{ __allocate(Kernel::HittableType::SPHERE) };
		auto const pContent	{ __getContextOf<Kernel::SphereContent>(handle) };

		std::memcpy(&(pContent->center), &center, sizeof(float3));
		pContent->radius = radius;

		return handle;
	}

	void HittableManager::_onValidate()
	{

	}

	size_t HittableManager::__allocate(
		Kernel::HittableType const hittableType) noexcept
	{
		size_t const handle{ __idAllocators[hittableType].allocate() };

		if (handle >= __headers.getSize())
		{
			auto &header	{ __headers.append<Kernel::HittableHeader>() };
			header.type		= hittableType;
			header.offset	= __contents.getSize();

			__contents.resize(
				__contents.getSize() +
				__getContentSizeOf(hittableType));
		}

		return handle;
	}

	void HittableManager::destroyHittable(
		size_t const handle)
	{
		auto const &header		{ __headers.at<Kernel::HittableHeader>(handle) };
		auto const hittableType	{ header.type };
		__idAllocators[hittableType].free(handle);
	}

	Hittable::Hittable(
		HittableManager &manager,
		size_t const handle) noexcept :
		Handle{ handle }, __manager{ manager }
	{}

	Hittable::~Hittable() noexcept
	{
		__manager.destroyHittable(getHandle());
	}
}