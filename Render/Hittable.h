#pragma once

#include "HittableContext.h"
#include "../Infra/GLM.h"
#include "../Infra/Stateful.h"
#include "../Infra/IdAllocator.h"
#include "../Infra/Handle.h"
#include "../Cuda/UnifiedBuffer.h"
#include <unordered_map>

namespace Render
{
	class HittableManager : public Infra::Stateful<HittableManager>
	{
	public:
		[[nodiscard]]
		size_t createSphere(
			glm::vec3 const &center,
			float radius) noexcept;

		void destroyHittable(
			size_t handle);

		[[nodiscard]]
		constexpr Cuda::UnifiedBuffer const &getHeaders() const noexcept;

		[[nodiscard]]
		constexpr Cuda::UnifiedBuffer const &getContents() const noexcept;

	protected:
		virtual void _onValidate() override;

	private:
		std::unordered_map<Kernel::HittableType, Infra::IdAllocator<size_t>> __idAllocators;

		Cuda::UnifiedBuffer __headers;
		Cuda::UnifiedBuffer __contents;

		[[nodiscard]]
		size_t __allocate(
			Kernel::HittableType hittableType) noexcept;

		template <typename $T>
		[[nodiscard]]
		constexpr $T *__getContextOf(
			size_t handle) noexcept;

		[[nodiscard]]
		static constexpr size_t __getContentSizeOf(
			Kernel::HittableType hittableType) noexcept;
	};

	class Hittable : public Infra::Handle<size_t>
	{
	public:
		Hittable(
			HittableManager &manager,
			size_t handle) noexcept;

		virtual ~Hittable() noexcept override;

	private:
		HittableManager &__manager;
	};

	constexpr Cuda::UnifiedBuffer const &HittableManager::getHeaders() const noexcept
	{
		return __headers;
	}

	constexpr Cuda::UnifiedBuffer const &HittableManager::getContents() const noexcept
	{
		return __contents;
	}

	template <typename $T>
	constexpr $T *HittableManager::__getContextOf(
		size_t const handle) noexcept
	{
		auto const &header	{ __headers.at<Kernel::HittableHeader>(handle) };
		auto const offset	{ header.offset };
		return reinterpret_cast<$T *>(__contents.getData() + offset);
	}

	constexpr size_t HittableManager::__getContentSizeOf(
		Kernel::HittableType const hittableType) noexcept
	{
		switch (hittableType)
		{
			case Kernel::HittableType::SPHERE:
				return sizeof(Kernel::SphereContent);
		}

		return 0ULL;
	}
}