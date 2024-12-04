#pragma once

#include "../Infra/Unique.h"
#include <cuda_runtime.h>

namespace Cuda
{
	class UnifiedBuffer : public Infra::Unique
	{
	public:
		UnifiedBuffer() = default;
		virtual ~UnifiedBuffer() noexcept override;

		void reserve(
			size_t size);

		void resize(
			size_t size);

		uint8_t *append(
			size_t amount);

		template <typename $T>
		$T &append();

		template <typename $T>
		$T &at(
			size_t index) noexcept;

		template <typename $T>
		$T const &at(
			size_t index) const noexcept;

		constexpr void clear() noexcept;

		[[nodiscard]]
		constexpr size_t getSize() const noexcept;

		[[nodiscard]]
		constexpr uint8_t *getData() noexcept;

		[[nodiscard]]
		constexpr uint8_t const *getData() const noexcept;

		template <typename $T>
		[[nodiscard]]
		constexpr $T *getData() noexcept;

		template <typename $T>
		[[nodiscard]]
		constexpr $T const *getData() const noexcept;

	private:
		uint8_t *__pBuffer{ };
		size_t __capacity{ };
		size_t __size{ };

		void __grow(
			size_t reqSize);
	};

	template <typename $T>
	$T &UnifiedBuffer::append()
	{
		return *(reinterpret_cast<$T *>(append(sizeof($T))));
	}

	template <typename $T>
	$T &UnifiedBuffer::at(
		size_t const index) noexcept
	{
		auto const pCasted{ reinterpret_cast<$T *>(__pBuffer) };
		return pCasted[index];
	}

	template <typename $T>
	$T const &UnifiedBuffer::at(
		size_t const index) const noexcept
	{
		auto const pCasted{ reinterpret_cast<$T const *>(__pBuffer) };
		return pCasted[index];
	}

	constexpr void UnifiedBuffer::clear() noexcept
	{
		__size = 0ULL;
	}

	constexpr size_t UnifiedBuffer::getSize() const noexcept
	{
		return __size;
	}

	constexpr uint8_t *UnifiedBuffer::getData() noexcept
	{
		return __pBuffer;
	}

	constexpr uint8_t const *UnifiedBuffer::getData() const noexcept
	{
		return __pBuffer;
	}

	template <typename $T>
	constexpr $T *UnifiedBuffer::getData() noexcept
	{
		return reinterpret_cast<$T *>(__pBuffer);
	}

	template <typename $T>
	constexpr $T const *UnifiedBuffer::getData() const noexcept
	{
		return reinterpret_cast<$T *>(__pBuffer);
	}
}