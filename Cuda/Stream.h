#pragma once

#include "Event.h"
#include "UnifiedBuffer.h"

namespace Cuda
{
	class Stream : public Infra::Handle<cudaStream_t>
	{
	public:
		Stream();
		virtual ~Stream() noexcept override;

		void memcpy(
			void *dst,
			const void *src,
			size_t count,
			cudaMemcpyKind kind);

		void recordEvent(
			Event &event);

		[[nodiscard]]
		cudaError_t queryEvent(
			Event &event);

		void syncEvent(
			Event &event);

		void prefetch(
			UnifiedBuffer const &buffer,
			size_t from,
			size_t count);

		void prefetch(
			UnifiedBuffer const &buffer);

		void sync();
	};
}