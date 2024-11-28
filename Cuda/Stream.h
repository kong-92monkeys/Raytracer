#pragma once

#include "Event.h"

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

		void syncEvent(
			Event &event);

		void sync();
	};
}