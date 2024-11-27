#pragma once

#include <memory>
#include <functional>
#include "Unique.h"
#include "WeakReferenceSet.h"

namespace Infra
{
	template <typename ...$Args>
	class EventListener;

	template <typename ...$Args>
	using EventListenerPtr = std::shared_ptr<EventListener<$Args...>>;

	template <typename ...$Args>
	using EventCallback = std::function<void($Args const &...)>;

	template <typename ...$Args>
	class EventListener : public Unique
	{
	public:
		EventListener(EventCallback<$Args...> &&callback) noexcept;
		virtual ~EventListener() noexcept = default;

		void send(
			$Args const &...args) const;

		[[nodiscard]]
		static EventListenerPtr<$Args...> make(
			EventCallback<$Args...> &&callback) noexcept;

		template <typename ...$Params>
		[[nodiscard]]
		static EventListenerPtr<$Args...> bind(
			$Params &&...params) noexcept;

	private:
		const EventCallback<$Args...> __callbackFunc;
	};

	template <typename ...$Args>
	class EventView : public Unique
	{
	public:
		virtual void addListener(
			EventListenerPtr<$Args...> const &pListener) noexcept = 0;

		virtual void removeListener(
			EventListenerPtr<$Args...> const &pListener) = 0;

		EventView &operator+=(
			EventListenerPtr<$Args...> const &pListener) noexcept;

		EventView &operator-=(
			EventListenerPtr<$Args...> const &pListener);
	};

	template <typename ...$Args>
	class Event : public EventView<$Args...>
	{
	public:
		virtual void addListener(
			EventListenerPtr<$Args...> const &pListener) noexcept override;

		virtual void removeListener(
			EventListenerPtr<$Args...> const &pListener) override;

		void invoke(
			$Args const &...args);

	private:
		WeakReferenceSet<EventListener<$Args...>> __listeners;
	};

	template <typename ...$Args>
	EventListener<$Args...>::EventListener(
		EventCallback<$Args...> &&callback) noexcept :
		__callbackFunc	{ std::move(callback) }
	{}

	template <typename ...$Args>
	void EventListener<$Args...>::send(
		$Args const &...args) const
	{
		__callbackFunc(args...);
	}

	template <typename ...$Args>
	EventListenerPtr<$Args...> EventListener<$Args...>::make(
		EventCallback<$Args...> &&callback) noexcept
	{
		return std::make_shared<EventListener<$Args...>>(std::move(callback));
	}

	template <typename ...$Args>
	template <typename ...$Params>
	EventListenerPtr<$Args...> EventListener<$Args...>::bind(
		$Params &&...params) noexcept
	{
		return make(std::bind(std::forward<$Params>(params)...));
	}

	template <typename ...$Args>
	EventView<$Args...> &EventView<$Args...>::operator+=(
		EventListenerPtr<$Args...> const &pListener) noexcept
	{
		addListener(pListener);
		return *this;
	}

	template <typename ...$Args>
	EventView<$Args...> &EventView<$Args...>::operator-=(
		EventListenerPtr<$Args...> const &pListener)
	{
		removeListener(pListener);
		return *this;
	}

	template <typename ...$Args>
	void Event<$Args...>::addListener(
		EventListenerPtr<$Args...> const &pListener) noexcept
	{
		__listeners.emplace(pListener);
	}

	template <typename ...$Args>
	void Event<$Args...>::removeListener(
		EventListenerPtr<$Args...> const &pListener)
	{
		__listeners.erase(pListener);
	}

	template <typename ...$Args>
	void Event<$Args...>::invoke(
		$Args const &...args)
	{
		for (auto const &listener : __listeners)
			listener.send(args...);
	}
}