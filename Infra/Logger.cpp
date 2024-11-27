#include "Logger.h"
#include <sstream>
#include <chrono>
#include <format>

namespace Infra
{
	void Logger::emplaceImpl(
		std::shared_ptr<Impl> pImpl) noexcept
	{
		__getInstance().__emplaceImpl(std::move(pImpl));
	}

	void Logger::log(
		Severity const severity,
		std::string const message) noexcept
	{
		__getInstance().__log(severity, message);
	}

	void Logger::__emplaceImpl(
		std::shared_ptr<Impl> &&pImpl) noexcept
	{
		std::lock_guard lock{ __mutex };

		__pImpl = std::move(pImpl);
		if (!__pImpl)
			return;

		for (auto &logMessage : __logBuffer)
			__pImpl->log(std::move(logMessage));

		__logBuffer.clear();
	}

	void Logger::__log(
		Severity const severity,
		std::string const &message) noexcept
	{
		auto logMessage{ __makeLogMessage(severity, message) };

		std::lock_guard lock{ __mutex };

		if (__pImpl)
			__pImpl->log(std::move(logMessage));
		else
			__logBuffer.emplace_back(std::move(logMessage));
	}

	Logger &Logger::__getInstance() noexcept
	{
		Logger static instance;
		return instance;
	}

	std::string Logger::__makeLogMessage(
		Severity const severity,
		std::string const &message) noexcept
	{
		std::ostringstream oss;
		oss << std::format(
			"[{}][{}] {}",
			__getCurrentTimeStr(),
			__getSeverityStrOf(severity),
			message);

		return oss.str();
	}

	std::string Logger::__getCurrentTimeStr() noexcept
	{
		using namespace std::chrono;

		const auto localNow{ current_zone()->to_local(system_clock::now()) };
		return std::format("{:%y-%m-%d %H:%M:%OS}", localNow);
	}

	constexpr char const *Logger::__getSeverityStrOf(
		Severity const severity) noexcept
	{
		switch (severity)
		{
			case Severity::FATAL: return "FATAL";
			case Severity::WARNING: return "WARNING";
			case Severity::INFO: return "INFO";
			case Severity::VERBOSE: return "VERBOSE";
		};

		return "Unknown";
	}
}