#include "AssetManager.h"
#include <fstream>
#include <sstream>

namespace Infra
{
	AssetManager::AssetManager(
		std::string_view const &rootPath) noexcept
	{
		setRootPath(rootPath);
	}

	void AssetManager::setRootPath(
		std::string_view const &rootPath) noexcept
	{
		std::lock_guard lock{ __mutex };
		__rootPath = rootPath;
	}

	bool AssetManager::exists(
		std::string_view const &path) noexcept
	{
		std::filesystem::path filePath;
		{
			std::lock_guard lock{ __mutex };
			filePath = (__rootPath / path);
		}
		
		return std::filesystem::exists(filePath);
	}

	std::string AssetManager::readString(
		std::string_view const &path)
	{
		std::filesystem::path filePath;
		{
			std::lock_guard lock{ __mutex };
			filePath = (__rootPath / path);
		}

		std::ifstream fin{ filePath };
		if (!fin)
			throw std::runtime_error{ std::format("Cannot open file: {}", filePath.string()) };

		std::ostringstream oss;
		oss << fin.rdbuf();

		return oss.str();
	}

	std::vector<std::byte> AssetManager::readBinary(
		std::string_view const &path)
	{
		std::filesystem::path filePath;
		{
			std::lock_guard lock{ __mutex };
			filePath = (__rootPath / path);
		}

		std::ifstream fin{ filePath, std::ios_base::binary };
		if (!fin)
			throw std::runtime_error{ std::format("Cannot open file: {}", filePath.string()) };

		fin.unsetf(std::ios_base::skipws);

		fin.seekg(0, std::ios::end);
		auto const memSize{ fin.tellg() };
		fin.seekg(0, std::ios::beg);

		std::vector<std::byte> retVal;
		retVal.resize(memSize);

		fin.read(reinterpret_cast<char *>(retVal.data()), memSize);
		return retVal;
	}
}