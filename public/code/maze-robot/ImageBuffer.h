#pragma once

#include <vector>
#include <cstdint>

class ImageBuffer
{
public:
    ImageBuffer() = default;
    ImageBuffer(int width, int height);

    void Resize(int width, int height);

    int GetWidth() const noexcept { return m_width; }
    int GetHeight() const noexcept { return m_height; }

    std::uint8_t* Data() noexcept { return m_data.data(); }
    const std::uint8_t* Data() const noexcept { return m_data.data(); }

    std::uint8_t* RowPtr(int y) noexcept;
    const std::uint8_t* RowPtr(int y) const noexcept;

private:
    int m_width{ 0 };
    int m_height{ 0 };
    std::vector<std::uint8_t> m_data; // grayscale
};
