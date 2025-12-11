// ImageBuffer.cpp
#include "pch.h"
#include "ImageBuffer.h"
#include <cstring>

ImageBuffer::ImageBuffer(int width, int height)
{
    Resize(width, height);
}

void ImageBuffer::Resize(int width, int height)
{
    if (width <= 0 || height <= 0)
    {
        m_width = m_height = 0;
        m_data.clear();
        return;
    }

    m_width = width;
    m_height = height;
    m_data.assign(static_cast<size_t>(width * height), 0);
}

std::uint8_t* ImageBuffer::RowPtr(int y) noexcept
{
    if (y < 0 || y >= m_height)
        return nullptr;
    return m_data.data() + static_cast<size_t>(y * m_width);
}

const std::uint8_t* ImageBuffer::RowPtr(int y) const noexcept
{
    if (y < 0 || y >= m_height)
        return nullptr;
    return m_data.data() + static_cast<size_t>(y * m_width);
}
