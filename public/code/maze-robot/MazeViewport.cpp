// MazeViewport.cpp
#include "pch.h"
#include "MazeViewport.h"
#include "MazeMap.h"
#include <algorithm>

#undef max
#undef min

MazeViewport::MazeViewport()
{
}

void MazeViewport::SetSizeInCells(const Int2& size) noexcept
{
    m_size = size;
}

Int2 MazeViewport::GetSizeInCells() const noexcept
{
    return m_size;
}

void MazeViewport::SetOriginCell(const Int2& origin) noexcept
{
    m_origin = origin;
}

Int2 MazeViewport::GetOriginCell() const noexcept
{
    return m_origin;
}

void MazeViewport::CenterOn(const Int2& mazeCell, const MazeMap& map)
{
    const int mapW = map.GetWidth();
    const int mapH = map.GetHeight();

    Int2 newOrigin;
    newOrigin.x = mazeCell.x - m_size.x / 2;
    newOrigin.y = mazeCell.y - m_size.y / 2;

    // 범위 클램프
    if (newOrigin.x < 0)
        newOrigin.x = 0;
    if (newOrigin.y < 0)
        newOrigin.y = 0;

    int maxOriginX = std::max(0, mapW - m_size.x);
    int maxOriginY = std::max(0, mapH - m_size.y);

    if (newOrigin.x > maxOriginX)
        newOrigin.x = maxOriginX;
    if (newOrigin.y > maxOriginY)
        newOrigin.y = maxOriginY;

    m_origin = newOrigin;
}
