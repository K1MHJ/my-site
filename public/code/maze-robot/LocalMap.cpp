// LocalMap.cpp
#include "pch.h"
#include "LocalMap.h"
#include <stdexcept>

LocalMap::LocalMap()
{
}

void LocalMap::Resize(const Int2& size)
{
    if (size.x <= 0 || size.y <= 0)
    {
        m_size = { 0, 0 };
        m_cells.clear();
        return;
    }

    m_size = size;
    m_cells.assign(static_cast<size_t>(size.x * size.y), CellType::Unknown);
}

Int2 LocalMap::GetSize() const noexcept
{
    return m_size;
}

void LocalMap::SetViewOriginInMaze(const Int2& origin) noexcept
{
    m_viewOriginInMaze = origin;
}

Int2 LocalMap::GetViewOriginInMaze() const noexcept
{
    return m_viewOriginInMaze;
}

int LocalMap::Index(int x, int y) const
{
    return y * m_size.x + x;
}

void LocalMap::SetCell(int x, int y, CellType type)
{
    if (x < 0 || x >= m_size.x || y < 0 || y >= m_size.y)
        return;

    m_cells[Index(x, y)] = type;
}

LocalMap::CellType LocalMap::GetCell(int x, int y) const
{
    if (x < 0 || x >= m_size.x || y < 0 || y >= m_size.y)
        return CellType::Unknown;

    return m_cells[Index(x, y)];
}
