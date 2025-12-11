// MazeAgent.cpp
#include "pch.h"
#include "MazeAgent.h"

MazeAgent::MazeAgent()
{
}

void MazeAgent::SetPosition(const Int2& pos) noexcept
{
    m_position = pos;
}

Int2 MazeAgent::GetPosition() const noexcept
{
    return m_position;
}

void MazeAgent::ClearVisited()
{
    m_visited.clear();
}

void MazeAgent::AddVisited(const Int2& cell)
{
    m_visited.push_back(cell);
}

bool MazeAgent::HasVisited(const Int2& cell) const
{
    for (const auto& v : m_visited)
    {
        if (v.x == cell.x && v.y == cell.y)
            return true;
    }
    return false;
}
