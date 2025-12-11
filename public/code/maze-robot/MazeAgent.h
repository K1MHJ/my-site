#pragma once

#include <vector>
#include "MazeTypes.h"

class MazeAgent
{
public:
    MazeAgent();

    void SetPosition(const Int2& pos) noexcept;
    Int2 GetPosition() const noexcept;

    // 단순 방문 기록 (나중에 알고리즘 고도화 가능)
    void ClearVisited();
    void AddVisited(const Int2& cell);
    bool HasVisited(const Int2& cell) const;

private:
    Int2 m_position{};
    std::vector<Int2> m_visited;
};
