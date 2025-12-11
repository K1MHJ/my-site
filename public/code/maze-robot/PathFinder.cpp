// PathFinder.cpp
#include "pch.h"
#include "PathFinder.h"
#include "MazeAgent.h"
#include "LocalMap.h"
#include <vector>

PathFinder::PathFinder()
{
}

Int2 PathFinder::DecideNextStep(const MazeAgent& agent,
    const LocalMap& localMap)
{
    Int2 current = agent.GetPosition();
    Int2 origin = localMap.GetViewOriginInMaze();

    // 에이전트의 로컬 좌표
    Int2 localPos{ current.x - origin.x, current.y - origin.y };
    Int2 size = localMap.GetSize();

    auto isFree = [&](int lx, int ly) -> bool
        {
            if (lx < 0 || lx >= size.x || ly < 0 || ly >= size.y)
                return false;

            auto cell = localMap.GetCell(lx, ly);
            return (cell == LocalMap::CellType::Free);
        };

    std::vector<Int2> candidates;
    candidates.reserve(4);

    // 4방향: 상하좌우
    const Int2 dirs[4] = {
        { 0, -1 }, { 0, 1 }, { -1, 0 }, { 1, 0 }
    };

    // 1순위: Free && 미방문
    for (const auto& d : dirs)
    {
        Int2 lp{ localPos.x + d.x, localPos.y + d.y };
        if (!isFree(lp.x, lp.y))
            continue;

        Int2 global{ origin.x + lp.x, origin.y + lp.y };
        if (!agent.HasVisited(global))
        {
            return global;
        }
    }

    // 2순위: Free 아무거나
    for (const auto& d : dirs)
    {
        Int2 lp{ localPos.x + d.x, localPos.y + d.y };
        if (!isFree(lp.x, lp.y))
            continue;

        Int2 global{ origin.x + lp.x, origin.y + lp.y };
        return global;
    }

    // 이동 불가 → 제자리
    return current;
}
