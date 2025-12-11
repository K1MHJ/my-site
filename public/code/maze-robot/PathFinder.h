#pragma once

#include "MazeTypes.h"

class MazeAgent;
class LocalMap;

class PathFinder
{
public:
    PathFinder();

    // agent 위치 + LocalMap을 보고 다음으로 이동할 셀 결정
    Int2 DecideNextStep(const MazeAgent& agent,
        const LocalMap& localMap);
};
