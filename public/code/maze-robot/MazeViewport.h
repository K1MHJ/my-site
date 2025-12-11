#pragma once

#include "MazeTypes.h"

class MazeMap;

class MazeViewport
{
public:
    MazeViewport();

    void SetSizeInCells(const Int2& size) noexcept;
    Int2 GetSizeInCells() const noexcept;

    void SetOriginCell(const Int2& origin) noexcept;
    Int2 GetOriginCell() const noexcept;

    // 에이전트 위치를 기준으로, 미로 범위 안에서 중심 이동
    void CenterOn(const Int2& mazeCell, const MazeMap& map);

private:
    Int2 m_origin{}; // 미로 좌표에서의 좌상단 셀
    Int2 m_size{};   // 화면에 보이는 셀 수 (w, h)
};
