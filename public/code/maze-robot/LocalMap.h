#pragma once

#include <vector>
#include <cstdint>
#include "MazeTypes.h"

class LocalMap
{
public:
    enum class CellType : std::uint8_t
    {
        Unknown = 0,
        Free = 1,
        Wall = 2
    };

    LocalMap();

    void Resize(const Int2& size);
    Int2 GetSize() const noexcept;

    void SetViewOriginInMaze(const Int2& origin) noexcept;
    Int2 GetViewOriginInMaze() const noexcept;

    void SetCell(int x, int y, CellType type);
    CellType GetCell(int x, int y) const;

private:
    Int2 m_size{};             // 로컬 맵 크기 (셀 단위)
    Int2 m_viewOriginInMaze{}; // 이 로컬맵 좌상단이 전체 미로에서 어디인지
    std::vector<CellType> m_cells;

    int Index(int x, int y) const;
};
