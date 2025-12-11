#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include "MazeTypes.h"

class MazeMap
{
public:
    MazeMap();

    void Clear();

    bool LoadFromFile(const std::string& path);
    bool SaveToFile(const std::string& path) const;

    void Resize(int width, int height);
    int  GetWidth() const noexcept;
    int  GetHeight() const noexcept;

    // 0 <= x < width, 0 <= y < height 가정
    bool IsWall(int x, int y) const;
    void SetWall(int x, int y, bool wall);

private:
    int m_width{ 0 };
    int m_height{ 0 };
    // 0 = 빈 칸, 1 = 벽 (필요하면 enum으로 확장 가능)
    std::vector<std::uint8_t> m_cells;

    int Index(int x, int y) const;
};
