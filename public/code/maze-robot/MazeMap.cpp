// MazeMap.cpp
#include "pch.h"
#include "MazeMap.h"
#include <fstream>
#include <stdexcept>

MazeMap::MazeMap()
{
}

void MazeMap::Clear()
{
    m_width = 0;
    m_height = 0;
    m_cells.clear();
}

void MazeMap::Resize(int width, int height)
{
    if (width <= 0 || height <= 0)
    {
        Clear();
        return;
    }

    m_width = width;
    m_height = height;
    m_cells.assign(static_cast<size_t>(width * height), 0);
}

int MazeMap::GetWidth() const noexcept
{
    return m_width;
}

int MazeMap::GetHeight() const noexcept
{
    return m_height;
}

int MazeMap::Index(int x, int y) const
{
    return y * m_width + x;
}

bool MazeMap::IsWall(int x, int y) const
{
    if (x < 0 || x >= m_width || y < 0 || y >= m_height)
        return true; // 범위 밖은 벽 취급

    return (m_cells[Index(x, y)] != 0);
}

void MazeMap::SetWall(int x, int y, bool wall)
{
    if (x < 0 || x >= m_width || y < 0 || y >= m_height)
        return;

    m_cells[Index(x, y)] = wall ? 1 : 0;
}

// 아주 단순한 텍스트 포맷 예시:
// width height
// ####....
// #..#....
// ...
bool MazeMap::LoadFromFile(const std::string& path)
{
    std::ifstream ifs(path);
    if (!ifs)
        return false;

    int w, h;
    ifs >> w >> h;
    if (!ifs || w <= 0 || h <= 0)
        return false;

    Resize(w, h);

    std::string line;
    std::getline(ifs, line); // 나머지 라인으로 이동
    for (int y = 0; y < h; ++y)
    {
        std::getline(ifs, line);
        if (static_cast<int>(line.size()) < w)
            return false;

        for (int x = 0; x < w; ++x)
        {
            char c = line[x];
            bool wall = (c == '#' || c == '1');
            SetWall(x, y, wall);
        }
    }
    return true;
}

bool MazeMap::SaveToFile(const std::string& path) const
{
    std::ofstream ofs(path);
    if (!ofs)
        return false;

    ofs << m_width << " " << m_height << "\n";
    for (int y = 0; y < m_height; ++y)
    {
        for (int x = 0; x < m_width; ++x)
        {
            ofs << (IsWall(x, y) ? '#' : '.');
        }
        ofs << "\n";
    }
    return true;
}
