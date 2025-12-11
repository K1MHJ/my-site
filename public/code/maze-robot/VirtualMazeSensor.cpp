// VirtualMazeSensor.cpp
#include "pch.h"
#include "VirtualMazeSensor.h"
#include <algorithm>

VirtualMazeSensor::VirtualMazeSensor(const MazeMap* map)
    : m_map(map)
{
}

bool VirtualMazeSensor::Capture(const MazeViewport& viewport,
    ImageBuffer& image)
{
    if (!m_map)
        return false;

    Int2 vpOrigin = viewport.GetOriginCell();
    Int2 vpSize = viewport.GetSizeInCells();

    if (vpSize.x <= 0 || vpSize.y <= 0)
        return false;

    // ¿¹: ¼¿´ç 8x8 ÇÈ¼¿·Î ·»´õ
    const int cellPx = 8;
    const int imgW = vpSize.x * cellPx;
    const int imgH = vpSize.y * cellPx;

    image.Resize(imgW, imgH);

    for (int y = 0; y < vpSize.y; ++y)
    {
        for (int x = 0; x < vpSize.x; ++x)
        {
            int mx = vpOrigin.x + x;
            int my = vpOrigin.y + y;

            bool wall = m_map->IsWall(mx, my);

            // º® = ¾îµÎ¿î ÇÈ¼¿(50), ±æ = ¹àÀº ÇÈ¼¿(220)
            std::uint8_t val = wall ? 50 : 220;

            for (int py = 0; py < cellPx; ++py)
            {
                std::uint8_t* row = image.RowPtr(y * cellPx + py);
                if (!row) continue;

                for (int px = 0; px < cellPx; ++px)
                {
                    row[x * cellPx + px] = val;
                }
            }
        }
    }

    return true;
}
