// MazeImageProcessor.cpp
#include "pch.h"
#include "MazeImageProcessor.h"
#include "MazeMap.h"      // 현재는 안 쓰지만, 필요하면 참조용
#include <algorithm>

#undef max
#undef min

MazeImageProcessor::MazeImageProcessor()
{
}

// 지금 구현은 "이미지가 셀 단위로 픽셀 블록을 가진다"는 단순 가정.
// - image 크기 / viewport 셀 수 = 타일 크기
// - 각 타일의 중앙 픽셀 기준으로 threshold 적용해서 벽/길 판정
LocalMap MazeImageProcessor::Analyze(const ImageBuffer& image,
    const MazeViewport& viewport) const
{
    LocalMap local;
    Int2 vpSize = viewport.GetSizeInCells();
    local.Resize(vpSize);
    local.SetViewOriginInMaze(viewport.GetOriginCell());

    if (image.GetWidth() <= 0 || image.GetHeight() <= 0 ||
        vpSize.x <= 0 || vpSize.y <= 0)
    {
        return local;
    }

    const int tileW = std::max(1, image.GetWidth() / vpSize.x);
    const int tileH = std::max(1, image.GetHeight() / vpSize.y);

    for (int cy = 0; cy < vpSize.y; ++cy)
    {
        for (int cx = 0; cx < vpSize.x; ++cx)
        {
            int px = cx * tileW + tileW / 2;
            int py = cy * tileH + tileH / 2;

            if (px >= image.GetWidth())
                px = image.GetWidth() - 1;
            if (py >= image.GetHeight())
                py = image.GetHeight() - 1;

            const std::uint8_t* row = image.RowPtr(py);
            std::uint8_t val = row ? row[px] : 0;

            // 임시 기준: 어두우면 벽, 밝으면 길
            LocalMap::CellType type =
                (val < 128) ? LocalMap::CellType::Wall : LocalMap::CellType::Free;

            local.SetCell(cx, cy, type);
        }
    }

    return local;
}
