#pragma once

#include "ImageBuffer.h"
#include "LocalMap.h"
#include "MazeViewport.h"

class MazeImageProcessor
{
public:
    MazeImageProcessor();

    // image: 현재 화면에 보이는 (혹은 그 주변) 실제 픽셀
    // viewport: 이 이미지가 전체 미로의 어느 구간인지 좌표 정보
    LocalMap Analyze(const ImageBuffer& image,
        const MazeViewport& viewport) const;
};
