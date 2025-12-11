#pragma once

#include "ImageBuffer.h"
#include "MazeViewport.h"

// 화면(또는 카메라)에서 현재 Viewport에 해당하는 이미지를 뽑아오는 인터페이스
class IMazeSensor
{
public:
    virtual ~IMazeSensor() = default;

    // viewport 정보에 맞춰 현재 화면을 캡쳐해서 image에 넣는다.
    // 성공 여부를 bool로 리턴 (실패 시 false)
    virtual bool Capture(const MazeViewport& viewport,
        ImageBuffer& image) = 0;
};
