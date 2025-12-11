#pragma once

#include "MazeSensor.h"
#include "MazeMap.h"

// MazeMap을 기반으로 가상의 "카메라 이미지"를 만들어주는 센서
class VirtualMazeSensor : public IMazeSensor
{
public:
    explicit VirtualMazeSensor(const MazeMap* map);

    // viewport 범위의 MazeMap을 그레이스케일 이미지로 생성
    bool Capture(const MazeViewport& viewport,
        ImageBuffer& image) override;

private:
    const MazeMap* m_map{ nullptr };
};
