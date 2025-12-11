// GameController.h

#pragma once

#include <memory>

class MazeMap;
class MazeAgent;
class MazeViewport;
class MazeImageProcessor;
class PathFinder;
class ImageBuffer;
class IMazeSensor;
class LocalMap;

class GameController
{
public:
    GameController(MazeMap* map,
        MazeAgent* agent,
        MazeViewport* viewport,
        MazeImageProcessor* imageProcessor,
        PathFinder* pathFinder,
        IMazeSensor* sensor);

    // 실시간(타이머)용: 한 번 호출 = 한 스텝
    void Update(float deltaTime);

    // ★ 시뮬레이션 / 로직용: 타이머와 무관한 단일 스텝
    void StepOnce();

    // ★ 시뮬레이션 루프: 원하는 횟수만큼 순식간에 반복
    void RunSimulation(int maxSteps);

    const ImageBuffer* GetLastCapture() const noexcept;
    const LocalMap* GetLastLocalMap() const noexcept;

    // (필요하면 나중에 목표 도착 여부 같은 상태 플래그도 추가)

private:
    MazeMap* m_map{ nullptr };
    MazeAgent* m_agent{ nullptr };
    MazeViewport* m_viewport{ nullptr };
    MazeImageProcessor* m_imageProcessor{ nullptr };
    PathFinder* m_pathFinder{ nullptr };
    IMazeSensor* m_sensor{ nullptr };

    std::unique_ptr<ImageBuffer> m_captureImage;
    std::unique_ptr<LocalMap>    m_lastLocalMap;
};