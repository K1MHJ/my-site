// GameController.cpp
#include "pch.h"
#include "GameController.h"
#include "MazeMap.h"
#include "MazeAgent.h"
#include "MazeViewport.h"
#include "MazeImageProcessor.h"
#include "PathFinder.h"
#include "ImageBuffer.h"
#include "LocalMap.h"
#include "MazeSensor.h"  // IMazeSensor

GameController::GameController(MazeMap* map,
    MazeAgent* agent,
    MazeViewport* viewport,
    MazeImageProcessor* imageProcessor,
    PathFinder* pathFinder,
    IMazeSensor* sensor)
    : m_map(map)
    , m_agent(agent)
    , m_viewport(viewport)
    , m_imageProcessor(imageProcessor)
    , m_pathFinder(pathFinder)
    , m_sensor(sensor)
{
    m_captureImage = std::make_unique<ImageBuffer>();
}
void GameController::Update(float /*deltaTime*/)
{
    // 실시간 모드에서 한 프레임당 한 번 호출
    StepOnce();
}

const ImageBuffer* GameController::GetLastCapture() const noexcept
{
    return m_captureImage.get();
}

const LocalMap* GameController::GetLastLocalMap() const noexcept
{
    return m_lastLocalMap.get();
}

void GameController::StepOnce()
{
    if (!m_map || !m_agent || !m_viewport || !m_pathFinder || !m_imageProcessor || !m_sensor)
        return;

    // 1) 센서에서 이미지 캡처
    if (!m_sensor->Capture(*m_viewport, *m_captureImage))
        return;

    // 2) 이미지 처리 → LocalMap
    LocalMap local = m_imageProcessor->Analyze(*m_captureImage, *m_viewport);

    // 마지막 LocalMap 저장 (디버그용)
    if (!m_lastLocalMap)
        m_lastLocalMap = std::make_unique<LocalMap>();
    *m_lastLocalMap = local;

#if MOVEAUTO
    // 3) PathFinder로 다음 칸 결정
    Int2 nextCell = m_pathFinder->DecideNextStep(*m_agent, local);

    // 4) 이동 (벽이면 무시)
    if (!m_map->IsWall(nextCell.x, nextCell.y))
    {
        m_agent->SetPosition(nextCell);
        m_agent->AddVisited(nextCell);
    }
#endif
    // 5) 뷰포트 갱신
    m_viewport->CenterOn(m_agent->GetPosition(), *m_map);
}

// ★ 시뮬레이션 루프
void GameController::RunSimulation(int maxSteps)
{
    if (maxSteps <= 0)
        return;

    for (int i = 0; i < maxSteps; ++i)
    {
        StepOnce();
        // 필요하면 여기에서 break 조건 추가 가능:
        // if (IsGoalReached()) break;
    }
}