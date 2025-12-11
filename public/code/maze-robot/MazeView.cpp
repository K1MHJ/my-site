
// MazeView.cpp: CMazeView 클래스의 구현
//

#include "pch.h"
#include "framework.h"
// SHARED_HANDLERS는 미리 보기, 축소판 그림 및 검색 필터 처리기를 구현하는 ATL 프로젝트에서 정의할 수 있으며
// 해당 프로젝트와 문서 코드를 공유하도록 해 줍니다.
#ifndef SHARED_HANDLERS
#include "Maze.h"
#endif

#include "MazeDoc.h"
#include "MazeView.h"
#include "MazeMap.h"
#include "MazeAgent.h"
#include "MazeViewport.h"
#include "GameController.h"
#include "LocalMap.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// CMazeView

IMPLEMENT_DYNCREATE(CMazeView, CView)

BEGIN_MESSAGE_MAP(CMazeView, CView)
	ON_WM_CREATE()
	ON_WM_DESTROY()
	ON_WM_TIMER()
	ON_WM_KEYDOWN()
    ON_WM_ERASEBKGND()
END_MESSAGE_MAP()

// CMazeView 생성/소멸

CMazeView::CMazeView() noexcept
{
	// TODO: 여기에 생성 코드를 추가합니다.

}

CMazeView::~CMazeView()
{
}

BOOL CMazeView::PreCreateWindow(CREATESTRUCT& cs)
{
	// TODO: CREATESTRUCT cs를 수정하여 여기에서
	//  Window 클래스 또는 스타일을 수정합니다.

	return CView::PreCreateWindow(cs);
}



// CMazeView 진단

#ifdef _DEBUG
void CMazeView::AssertValid() const
{
	CView::AssertValid();
}

void CMazeView::Dump(CDumpContext& dc) const
{
	CView::Dump(dc);
}

CMazeDoc* CMazeView::GetDocument() const // 디버그되지 않은 버전은 인라인으로 지정됩니다.
{
	ASSERT(m_pDocument->IsKindOf(RUNTIME_CLASS(CMazeDoc)));
	return (CMazeDoc*)m_pDocument;
}
#endif //_DEBUG


// CMazeView 메시지 처리기

int CMazeView::OnCreate(LPCREATESTRUCT lpCreateStruct)
{
    if (CView::OnCreate(lpCreateStruct) == -1)
        return -1;

    // 100ms마다 타이머
    m_timerId = SetTimer(1, 100, nullptr);

    // 키 입력 받기 위해 포커스 설정 (필요시)
    SetFocus();

    return 0;
}

void CMazeView::OnDestroy()
{
    if (m_timerId != 0)
    {
        KillTimer(m_timerId);
        m_timerId = 0;
    }

    CView::OnDestroy();
}

void CMazeView::OnTimer(UINT_PTR nIDEvent)
{
    if (nIDEvent == m_timerId)
    {
        CMazeDoc* pDoc = GetDocument();
        if (pDoc && pDoc->m_gameController)
        {
            pDoc->m_gameController->Update(0.1f);
            Invalidate(FALSE);
        }
    }

    CView::OnTimer(nIDEvent);
}

void CMazeView::OnKeyDown(UINT nChar, UINT nRepCnt, UINT nFlags)
{
    CMazeDoc* pDoc = GetDocument();
    if (!pDoc || !pDoc->m_mazeMap || !pDoc->m_agent || !pDoc->m_viewport)
    {
        CView::OnKeyDown(nChar, nRepCnt, nFlags);
        return;
    }

    if (pDoc && pDoc->m_gameController)
    {
        // ★ 'S' 키: 1000 스텝 시뮬레이션
        if (nChar == 'S')
        {
            // 예: 1000번 StepOnce 실행 (타이머 없이)
            pDoc->m_gameController->RunSimulation(1000);

            // 최종 상태만 화면에 한 번 그리기
            Invalidate(FALSE);
            return;
        }
    }


    Int2 pos = pDoc->m_agent->GetPosition();
    Int2 next = pos;

    switch (nChar)
    {
    case VK_LEFT:  next.x -= 1; break;
    case VK_RIGHT: next.x += 1; break;
    case VK_UP:    next.y -= 1; break;
    case VK_DOWN:  next.y += 1; break;
    default:
        break;
    }
   
    if (next.x != pos.x || next.y != pos.y)
    {
        if (!pDoc->m_mazeMap->IsWall(next.x, next.y))
        {
            pDoc->m_agent->SetPosition(next);
            pDoc->m_agent->AddVisited(next);
            pDoc->m_viewport->CenterOn(next, *pDoc->m_mazeMap);
            Invalidate(FALSE);
        }
    }

    CView::OnKeyDown(nChar, nRepCnt, nFlags);
}
// CMazeView 그리기

void CMazeView::OnDraw(CDC* pDC)
{
    CMazeDoc* pDoc = GetDocument();
    if (!pDoc || !pDoc->m_mazeMap || !pDoc->m_agent || !pDoc->m_viewport)
        return;

    // ---- 1) 메모리 DC/비트맵 준비 ----
    CRect clientRect;
    GetClientRect(&clientRect);

    CDC memDC;
    memDC.CreateCompatibleDC(pDC);

    CBitmap bmp;
    bmp.CreateCompatibleBitmap(pDC, clientRect.Width(), clientRect.Height());

    CBitmap* pOldBmp = memDC.SelectObject(&bmp);

    // 이후 모든 그리기는 memDC에 한다
    CDC* pDrawDC = &memDC;

    // 배경 채우기
    pDrawDC->FillSolidRect(&clientRect, RGB(200, 200, 200));

    // ---- 2) 여기서부터 기존 OnDraw 코드 내용(미로/에이전트/LocalMap) 사용 ----

    const int cellSize = 20;

    Int2 vpOrigin = pDoc->m_viewport->GetOriginCell();
    Int2 vpSize = pDoc->m_viewport->GetSizeInCells();
    Int2 agentPos = pDoc->m_agent->GetPosition();

    // 디버그 텍스트
    CString info;
    info.Format(L"Viewport origin = (%d, %d), Agent = (%d, %d)",
        vpOrigin.x, vpOrigin.y, agentPos.x, agentPos.y);
    pDrawDC->TextOutW(5, 5, info);

    // 메인 미로 그리기
    for (int y = 0; y < vpSize.y; ++y)
    {
        for (int x = 0; x < vpSize.x; ++x)
        {
            int mx = vpOrigin.x + x;
            int my = vpOrigin.y + y;

            CRect r(
                x * cellSize,
                y * cellSize,
                (x + 1) * cellSize,
                (y + 1) * cellSize
            );

            bool wall = pDoc->m_mazeMap->IsWall(mx, my);

            COLORREF color;
            if (wall)
            {
                color = RGB(0, 0, 0);
            }
            else
            {
                int blockX = mx / 5;
                int blockY = my / 5;
                bool alt = ((blockX + blockY) % 2) != 0;

                if (alt)
                    color = RGB(240, 255, 220);
                else
                    color = RGB(220, 240, 255);
            }

            pDrawDC->FillSolidRect(&r, color);
            pDrawDC->Rectangle(&r);
        }
    }

    // 에이전트
    if (agentPos.x >= vpOrigin.x && agentPos.x < vpOrigin.x + vpSize.x &&
        agentPos.y >= vpOrigin.y && agentPos.y < vpOrigin.y + vpSize.y)
    {
        int ax = agentPos.x - vpOrigin.x;
        int ay = agentPos.y - vpOrigin.y;

        CRect r(
            ax * cellSize,
            ay * cellSize,
            (ax + 1) * cellSize,
            (ay + 1) * cellSize
        );

        r.DeflateRect(4, 4);
        CBrush brush(RGB(255, 0, 0));
        CBrush* pOldBrush = pDrawDC->SelectObject(&brush);
        pDrawDC->Ellipse(&r);
        pDrawDC->SelectObject(pOldBrush);
    }

    // LocalMap 미니맵 그리기 (있다면)
    if (pDoc->m_gameController)
    {
        const LocalMap* local = pDoc->m_gameController->GetLastLocalMap();
        if (local)
        {
            Int2 lmSize = local->GetSize();
            if (lmSize.x > 0 && lmSize.y > 0)
            {
                const int dbgCell = 8;
                int lmW = lmSize.x * dbgCell;
                int lmH = lmSize.y * dbgCell;

                int margin = 10;
                int top = 30;
                int left = clientRect.right - lmW - margin;

                CRect dbgRect(left - 2, top - 2, left + lmW + 2, top + lmH + 2);
                pDrawDC->FillSolidRect(&dbgRect, RGB(220, 220, 220));
                pDrawDC->Rectangle(&dbgRect);

                pDrawDC->TextOutW(left, top - 18, L"LocalMap (sensor+image)");

                for (int y = 0; y < lmSize.y; ++y)
                {
                    for (int x = 0; x < lmSize.x; ++x)
                    {
                        CRect r(
                            left + x * dbgCell,
                            top + y * dbgCell,
                            left + (x + 1) * dbgCell,
                            top + (y + 1) * dbgCell
                        );

                        LocalMap::CellType t = local->GetCell(x, y);
                        COLORREF c;

                        switch (t)
                        {
                        case LocalMap::CellType::Wall:
                            c = RGB(30, 30, 30);
                            break;
                        case LocalMap::CellType::Free:
                            c = RGB(230, 230, 255);
                            break;
                        case LocalMap::CellType::Unknown:
                        default:
                            c = RGB(150, 150, 150);
                            break;
                        }

                        pDrawDC->FillSolidRect(&r, c);
                        pDrawDC->Rectangle(&r);
                    }
                }
            }
        }
    }

    // ---- 3) 메모리 DC → 화면 DC로 한 번에 복사 ----
    pDC->BitBlt(0, 0,
        clientRect.Width(),
        clientRect.Height(),
        &memDC,
        0, 0,
        SRCCOPY);

    memDC.SelectObject(pOldBmp);
}
BOOL CMazeView::OnEraseBkgnd(CDC* /*pDC*/)
{
    // 기본 동작(배경 흰색으로 지우기)을 막는다
    return TRUE;
}