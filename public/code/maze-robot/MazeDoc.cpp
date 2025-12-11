
// MazeDoc.cpp: CMazeDoc 클래스의 구현
//

#include "pch.h"
#include "framework.h"
// SHARED_HANDLERS는 미리 보기, 축소판 그림 및 검색 필터 처리기를 구현하는 ATL 프로젝트에서 정의할 수 있으며
// 해당 프로젝트와 문서 코드를 공유하도록 해 줍니다.
#ifndef SHARED_HANDLERS
#include "Maze.h"
#endif

#include "MazeDoc.h"

#include <propkey.h>

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

// CMazeDoc

IMPLEMENT_DYNCREATE(CMazeDoc, CDocument)

BEGIN_MESSAGE_MAP(CMazeDoc, CDocument)
END_MESSAGE_MAP()


// CMazeDoc 생성/소멸

CMazeDoc::CMazeDoc() noexcept
{
	// TODO: 여기에 일회성 생성 코드를 추가합니다.

}

CMazeDoc::~CMazeDoc()
{
}

BOOL CMazeDoc::OnNewDocument()
{
	if (!CDocument::OnNewDocument())
		return FALSE;

	// --- 여기서 게임 객체들 초기화 ---

	m_mazeMap = std::make_unique<MazeMap>();
	m_agent = std::make_unique<MazeAgent>();
	m_viewport = std::make_unique<MazeViewport>();
	m_imageProcessor = std::make_unique<MazeImageProcessor>();
	m_pathFinder = std::make_unique<PathFinder>();

	// 미로 사이즈
	const int mapW = 40;
	const int mapH = 30;
	m_mazeMap->Resize(mapW, mapH);

	// 간단한 벽: 가장자리 막기
	for (int y = 0; y < mapH; ++y)
	{
		for (int x = 0; x < mapW; ++x)
		{
			bool border = (x == 0 || x == mapW - 1 ||
				y == 0 || y == mapH - 1);
			m_mazeMap->SetWall(x, y, border);
		}
	}

	// 내부에 대충 몇 개의 벽 추가
	for (int x = 5; x < 20; ++x)
		m_mazeMap->SetWall(x, 10, true);
	for (int y = 10; y < 25; ++y)
		m_mazeMap->SetWall(15, y, true);

	// 에이전트 시작 위치
	m_agent->SetPosition({ 1, 1 });
	m_agent->ClearVisited();
	m_agent->AddVisited({ 1, 1 });

	// 뷰포트 크기와 초기 위치
	m_viewport->SetSizeInCells({ 20, 15 });
	m_viewport->CenterOn(m_agent->GetPosition(), *m_mazeMap);

	// 센서 생성
	m_sensor = std::make_unique<VirtualMazeSensor>(m_mazeMap.get());

	// GameController 생성
	m_gameController = std::make_unique<GameController>(
		m_mazeMap.get(),
		m_agent.get(),
		m_viewport.get(),
		m_imageProcessor.get(),
		m_pathFinder.get(),
		m_sensor.get()
	);

	return TRUE;
}




// CMazeDoc serialization

void CMazeDoc::Serialize(CArchive& ar)
{
	if (ar.IsStoring())
	{
		// TODO: 여기에 저장 코드를 추가합니다.
	}
	else
	{
		// TODO: 여기에 로딩 코드를 추가합니다.
	}
}

#ifdef SHARED_HANDLERS

// 축소판 그림을 지원합니다.
void CMazeDoc::OnDrawThumbnail(CDC& dc, LPRECT lprcBounds)
{
	// 문서의 데이터를 그리려면 이 코드를 수정하십시오.
	dc.FillSolidRect(lprcBounds, RGB(255, 255, 255));

	CString strText = _T("TODO: implement thumbnail drawing here");
	LOGFONT lf;

	CFont* pDefaultGUIFont = CFont::FromHandle((HFONT) GetStockObject(DEFAULT_GUI_FONT));
	pDefaultGUIFont->GetLogFont(&lf);
	lf.lfHeight = 36;

	CFont fontDraw;
	fontDraw.CreateFontIndirect(&lf);

	CFont* pOldFont = dc.SelectObject(&fontDraw);
	dc.DrawText(strText, lprcBounds, DT_CENTER | DT_WORDBREAK);
	dc.SelectObject(pOldFont);
}

// 검색 처리기를 지원합니다.
void CMazeDoc::InitializeSearchContent()
{
	CString strSearchContent;
	// 문서의 데이터에서 검색 콘텐츠를 설정합니다.
	// 콘텐츠 부분은 ";"로 구분되어야 합니다.

	// 예: strSearchContent = _T("point;rectangle;circle;ole object;");
	SetSearchContent(strSearchContent);
}

void CMazeDoc::SetSearchContent(const CString& value)
{
	if (value.IsEmpty())
	{
		RemoveChunk(PKEY_Search_Contents.fmtid, PKEY_Search_Contents.pid);
	}
	else
	{
		CMFCFilterChunkValueImpl *pChunk = nullptr;
		ATLTRY(pChunk = new CMFCFilterChunkValueImpl);
		if (pChunk != nullptr)
		{
			pChunk->SetTextValue(PKEY_Search_Contents, value, CHUNK_TEXT);
			SetChunkValue(pChunk);
		}
	}
}

#endif // SHARED_HANDLERS

// CMazeDoc 진단

#ifdef _DEBUG
void CMazeDoc::AssertValid() const
{
	CDocument::AssertValid();
}

void CMazeDoc::Dump(CDumpContext& dc) const
{
	CDocument::Dump(dc);
}
#endif //_DEBUG


// CMazeDoc 명령
