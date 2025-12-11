
// MazeDoc.h: CMazeDoc 클래스의 인터페이스
//


#pragma once

#include <memory>
#include "MazeMap.h"
#include "MazeAgent.h"
#include "MazeViewport.h"
#include "GameController.h"
#include "MazeImageProcessor.h"
#include "PathFinder.h"
#include "VirtualMazeSensor.h" 
class CMazeDoc : public CDocument
{
protected: // serialization에서만 만들어집니다.
	CMazeDoc() noexcept;
	DECLARE_DYNCREATE(CMazeDoc)

public:
	// 게임 모델/엔진 객체들
	std::unique_ptr<MazeMap>            m_mazeMap;
	std::unique_ptr<MazeAgent>          m_agent;
	std::unique_ptr<MazeViewport>       m_viewport;
	std::unique_ptr<MazeImageProcessor> m_imageProcessor;
	std::unique_ptr<PathFinder>         m_pathFinder;
	std::unique_ptr<VirtualMazeSensor>  m_sensor;
	std::unique_ptr<GameController>     m_gameController;

// 작업입니다.
public:

// 재정의입니다.
public:
	virtual BOOL OnNewDocument();
	virtual void Serialize(CArchive& ar);
#ifdef SHARED_HANDLERS
	virtual void InitializeSearchContent();
	virtual void OnDrawThumbnail(CDC& dc, LPRECT lprcBounds);
#endif // SHARED_HANDLERS

// 구현입니다.
public:
	virtual ~CMazeDoc();
#ifdef _DEBUG
	virtual void AssertValid() const;
	virtual void Dump(CDumpContext& dc) const;
#endif

protected:

// 생성된 메시지 맵 함수
protected:
	DECLARE_MESSAGE_MAP()

#ifdef SHARED_HANDLERS
	// 검색 처리기에 대한 검색 콘텐츠를 설정하는 도우미 함수
	void SetSearchContent(const CString& value);
#endif // SHARED_HANDLERS
};
