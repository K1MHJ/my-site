
// MazeView.h: CMazeView 클래스의 인터페이스
//

#pragma once


class CMazeView : public CView
{
protected: // serialization에서만 만들어집니다.
	CMazeView() noexcept;
	DECLARE_DYNCREATE(CMazeView)

// 특성입니다.
public:
	CMazeDoc* GetDocument() const;
	UINT_PTR m_timerId{ 0 };
// 작업입니다.
public:

// 재정의입니다.
public:
	virtual void OnDraw(CDC* pDC);  // 이 뷰를 그리기 위해 재정의되었습니다.
	virtual BOOL PreCreateWindow(CREATESTRUCT& cs);
protected:

// 구현입니다.
public:
	virtual ~CMazeView();
#ifdef _DEBUG
	virtual void AssertValid() const;
	virtual void Dump(CDumpContext& dc) const;
#endif

protected:
	afx_msg BOOL OnEraseBkgnd(CDC* pDC);
	afx_msg int  OnCreate(LPCREATESTRUCT lpCreateStruct);
	afx_msg void OnDestroy();
	afx_msg void OnTimer(UINT_PTR nIDEvent);
	afx_msg void OnKeyDown(UINT nChar, UINT nRepCnt, UINT nFlags);
// 생성된 메시지 맵 함수
protected:
	DECLARE_MESSAGE_MAP()
};

#ifndef _DEBUG  // MazeView.cpp의 디버그 버전
inline CMazeDoc* CMazeView::GetDocument() const
   { return reinterpret_cast<CMazeDoc*>(m_pDocument); }
#endif

