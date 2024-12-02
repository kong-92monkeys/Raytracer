// MainView.h : interface of the CMainView class
//

#pragma once

#include "../Frameworks/Display.h"

// CMainView window

class CMainView : public CWnd
{
// Construction
public:
	CMainView();

// Attributes
public:

// Operations
public:

// Overrides
	protected:
	virtual BOOL PreCreateWindow(CREATESTRUCT& cs);

// Implementation
public:
	virtual ~CMainView();

	// Generated message map functions
protected:
	afx_msg void OnPaint();
	DECLARE_MESSAGE_MAP()

public:
	afx_msg int OnCreate(LPCREATESTRUCT lpCreateStruct);
	afx_msg void OnSize(UINT nType, int cx, int cy);
	afx_msg void OnDestroy();
	afx_msg BOOL OnEraseBkgnd(CDC *pDC);

private:
	std::unique_ptr<Frx::Display> __pDisplay;
};