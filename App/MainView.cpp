
// MainView.cpp : implementation of the CMainView class
//

#include "pch.h"
#include "framework.h"
#include "App.h"
#include "MainView.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// CMainView

CMainView::CMainView()
{
}

CMainView::~CMainView()
{
}


BEGIN_MESSAGE_MAP(CMainView, CWnd)
	ON_WM_PAINT()
END_MESSAGE_MAP()



// CMainView message handlers

BOOL CMainView::PreCreateWindow(CREATESTRUCT& cs) 
{
	if (!CWnd::PreCreateWindow(cs))
		return FALSE;

	cs.dwExStyle |= WS_EX_CLIENTEDGE;
	cs.style &= ~WS_BORDER;
	cs.lpszClass = AfxRegisterWndClass(CS_HREDRAW|CS_VREDRAW|CS_DBLCLKS, 
		::LoadCursor(nullptr, IDC_ARROW), reinterpret_cast<HBRUSH>(COLOR_WINDOW+1), nullptr);

	return TRUE;
}

void CMainView::OnPaint() 
{
	CPaintDC dc(this); // device context for painting
	
	// TODO: Add your message handler code here
	
	// Do not call CWnd::OnPaint() for painting messages
}

