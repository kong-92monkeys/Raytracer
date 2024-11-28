
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
	ON_WM_CREATE()
	ON_WM_SIZE()
	ON_WM_DESTROY()
	ON_WM_ERASEBKGND()
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
	if (__pRenderTarget->isPresentable())
		theApp.reserveRender(__pRenderTarget.get());

	ValidateRect(nullptr);
}

int CMainView::OnCreate(LPCREATESTRUCT lpCreateStruct)
{
	if (CWnd::OnCreate(lpCreateStruct) == -1)
		return -1;

	__pRenderTarget = std::unique_ptr<Render::RenderTarget>
	{
		theApp.createRenderTarget(
			GetSafeHwnd(), lpCreateStruct->cx, lpCreateStruct->cy, 4U)
	};

	return 0;
}


void CMainView::OnSize(UINT nType, int cx, int cy)
{
	CWnd::OnSize(nType, cx, cy);

	// TODO: Add your message handler code here
	__pRenderTarget->resize(cx, cy);
}


void CMainView::OnDestroy()
{
	CWnd::OnDestroy();

	// TODO: Add your message handler code here
	theApp.cancelRender(__pRenderTarget.get());
	__pRenderTarget = nullptr;
}


BOOL CMainView::OnEraseBkgnd(CDC *pDC)
{
	// TODO: Add your message handler code here and/or call default
	return FALSE;
}
