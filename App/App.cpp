// App.cpp : Defines the class behaviors for the application.
//

#include "pch.h"
#include "framework.h"
#include "afxwinappex.h"
#include "afxdialogex.h"
#include "App.h"
#include "MainFrm.h"
#include "AboutDlg.h"


#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// CApp

BEGIN_MESSAGE_MAP(CApp, CWinApp)
	ON_COMMAND(ID_APP_ABOUT, &CApp::OnAppAbout)
END_MESSAGE_MAP()


// CApp construction

CApp::CApp() noexcept
{

	// TODO: replace application ID string below with unique ID string; recommended
	// format for string is CompanyName.ProductName.SubProduct.VersionInformation
	SetAppID(_T("com.92monkeys.raytracer"));

	// TODO: add construction code here,
	// Place all significant initialization in InitInstance
}

// The one and only CApp object

CApp theApp;


// CApp initialization

[[nodiscard]]
Render::RenderTarget *CApp::createRenderTarget(
	HWND const hwnd,
	UINT const width,
	UINT const height,
	UINT const swapchainImageCount)
{
	return __pRenderEngine->createRenderTarget(
		hwnd, width, height, swapchainImageCount);
}

void CApp::setMainRenderTarget(
	Render::RenderTarget *const pRenderTarget)
{
	__pMainRenderTarget = pRenderTarget;
}

BOOL CApp::InitInstance()
{
	CWinApp::InitInstance();


	EnableTaskbarInteraction(FALSE);

	// AfxInitRichEdit2() is required to use RichEdit control
	// AfxInitRichEdit2();

	// Standard initialization
	// If you are not using these features and wish to reduce the size
	// of your final executable, you should remove from the following
	// the specific initialization routines you do not need
	// Change the registry key under which our settings are stored
	// TODO: You should modify this string to be something appropriate
	// such as the name of your company or organization
	SetRegistryKey(_T("92Monkeys"));

	__customInit();

	// To create the main window, this code creates a new frame window
	// object and then sets it as the application's main window object
	CFrameWnd* pFrame = new CMainFrame;
	if (!pFrame)
		return FALSE;
	m_pMainWnd = pFrame;
	// create and load the frame with its resources
	pFrame->LoadFrame(IDR_MAINFRAME,
		WS_OVERLAPPEDWINDOW | FWS_ADDTOTITLE, nullptr,
		nullptr);

	// The one and only window has been initialized, so show and update it
	pFrame->ShowWindow(SW_SHOW);
	pFrame->UpdateWindow();
	return TRUE;
}

int CApp::ExitInstance()
{
	//TODO: handle additional resources you may have added
	__pRenderEngine = nullptr;
	return CWinApp::ExitInstance();
}

// CApp message handlers

// App command to run the dialog
void CApp::OnAppAbout()
{
	CAboutDlg aboutDlg;
	aboutDlg.DoModal();
}

void CApp::__customInit()
{
	__pRenderEngine = std::make_unique<Render::Engine>();
}

BOOL CApp::OnIdle(LONG lCount)
{
	// TODO: Add your specialized code here and/or call the base class
	if (__pMainRenderTarget)
		__pRenderEngine->reserveRender(__pMainRenderTarget);

	__pRenderEngine->render();
	return CWinApp::OnIdle(lCount);
}