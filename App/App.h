
// App.h : main header file for the App application
//
#pragma once

#ifndef __AFXWIN_H__
	#error "include 'pch.h' before including this file for PCH"
#endif

#include "resource.h"       // main symbols
#include "../Frameworks/RenderSystem.h"
#include "FPSCamera.h"

// CApp:
// See App.cpp for the implementation of this class
//

class CApp : public CWinApp
{
public:
	CApp() noexcept;

	[[nodiscard]]
	Frx::Display *createDisplay(
		HWND hwnd,
		UINT width,
		UINT height,
		UINT swapchainImageCount);

	void setMainDisplay(
		Frx::Display *pDisplay);

	void onKeyDown(UINT nChar);
	void onKeyUp(UINT nChar);

// Overrides
public:
	virtual BOOL InitInstance();
	virtual int ExitInstance();

// Implementation

public:
	afx_msg void OnAppAbout();
	DECLARE_MESSAGE_MAP()

public:
	virtual BOOL OnIdle(LONG lCount);

private:
	std::unique_ptr<Frx::RenderSystem> __pRenderSystem;
	Frx::Display *__pMainDisplay{ };

	FPSCamera __camera;

	Infra::EventListenerPtr<Frx::Display *> __pMainDisplayResizeListener;

	void __customInit();
	void __onMainDisplayResized() noexcept;
};

extern CApp theApp;