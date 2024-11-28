
// App.h : main header file for the App application
//
#pragma once

#ifndef __AFXWIN_H__
	#error "include 'pch.h' before including this file for PCH"
#endif

#include "resource.h"       // main symbols
#include "../Frameworks/RenderSystem.h"
#include "../Infra/Event.h"

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

	void setDisplay(
		Frx::Display *pDisplay);

	[[nodiscard]]
	constexpr Infra::Event<> const &getUIIdleEvent() noexcept;

// Overrides
public:
	virtual BOOL InitInstance();
	virtual int ExitInstance();

// Implementation

public:
	afx_msg void OnAppAbout();
	DECLARE_MESSAGE_MAP()

private:
	std::unique_ptr<Frx::RenderSystem> __pRenderSystem;
	Frx::Display *__pDisplay{ };

	mutable Infra::Event<> __uiIdleEvent;

	void __customInit();
public:
	virtual BOOL OnIdle(LONG lCount);
};

constexpr Infra::Event<> const &CApp::getUIIdleEvent() noexcept
{
	return __uiIdleEvent;
}

extern CApp theApp;