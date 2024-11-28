
// App.h : main header file for the App application
//
#pragma once

#ifndef __AFXWIN_H__
	#error "include 'pch.h' before including this file for PCH"
#endif

#include "resource.h"       // main symbols
#include "../Render/RenderTarget.h"
#include "../Render/Engine.h"
#include "../Infra/Event.h"

// CApp:
// See App.cpp for the implementation of this class
//

class CApp : public CWinApp
{
public:
	CApp() noexcept;

	[[nodiscard]]
	Render::RenderTarget *createRenderTarget(
		HWND hwnd,
		UINT width,
		UINT height,
		UINT swapchainImageCount);

	void reserveRender(
		Render::RenderTarget *pRenderTarget);

	void cancelRender(
		Render::RenderTarget *pRenderTarget);

// Overrides
public:
	virtual BOOL InitInstance();
	virtual int ExitInstance();

// Implementation

public:
	afx_msg void OnAppAbout();
	DECLARE_MESSAGE_MAP()

private:
	std::unique_ptr<Render::Engine> __pRenderEngine;

	void __customInit();
public:
	virtual BOOL OnIdle(LONG lCount);
};

extern CApp theApp;