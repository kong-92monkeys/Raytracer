
// App.h : main header file for the App application
//
#pragma once

#ifndef __AFXWIN_H__
	#error "include 'pch.h' before including this file for PCH"
#endif

#include "resource.h"       // main symbols
#include "../Infra/Event.h"
#include "../Render/RenderTarget.h"
#include "../Render/Engine.h"
#include "../Frameworks/Camera.h"

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

	void setMainRenderTarget(
		Render::RenderTarget *pRenderTarget);

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
	std::unique_ptr<Render::Engine> __pRenderEngine;
	Render::RenderTarget *__pMainRenderTarget{ };

	Frx::Camera __camera;

	Infra::EventListenerPtr<Render::RenderTarget *> __pMainRenderTargetResizeListener;

	void __customInit();
	void __onMainRenderTargetResized() noexcept;
};

extern CApp theApp;