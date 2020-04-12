---
layout: post
title: "Window, GUI"
categories: ETC
author: lee gunjun
---

# X window system (X11)

라눅스, 유닉스에서 사용하는 window manager. Xorg, Wayland, Mir 등이 있다. Xorg 는 XFree86 을 토대로 개발함. x window protocal library 인 xlib 을 제공함.

# Xlib

X window System protocal client library. X server 와 interact 하기 위한 함수들을 가짐. gui toolkit 들인 gtk, qt, tk 등이 이를 이용함.

내 컴퓨터 (Ubuntu 18.04.3 LTS (GNU/Linux 4.15.0-55-generic x86_64)) 에선 /usr/lib/x86_64-linux-gnu/libX11.so.6 에 위치함.

기본적인 X11 코드
```
#include <X11/Xlib.h>

main() {
    Display *d ;

    d = XOpenDisplay ("localhost:0.0") ;

    XCloseDisplay (d) ;
}
```

Display 란 x server 가 관리하는 1개 이상의 Screen (듀얼모니터도 가능), keyboard, mouse 등으로 구성된 구조체임.

XOpenDisplay 를 통해 서버에 접속함. localhost 의 0 번 display 의 0 번 screen 에 접속하게 됨.

참괴: https://wiki.kldp.org/KoreanDoc/html/X-Window-Programming/X-Window-Programming-2.html

# GTK, GTK+, QT

GUI toolkit 이다. 즉, graphical interface 를 만들기 위한 라이브러리. xlib 의 wrapper 라고 할 수 있을거 같음. 모두 X server 위에서 돌아감. 윈도우의 GDI, GDI+ 에 대응함.

# GNOME, KDE

GNOME 과 KDE 는 desktop environment.

GNOME: gtk 이용. easy-to-use. 쉽게 사용할 수 있는 데스크탑 환경 제공

KDE: qt 이용하기 때문에 gtk 를 사용하는 애플리케이션을 쓰기위해서는 (크롬, 파폭, 등등..) gtk 라이브러리를 설치해야함.

GNOME 과 KDE 는 각각 애플리케이션들이 있음. gedit (editor), nautilus (파일탐색기) 이 GNOME 애플리케이션, KDE 는 안써봐서 모름.
