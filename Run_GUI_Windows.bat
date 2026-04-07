@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (
  py -3 "%~dp0Microwell_Spheroid_Profiling_GUI.py"
  goto :after_run
)
where python >nul 2>nul
if %errorlevel%==0 (
  python "%~dp0Microwell_Spheroid_Profiling_GUI.py"
  goto :after_run
)
echo [ERROR] Python was not found in PATH.
echo Install Python 3.11+ and re-run this launcher.
pause
goto :eof
:after_run
if errorlevel 1 (
  echo.
  echo [INFO] GUI exited with an error. You can also run:
  echo        python Microwell_Spheroid_Profiling_GUI.py
  pause
)
endlocal
