@echo off
setlocal
cd /d "%~dp0"
set "SCRIPT=%~dp0Microwell_Spheroid_Profiling_GUI.py"
where py >nul 2>nul
if %errorlevel%==0 (
  py -3.11 -c "import sys" >nul 2>nul
  if %errorlevel%==0 (
    py -3.11 "%SCRIPT%"
    goto :after_run
  )
  py -3.12 -c "import sys" >nul 2>nul
  if %errorlevel%==0 (
    py -3.12 "%SCRIPT%"
    goto :after_run
  )
  py -3.10 -c "import sys" >nul 2>nul
  if %errorlevel%==0 (
    py -3.10 "%SCRIPT%"
    goto :after_run
  )
)
where python >nul 2>nul
if %errorlevel%==0 (
  python -c "import sys; raise SystemExit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)" >nul 2>nul
  if %errorlevel%==0 (
    python "%SCRIPT%"
    goto :after_run
  ) else (
    echo [ERROR] Unsupported Python version from 'python' command.
    echo Please install Python 3.11 and re-run this launcher.
    pause
    goto :eof
  )
)
echo [ERROR] Python was not found in PATH, or only unsupported versions were found.
echo Please install Python 3.11 (recommended) and re-run this launcher.
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
