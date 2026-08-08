@echo off
setlocal
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_trace_viewer.ps1" %*
set "VLM_TRACE_EXIT=%ERRORLEVEL%"
if not "%VLM_TRACE_EXIT%"=="0" (
    echo.
    echo VLM Trace launcher failed with exit code %VLM_TRACE_EXIT%.
    if "%VLM_TRACE_NO_PAUSE%"=="" pause
)
exit /b %VLM_TRACE_EXIT%
