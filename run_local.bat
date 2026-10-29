@echo off
REM Start TrueTag on this computer, then open http://127.0.0.1:7860
cd /d "%~dp0"
set HF_HOME=D:\hf_cache
python -m uvicorn main:app --app-dir app --host 127.0.0.1 --port 7860
