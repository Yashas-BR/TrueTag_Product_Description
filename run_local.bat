@echo off
REM Start TrueTag on this computer, then open http://127.0.0.1:7860
REM Works without internet: models load from D:\hf_cache (downloaded once).
cd /d "%~dp0"
set HF_HOME=D:\hf_cache
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
python -m uvicorn main:app --app-dir app --host 127.0.0.1 --port 7860
