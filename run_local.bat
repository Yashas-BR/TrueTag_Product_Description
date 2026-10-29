@echo off
REM Start TrueTag on this computer, then open http://127.0.0.1:7860
cd /d "%~dp0"
REM Use the D: model cache if this machine has one (keeps C: free).
if exist "D:\hf_cache" set HF_HOME=D:\hf_cache
REM With both models in .\models, run fully offline (no Hugging Face lookups).
if exist "models\google__siglip-so400m-patch14-384\config.json" if exist "models\Qwen__Qwen2.5-1.5B-Instruct\config.json" (
  set HF_HUB_OFFLINE=1
  set TRANSFORMERS_OFFLINE=1
)
python -m uvicorn main:app --app-dir app --host 127.0.0.1 --port 7860
pause
