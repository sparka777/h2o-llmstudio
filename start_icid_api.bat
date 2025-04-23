@echo off
cd /d "C:\Users\Sydney Parker\h2o-llmstudio"
call .\llm-env\Scripts\activate.bat
set PYTHONPATH=%cd%
uvicorn icid_api:app --host 127.0.0.1 --port 8000
pause