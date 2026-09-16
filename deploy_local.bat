@echo off
echo ====================================================================
echo Intelligent SDN Traffic Engineering - Production Deployment
echo ====================================================================
echo.

echo [1/3] Running automated pytest test suite...
if exist .venv\Scripts\python.exe (
    .venv\Scripts\python.exe -m pytest
) else (
    python -m pytest
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Unit tests failed! Aborting deployment.
    exit /b %ERRORLEVEL%
)

echo.
echo [2/3] Building and starting Docker containers...
docker compose up --build -d

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Docker compose deployment failed!
    exit /b %ERRORLEVEL%
)

echo.
echo [3/3] Verifying service deployment status...
docker compose ps

echo.
echo ====================================================================
echo Deployment Successful!
echo  - FastAPI REST API:      http://localhost:8000/docs
echo  - Streamlit Dashboard:   http://localhost:8501
echo ====================================================================
