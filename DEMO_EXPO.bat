@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo.
echo  ============================================================
echo    IRRIGACAO PREDITIVA - DEMO EXPO SAO JUDAS
echo  ============================================================
echo.
echo   Abrindo 3 janelas:  Servidor  -  No virtual  -  Painel
echo   O navegador abre sozinho em  http://localhost:8501
echo.
echo   Para ENCERRAR a demo: feche as 3 janelas pretas.
echo  ------------------------------------------------------------
echo.

REM --- Confere se o ambiente virtual existe ---
if not exist "%~dp0.venv\Scripts\python.exe" (
  echo  [ERRO] Nao encontrei .venv\Scripts\python.exe nesta pasta.
  echo         Rode antes:  python -m venv .venv
  echo                      .venv\Scripts\python.exe -m pip install -r requirements.txt
  echo.
  pause
  exit /b 1
)

REM --- Janela 1: servidor (constantes de tempo encolhidas p/ demo) ---
start "Irrigacao - Servidor" cmd /k "cd /d ""%~dp0"" && set PYTHONIOENCODING=utf-8&& set IRRIGACAO_LOCKOUT_S=15&& set IRRIGACAO_TAU_TRANSITO_S=30&& set IRRIGACAO_PULSO_MAX_S=900&& set IRRIGACAO_DB=irrigacao.db&& set IRRIGACAO_BUNDLE=modelos_final\solo_sao_paulo_h24.joblib&& .venv\Scripts\python.exe -m uvicorn servidor:app --port 8000"

echo   Subindo o servidor...
timeout /t 7 /nobreak >nul

REM --- Janela 2: no virtual (sensor/bomba/solo de mentira) ---
start "Irrigacao - No Virtual" cmd /k "cd /d ""%~dp0"" && set PYTHONIOENCODING=utf-8&& .venv\Scripts\python.exe no_virtual.py --aceleracao 150 --eto 8 --chuva 0 --umidade-inicial 85 --passos 100000"

REM --- Janela 3: painel Streamlit ---
start "Irrigacao - Painel" cmd /k "cd /d ""%~dp0"" && set PYTHONIOENCODING=utf-8&& set IRRIGACAO_API_URL=http://127.0.0.1:8000&& .venv\Scripts\python.exe -m streamlit run dashboard.py --server.port 8501 --server.headless true"

echo   Subindo o painel...
timeout /t 9 /nobreak >nul

REM --- Abre o navegador no painel ---
start "" http://localhost:8501

echo.
echo   Pronto! O painel esta em  http://localhost:8501
echo   (se o navegador nao abrir, digite esse endereco manualmente)
echo.
timeout /t 4 /nobreak >nul
exit /b 0
