@echo off
echo ===============================================================================
echo Swapping to v3 FP16-Constrained Engine
echo ===============================================================================
echo.

cd /d D:\watermarkz\engines\transformer

echo Backing up v2 engine...
if exist transformer_full_v2.engine (
    move transformer_full_v2.engine transformer_full_v2.engine.old
    echo [OK] v2 backed up
) else (
    echo [WARNING] v2 engine not found
)

echo.
echo Activating v3 FP16-constrained engine...
if exist transformer_full_v3_fp16_constrained.engine (
    copy transformer_full_v3_fp16_constrained.engine transformer_full_v2.engine /Y
    echo [OK] v3 engine is now active as transformer_full_v2.engine
) else (
    echo [ERROR] v3 engine not found!
    pause
    exit /b 1
)

echo.
echo ===============================================================================
echo SUCCESS! v3 FP16-Constrained Engine is now active
echo ===============================================================================
echo.
echo Expected improvements:
echo   - std stays at ~1.0 (no drift to 3.7)
echo   - max values ~8-10 (natural range)
echo   - 5-10x faster than PyTorch
echo   - Quality should match PyTorch
echo.
echo Now restart Celery (Ctrl+C, then START_CELERY_TRT.bat)
echo.
pause
