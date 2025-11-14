@echo off
REM ========================================
REM FIX SUBMODULE ERRORS IN GITHUB GUI
REM Run this ONCE to stop submodule errors
REM ========================================

echo.
echo ========================================
echo Fixing Git Submodule Errors
echo ========================================
echo.

cd /d "%~dp0"

echo [1/5] Disabling submodule recursion...
git config --local submodule.recurse false

echo [2/5] Disabling submodule fetching...
git config --local fetch.recurseSubmodules no

echo [3/5] Hiding submodule summaries in status...
git config --local status.submoduleSummary false

echo [4/5] Ignoring submodule changes in diffs...
git config --local diff.ignoreSubmodules all

echo [5/5] Verifying settings...
echo.
git config --list --local | findstr /i "submodule fetch.recurse status.submodule diff.ignore"

echo.
echo ========================================
echo ✅ DONE! Submodule errors are now GONE
echo ========================================
echo.
echo Your GitHub Desktop will no longer show:
echo   - "Could not access submodule 'third_party/cub'"
echo   - "Could not access submodule 'third_party/pybind11'"
echo   - "Could not access submodule 'third_party/spdlog'"
echo.
echo You can now pull/push normally in GitHub Desktop!
echo.
pause
