@echo off
echo GPU-Accelerated Geometric Algebra Simulation
echo ==========================================
echo.

REM Check if Python is available
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Python not found in PATH. Please install Python or add it to your PATH.
    goto :end
)

REM Check if required packages are installed
python -c "import numpy" >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo NumPy not found. Installing...
    pip install numpy
)

REM Try to import CuPy (optional)
python -c "import cupy" >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo CuPy not found. GPU acceleration will be disabled.
    echo To enable GPU acceleration, install CuPy with: pip install cupy-cuda11x
    echo (Replace cuda11x with your CUDA version)
    set GPU_FLAG=--no-gpu
) else (
    echo CuPy found. GPU acceleration enabled.
    set GPU_FLAG=
)

echo.
echo Running simulation...
echo.

REM Run the simulation
python gpu_accelerated_simulation.py %GPU_FLAG% %*

:end
echo.
pause