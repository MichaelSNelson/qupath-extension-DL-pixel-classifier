@echo off
setlocal enabledelayedexpansion

rem ============================================================================
rem  DL Pixel Classifier - Python Server Setup
rem
rem  GPU auto-detection: automatically detects NVIDIA GPU and installs the
rem  correct CUDA-enabled PyTorch. Use --cpu to force CPU-only mode.
rem
rem  Usage:
rem    setup_server.bat                              (venv, auto-detect GPU)
rem    setup_server.bat F:\dl-classifier-env         (venv, custom path)
rem    setup_server.bat --cpu                        (venv, force CPU-only)
rem    setup_server.bat --cuda                       (venv, force CUDA)
rem    setup_server.bat F:\dl-env --cuda             (venv, custom path, CUDA)
rem    setup_server.bat --conda                      (conda env "dlclassifier")
rem    setup_server.bat --conda F:\dl-env --cuda     (conda env, custom path, CUDA)
rem    setup_server.bat --conda dlclassifier --cuda  (conda env by name, CUDA)
rem ============================================================================

set "SCRIPT_DIR=%~dp0"
set "SERVER_DIR=%SCRIPT_DIR%python_server"
set "ENV_PATH="
set "USE_CUDA=0"
set "USE_CONDA=0"
set "FORCE_CPU=0"
set "FORCE_CUDA=0"
set "GPU_NAME="
set "CUDA_VER="
set "CUDA_MAJOR="
set "CUDA_INDEX_URL="
set "PYTORCH_CUDA_TAG="
set "CONFIG_FILE=%SCRIPT_DIR%.server_config"

rem Parse arguments
:parse_args
if "%~1"=="" goto done_args
if /i "%~1"=="--cuda" (
    set "FORCE_CUDA=1"
    shift
    goto parse_args
)
if /i "%~1"=="--cpu" (
    set "FORCE_CPU=1"
    shift
    goto parse_args
)
if /i "%~1"=="--conda" (
    set "USE_CONDA=1"
    shift
    goto parse_args
)
if /i "%~1"=="--help" goto show_help
if /i "%~1"=="-h" goto show_help
rem Treat as env path/name
set "ENV_PATH=%~1"
shift
goto parse_args
:done_args

rem ---- Validate flag combinations ----
if "%FORCE_CPU%"=="1" if "%FORCE_CUDA%"=="1" (
    echo.
    echo  ERROR: Cannot use both --cpu and --cuda flags.
    exit /b 1
)

rem ---- GPU auto-detection ----
if "%FORCE_CPU%"=="1" (
    echo.
    echo  CPU-only mode requested [--cpu flag].
    set "USE_CUDA=0"
    goto detection_done
)

call :detect_gpu

if "%FORCE_CUDA%"=="1" if "%USE_CUDA%"=="0" (
    echo.
    echo  ERROR: --cuda flag specified but no NVIDIA GPU was detected.
    echo  If you have an NVIDIA GPU, ensure drivers are installed and nvidia-smi is in PATH.
    exit /b 1
)

:detection_done

rem ---- Determine environment type and defaults ----
if "%USE_CONDA%"=="1" goto pick_conda
goto pick_venv

:pick_conda
if "%ENV_PATH%"=="" set "ENV_PATH=dlclassifier"
goto setup_conda

:pick_venv
if "%ENV_PATH%"=="" set "ENV_PATH=%SERVER_DIR%\venv"
goto setup_venv

rem ================================================================
rem  GPU DETECTION SUBROUTINE
rem ================================================================
:detect_gpu

nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo.
    echo  No NVIDIA GPU detected. Installing CPU-only PyTorch.
    set "USE_CUDA=0"
    set "CUDA_INDEX_URL="
    goto :eof
)

rem Parse GPU name
for /f "tokens=*" %%g in ('nvidia-smi --query-gpu=name --format^=csv^,noheader^,nounits 2^>nul') do (
    set "GPU_NAME=%%g"
)

rem Parse CUDA version from nvidia-smi header
for /f "tokens=9 delims= " %%v in ('nvidia-smi ^| findstr /c:"CUDA Version"') do (
    set "CUDA_VER_RAW=%%v"
)
rem Strip trailing pipe/whitespace
for /f "tokens=1 delims=| " %%c in ("!CUDA_VER_RAW!") do set "CUDA_VER=%%c"

rem Extract major version
for /f "tokens=1 delims=." %%m in ("!CUDA_VER!") do set "CUDA_MAJOR=%%m"

echo.
echo  Detected GPU: !GPU_NAME!
echo  CUDA driver version: !CUDA_VER!

if "!CUDA_MAJOR!"=="12" goto gpu_cuda12
if "!CUDA_MAJOR!"=="11" goto gpu_cuda11
goto gpu_old

:gpu_cuda12
set "USE_CUDA=1"
set "CUDA_INDEX_URL=https://download.pytorch.org/whl/cu124"
set "PYTORCH_CUDA_TAG=12.4"
echo  PyTorch CUDA target: 12.4
goto :eof

:gpu_cuda11
set "USE_CUDA=1"
set "CUDA_INDEX_URL=https://download.pytorch.org/whl/cu118"
set "PYTORCH_CUDA_TAG=11.8"
echo  PyTorch CUDA target: 11.8
goto :eof

:gpu_old
echo  WARNING: CUDA !CUDA_VER! is too old for current PyTorch. Installing CPU-only.
set "USE_CUDA=0"
set "CUDA_INDEX_URL="
goto :eof

rem ================================================================
rem  VENV SETUP
rem ================================================================
:setup_venv

echo.
echo  DL Pixel Classifier - Server Setup [venv]
echo  ==========================================
echo.
echo  Server source: %SERVER_DIR%
echo  Virtual env:   %ENV_PATH%
if "%USE_CUDA%"=="1" (
    echo  GPU support:   CUDA !PYTORCH_CUDA_TAG!
) else (
    echo  GPU support:   CPU only
)
echo.

rem Check Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo  ERROR: Python not found in PATH.
    echo  Install Python 3.10+ from https://www.python.org/downloads/
    echo  Or use --conda flag if you have Anaconda/Miniconda installed.
    exit /b 1
)

for /f "tokens=*" %%i in ('python --version 2^>^&1') do set "PY_VER=%%i"
echo  Found: %PY_VER%
echo.

rem Create virtual environment
if not exist "%ENV_PATH%\Scripts\activate.bat" goto venv_create
echo  Virtual environment already exists at %ENV_PATH%
set /p RECREATE="  Recreate it? [y/N]: "
if /i "!RECREATE!"=="y" (
    echo  Removing old environment...
    rmdir /s /q "%ENV_PATH%"
    goto venv_create
)
echo  Reusing existing environment.
goto venv_install

:venv_create
echo  Creating virtual environment at %ENV_PATH% ...
python -m venv "%ENV_PATH%"
if errorlevel 1 (
    echo  ERROR: Failed to create virtual environment.
    exit /b 1
)
echo  Virtual environment created.

:venv_install
echo.
echo  Activating environment and installing dependencies...
echo  [This may take several minutes for PyTorch download]
echo.

call "%ENV_PATH%\Scripts\activate.bat"
python -m pip install --upgrade pip >nul 2>&1

if "%USE_CUDA%"=="0" goto venv_install_cpu

echo  Installing PyTorch with CUDA %PYTORCH_CUDA_TAG% support...
pip install torch torchvision --index-url %CUDA_INDEX_URL%
if errorlevel 1 (
    echo.
    echo  ERROR: PyTorch CUDA installation failed. Check the output above.
    exit /b 1
)
echo.
echo  Installing server package with GPU extras...
pip install -e "%SERVER_DIR%[cuda]"
goto venv_install_done

:venv_install_cpu
echo  Installing CPU-only...
pip install -e "%SERVER_DIR%"

:venv_install_done
if errorlevel 1 (
    echo.
    echo  ERROR: Installation failed. Check the output above.
    exit /b 1
)

rem Save config
echo ENV_TYPE=venv> "%CONFIG_FILE%"
echo ENV_PATH=%ENV_PATH%>> "%CONFIG_FILE%"
echo SERVER_DIR=%SERVER_DIR%>> "%CONFIG_FILE%"

rem Create start script
echo @echo off> "%SCRIPT_DIR%start_server.bat"
echo rem Auto-generated by setup_server.bat [venv mode]>> "%SCRIPT_DIR%start_server.bat"
echo call "%ENV_PATH%\Scripts\activate.bat">> "%SCRIPT_DIR%start_server.bat"
echo echo Starting DL Pixel Classifier server...>> "%SCRIPT_DIR%start_server.bat"
echo echo Press Ctrl+C to stop.>> "%SCRIPT_DIR%start_server.bat"
echo echo.>> "%SCRIPT_DIR%start_server.bat"
echo dlclassifier-server>> "%SCRIPT_DIR%start_server.bat"

goto finish

rem ================================================================
rem  CONDA SETUP
rem ================================================================
:setup_conda

rem Check if ENV_PATH looks like a directory path (contains \ or : or /)
set "CONDA_USE_PREFIX=0"
echo !ENV_PATH! | findstr /c:"\" >nul 2>&1
if not errorlevel 1 set "CONDA_USE_PREFIX=1"
echo !ENV_PATH! | findstr /c:":" >nul 2>&1
if not errorlevel 1 set "CONDA_USE_PREFIX=1"
echo !ENV_PATH! | findstr /c:"/" >nul 2>&1
if not errorlevel 1 set "CONDA_USE_PREFIX=1"

echo.
echo  DL Pixel Classifier - Server Setup [conda]
echo  ============================================
echo.
echo  Server source: %SERVER_DIR%
if "%CONDA_USE_PREFIX%"=="1" (
    echo  Conda env at:  %ENV_PATH%
) else (
    echo  Conda env:     %ENV_PATH%
)
if "%USE_CUDA%"=="1" (
    echo  GPU support:   CUDA !PYTORCH_CUDA_TAG!
) else (
    echo  GPU support:   CPU only
)
echo.

rem Check conda is available
call conda --version >nul 2>&1
if errorlevel 1 (
    echo  ERROR: conda not found.
    echo  Run this from an Anaconda Prompt, or add conda to PATH.
    exit /b 1
)

for /f "tokens=*" %%i in ('call conda --version 2^>^&1') do set "CONDA_VER=%%i"
echo  Found: %CONDA_VER%
echo.

rem Check if env already exists
if "%CONDA_USE_PREFIX%"=="1" goto conda_check_prefix
goto conda_check_name

:conda_check_prefix
if not exist "%ENV_PATH%\conda-meta" goto conda_create
echo  Conda environment already exists at %ENV_PATH%
set /p RECREATE="  Recreate it? [y/N]: "
if /i "!RECREATE!"=="y" (
    echo  Removing old environment...
    call conda env remove --prefix "%ENV_PATH%" -y >nul 2>&1
    goto conda_create
)
echo  Reusing existing environment.
goto conda_install

:conda_check_name
call conda env list 2>nul | findstr /c:"%ENV_PATH%" >nul 2>&1
if errorlevel 1 goto conda_create
echo  Conda environment "%ENV_PATH%" already exists.
set /p RECREATE="  Recreate it? [y/N]: "
if /i "!RECREATE!"=="y" (
    echo  Removing old environment...
    call conda env remove --name "%ENV_PATH%" -y >nul 2>&1
    goto conda_create
)
echo  Reusing existing environment.
goto conda_install

:conda_create
if "%CONDA_USE_PREFIX%"=="1" (
    echo  Creating conda environment at %ENV_PATH% ...
    call conda create --prefix "%ENV_PATH%" python=3.11 -y
) else (
    echo  Creating conda environment "%ENV_PATH%" ...
    call conda create --name "%ENV_PATH%" python=3.11 -y
)
if errorlevel 1 (
    echo  ERROR: Failed to create conda environment.
    exit /b 1
)
echo  Conda environment created.

:conda_install
echo.
echo  Activating environment and installing dependencies...
echo  [This may take several minutes for PyTorch download]
echo.

if "%CONDA_USE_PREFIX%"=="1" (
    call conda activate "%ENV_PATH%"
) else (
    call conda activate %ENV_PATH%
)

rem Install PyTorch via conda (better CUDA toolkit bundling)
if "%USE_CUDA%"=="0" goto conda_pytorch_cpu
if "!CUDA_MAJOR!"=="12" goto conda_pytorch_cu12
if "!CUDA_MAJOR!"=="11" goto conda_pytorch_cu11
goto conda_pytorch_cu12_default

:conda_pytorch_cu12
echo  Installing PyTorch with CUDA 12.4 via conda...
call conda install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia -y
if errorlevel 1 goto conda_pytorch_pip_fallback
goto conda_pytorch_done

:conda_pytorch_cu11
echo  Installing PyTorch with CUDA 11.8 via conda...
call conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y
if errorlevel 1 goto conda_pytorch_pip_fallback
goto conda_pytorch_done

:conda_pytorch_cu12_default
echo  Installing PyTorch with CUDA 12.4 via conda [default]...
call conda install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia -y
if errorlevel 1 goto conda_pytorch_pip_fallback
goto conda_pytorch_done

:conda_pytorch_cpu
echo  Installing PyTorch CPU via conda...
call conda install pytorch torchvision cpuonly -c pytorch -y
if errorlevel 1 goto conda_pytorch_pip_fallback
goto conda_pytorch_done

:conda_pytorch_pip_fallback
echo  WARNING: conda PyTorch install failed, falling back to pip...
if "%USE_CUDA%"=="1" (
    pip install torch torchvision --index-url %CUDA_INDEX_URL%
) else (
    pip install torch torchvision
)

:conda_pytorch_done

rem Install the server package via pip (fastapi, segmentation-models-pytorch, etc.)
rem PyTorch is already installed above, pip will skip it.
echo.
echo  Installing server package via pip...
if "%USE_CUDA%"=="1" (
    pip install -e "%SERVER_DIR%[cuda]"
) else (
    pip install -e "%SERVER_DIR%"
)
if errorlevel 1 (
    echo.
    echo  ERROR: pip install failed. Check the output above.
    exit /b 1
)

rem Save config
echo ENV_TYPE=conda> "%CONFIG_FILE%"
echo ENV_PATH=%ENV_PATH%>> "%CONFIG_FILE%"
echo CONDA_USE_PREFIX=%CONDA_USE_PREFIX%>> "%CONFIG_FILE%"
echo SERVER_DIR=%SERVER_DIR%>> "%CONFIG_FILE%"

rem Create start script for conda
echo @echo off> "%SCRIPT_DIR%start_server.bat"
echo rem Auto-generated by setup_server.bat [conda mode]>> "%SCRIPT_DIR%start_server.bat"
if "%CONDA_USE_PREFIX%"=="1" (
    echo call conda activate "%ENV_PATH%">> "%SCRIPT_DIR%start_server.bat"
) else (
    echo call conda activate %ENV_PATH%>> "%SCRIPT_DIR%start_server.bat"
)
echo echo Starting DL Pixel Classifier server...>> "%SCRIPT_DIR%start_server.bat"
echo echo Press Ctrl+C to stop.>> "%SCRIPT_DIR%start_server.bat"
echo echo.>> "%SCRIPT_DIR%start_server.bat"
echo dlclassifier-server>> "%SCRIPT_DIR%start_server.bat"

goto finish

rem ================================================================
rem  FINISH
rem ================================================================
:finish

echo.
echo  ==========================================
echo  Setup complete!
echo  ==========================================
echo.
echo  To start the server:
echo    start_server.bat
echo.

rem Quick GPU check
echo  Checking GPU availability...
python -c "import torch; print('  PyTorch ' + torch.__version__); print('  CUDA available: ' + str(torch.cuda.is_available())); print('  Device: ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else '  CPU mode')"
echo.

rem Ask if user wants to start server now
set /p START_NOW="  Start server now? [Y/n]: "
if /i "!START_NOW!"=="n" goto end
echo.
dlclassifier-server

:end
endlocal
exit /b 0

:show_help
echo.
echo  Usage: setup_server.bat [ENV_PATH] [--cuda] [--cpu] [--conda]
echo.
echo  GPU Detection:
echo    By default, the script auto-detects NVIDIA GPUs via nvidia-smi.
echo    If a GPU is found, CUDA-enabled PyTorch is installed automatically.
echo    If no GPU is found, CPU-only PyTorch is installed.
echo.
echo  Arguments:
echo    ENV_PATH     Path or name for the environment
echo                   venv mode:  directory path [default: python_server\venv]
echo                   conda mode: env name or path [default: "dlclassifier"]
echo    --cuda       Force CUDA install [error if no GPU detected]
echo    --cpu        Force CPU-only install [skip GPU detection]
echo    --conda      Use conda instead of venv [run from Anaconda Prompt]
echo.
echo  Examples [venv]:
echo    setup_server.bat                              Auto-detect GPU, default location
echo    setup_server.bat --cpu                        Force CPU, default location
echo    setup_server.bat --cuda                       Force CUDA, default location
echo    setup_server.bat F:\dl-classifier-env         Auto-detect, custom path
echo.
echo  Examples [conda]:
echo    setup_server.bat --conda                      Auto-detect GPU, env "dlclassifier"
echo    setup_server.bat --conda myenv --cuda         Force CUDA, env "myenv"
echo    setup_server.bat --conda F:\dl-env --cpu      Force CPU, custom path
echo.
exit /b 0
