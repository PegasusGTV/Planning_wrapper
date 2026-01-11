# Windows Setup Guide for Planning Wrapper

This guide provides step-by-step instructions to set up and run the Planning Wrapper demos on Windows using Command Prompt or PowerShell.

## Prerequisites

- Windows 10/11
- Administrator access (for some installations)
- Internet connection

## Step 1: Install Python

### Option A: Using winget (Recommended)
```powershell
winget install Python.Python.3.12 --silent --accept-package-agreements --accept-source-agreements
```

### Option B: Manual Installation
1. Download Python 3.9+ from https://www.python.org/downloads/
2. **Important**: Check "Add Python to PATH" during installation
3. Verify installation:
   ```powershell
   python --version
   ```

## Step 2: Install Miniconda

### Using winget (Recommended)
```powershell
winget install Anaconda.Miniconda3 --silent --accept-package-agreements --accept-source-agreements
```

### Manual Installation
1. Download Miniconda from https://docs.conda.io/en/latest/miniconda.html
2. Run the installer and follow the prompts
3. Accept the Terms of Service when prompted

## Step 3: Initialize Conda

Open PowerShell and run:
```powershell
& "$env:USERPROFILE\miniconda3\Scripts\conda.exe" init powershell
```

**Note**: You may need to close and reopen PowerShell for changes to take effect.

## Step 4: Accept Conda Terms of Service

```powershell
& "$env:USERPROFILE\miniconda3\Scripts\conda.exe" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
& "$env:USERPROFILE\miniconda3\Scripts\conda.exe" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
& "$env:USERPROFILE\miniconda3\Scripts\conda.exe" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2
```

## Step 5: Install Git (Required for ManiSkill3)

```powershell
winget install Git.Git --silent --accept-package-agreements --accept-source-agreements
```

## Step 6: Install Visual C++ Redistributables

```powershell
winget install Microsoft.VCRedist.2015+.x64 --silent --accept-package-agreements --accept-source-agreements
```

**Note**: You may need to restart your computer after installing Visual C++ Redistributables.

## Step 7: Navigate to Project Directory

```powershell
cd C:\Users\mirak\Downloads\Planning_wrapper-main
```

*(Replace with your actual project path)*

## Step 8: Create Conda Environment

```powershell
& "$env:USERPROFILE\miniconda3\Scripts\conda.exe" create -n planning_wrapper python=3.11 -y
```

## Step 9: Install Project Dependencies

```powershell
& "$env:USERPROFILE\miniconda3\Scripts\conda.exe" run -n planning_wrapper python -m pip install --upgrade pip
& "$env:USERPROFILE\miniconda3\Scripts\conda.exe" run -n planning_wrapper python -m pip install -e .
```

## Step 10: Install ManiSkill3

```powershell
& "$env:USERPROFILE\miniconda3\Scripts\conda.exe" run -n planning_wrapper python -m pip install git+https://github.com/haosulab/ManiSkill.git
```

This may take several minutes as it downloads and installs many dependencies.

## Step 11: Install Pinocchio

```powershell
& "$env:USERPROFILE\miniconda3\Scripts\conda.exe" install -n planning_wrapper -c conda-forge pinocchio -y
```

## Step 12: Fix PyTorch Compatibility

Install a compatible version of PyTorch:
```powershell
& "$env:USERPROFILE\miniconda3\Scripts\conda.exe" run -n planning_wrapper python -m pip uninstall torch -y
& "$env:USERPROFILE\miniconda3\Scripts\conda.exe" run -n planning_wrapper python -m pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu
```

## Step 13: Fix NumPy Compatibility

Downgrade NumPy to version 1.x for compatibility:
```powershell
& "$env:USERPROFILE\miniconda3\Scripts\conda.exe" run -n planning_wrapper python -m pip install "numpy<2"
```

## Step 14: Verify Installation

Test that everything is installed correctly:
```powershell
& "$env:USERPROFILE\miniconda3\Scripts\conda.exe" run -n planning_wrapper python -c "import planning_wrapper; print('✅ planning_wrapper installed!')"
& "$env:USERPROFILE\miniconda3\Scripts\conda.exe" run -n planning_wrapper python -c "import mani_skill; print('✅ ManiSkill3 installed!')"
& "$env:USERPROFILE\miniconda3\Scripts\conda.exe" run -n planning_wrapper python -c "import pinocchio; print('✅ pinocchio installed!')"
& "$env:USERPROFILE\miniconda3\Scripts\conda.exe" run -n planning_wrapper python -c "import torch; print('✅ PyTorch version:', torch.__version__)"
```

## Running the Demos

### Using PowerShell (Recommended: With setup_conda.ps1)

**First, load the convenience script** (only needed once per PowerShell session):
```powershell
cd C:\Users\mirak\Downloads\Planning_wrapper-main
. .\setup_conda.ps1
```

**Then activate and run:**
```powershell
conda activate planning_wrapper
python -m examples.pusht_demo3
```

Or use `conda run`:
```powershell
conda run -n planning_wrapper python -m examples.pusht_demo3
```

### Using PowerShell (Without setup_conda.ps1)

Navigate to the project directory and run:

**PushT Demo 3:**
```powershell
cd C:\Users\mirak\Downloads\Planning_wrapper-main
& "$env:USERPROFILE\miniconda3\Scripts\conda.exe" run -n planning_wrapper python -m examples.pusht_demo3
```

**Shelf Demo 3:**
```powershell
cd C:\Users\mirak\Downloads\Planning_wrapper-main
& "$env:USERPROFILE\miniconda3\Scripts\conda.exe" run -n planning_wrapper python -m examples.shelf_demo3
```

### Using Command Prompt (CMD)

**PushT Demo 3:**
```cmd
cd C:\Users\mirak\Downloads\Planning_wrapper-main
"%USERPROFILE%\miniconda3\Scripts\conda.exe" run -n planning_wrapper python -m examples.pusht_demo3
```

**Shelf Demo 3:**
```cmd
cd C:\Users\mirak\Downloads\Planning_wrapper-main
"%USERPROFILE%\miniconda3\Scripts\conda.exe" run -n planning_wrapper python -m examples.shelf_demo3
```

### Other Available Demos

You can run any of the following demos using the same pattern:

- `python -m examples.pusht_demo1`
- `python -m examples.pusht_demo2`
- `python -m examples.pusht_demo3`
- `python -m examples.pusht_demo4`
- `python -m examples.shelf_demo1`
- `python -m examples.shelf_demo2`
- `python -m examples.shelf_demo3`
- `python -m examples.smoke_test`
- `python -m examples.replay_smoke`
- `python -m examples.replay_shelf_smoke`

## Easy Conda Access: Using setup_conda.ps1 (Recommended)

To avoid typing the full conda path every time, use the convenience script `setup_conda.ps1`:

### First Time Setup

1. **Set Execution Policy** (one-time, if needed):
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```
   Or bypass for just this script:
   ```powershell
   powershell -ExecutionPolicy Bypass -File .\setup_conda.ps1
   ```

2. **Load the script** in your PowerShell session:
   ```powershell
   . .\setup_conda.ps1
   ```

   The dot (`.`) before the script is important - it runs the script in the current session.

### After Running setup_conda.ps1

You can now use conda commands directly without the full path:

```powershell
conda activate planning_wrapper
conda list
conda deactivate
conda info
```

Or use the shortcut:
```powershell
conda-activate planning_wrapper
```

**Note**: You need to run `. .\setup_conda.ps1` in each new PowerShell session, or add it to your PowerShell profile for automatic loading.

## Alternative: Activating the Environment (Without setup_conda.ps1)

If you prefer to activate the environment first (instead of using `conda run`):

### PowerShell:
```powershell
& "$env:USERPROFILE\miniconda3\Scripts\activate.bat" planning_wrapper
python -m examples.pusht_demo3
```

### Command Prompt:
```cmd
"%USERPROFILE%\miniconda3\Scripts\activate.bat" planning_wrapper
python -m examples.pusht_demo3
```

## Troubleshooting

### Issue: "conda is not recognized"
**Solution 1** (Recommended): Use the convenience script:
```powershell
. .\setup_conda.ps1
conda --version
```

**Solution 2**: Use the full path to conda.exe:
```powershell
& "$env:USERPROFILE\miniconda3\Scripts\conda.exe" --version
```

### Issue: PyTorch DLL errors
**Solution**: 
1. Make sure Visual C++ Redistributables are installed
2. Restart your computer
3. Try installing PyTorch 2.3.0 as shown in Step 12

### Issue: NumPy compatibility errors
**Solution**: Make sure NumPy is version 1.x:
```powershell
& "$env:USERPROFILE\miniconda3\Scripts\conda.exe" run -n planning_wrapper python -m pip install "numpy<2"
```

### Issue: Pinocchio not found
**Solution**: Install via conda-forge:
```powershell
& "$env:USERPROFILE\miniconda3\Scripts\conda.exe" install -n planning_wrapper -c conda-forge pinocchio -y
```

### Issue: Git not found when installing ManiSkill3
**Solution**: Install Git using winget (see Step 5)

### Issue: Execution Policy errors in PowerShell
**Solution 1** (Recommended): Bypass execution policy for just the script:
```powershell
powershell -ExecutionPolicy Bypass -File .\setup_conda.ps1
```

**Solution 2**: Set execution policy permanently (run PowerShell as Administrator):
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
Then you can use:
```powershell
. .\setup_conda.ps1
```

## Quick Reference: All Setup Commands

Here's a complete script you can copy and paste (adjust the project path as needed):

```powershell
# Navigate to project
cd C:\Users\mirak\Downloads\Planning_wrapper-main

# Create environment
& "$env:USERPROFILE\miniconda3\Scripts\conda.exe" create -n planning_wrapper python=3.11 -y

# Install project
& "$env:USERPROFILE\miniconda3\Scripts\conda.exe" run -n planning_wrapper python -m pip install -e .

# Install ManiSkill3
& "$env:USERPROFILE\miniconda3\Scripts\conda.exe" run -n planning_wrapper python -m pip install git+https://github.com/haosulab/ManiSkill.git

# Install pinocchio
& "$env:USERPROFILE\miniconda3\Scripts\conda.exe" install -n planning_wrapper -c conda-forge pinocchio -y

# Fix PyTorch
& "$env:USERPROFILE\miniconda3\Scripts\conda.exe" run -n planning_wrapper python -m pip uninstall torch -y
& "$env:USERPROFILE\miniconda3\Scripts\conda.exe" run -n planning_wrapper python -m pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu

# Fix NumPy
& "$env:USERPROFILE\miniconda3\Scripts\conda.exe" run -n planning_wrapper python -m pip install "numpy<2"

# Test
& "$env:USERPROFILE\miniconda3\Scripts\conda.exe" run -n planning_wrapper python -m examples.pusht_demo3
```

## System Requirements

- **Python**: 3.9 or higher (3.11 recommended)
- **RAM**: At least 8GB recommended
- **Disk Space**: ~5GB for all dependencies
- **OS**: Windows 10/11 (64-bit)

## Notes

- The first run of any demo may take longer as it initializes the environment
- Some demos require rendering which may open a window
- If you encounter DLL errors, restart your computer after installing Visual C++ Redistributables
- The conda environment name `planning_wrapper` can be changed if desired
- **Tip**: Run `. .\setup_conda.ps1` at the start of each PowerShell session to enable easy conda commands
- To make `setup_conda.ps1` load automatically, add it to your PowerShell profile:
  ```powershell
  Add-Content $PROFILE "`n. `"$PWD\setup_conda.ps1`""
  ```

## Getting Help

If you encounter issues not covered in this guide:
1. Check the main [README.md](README.md) for general information
2. Review [SETUP_INSTRUCTIONS.md](SETUP_INSTRUCTIONS.md) for additional details
3. Check the ManiSkill3 documentation: https://github.com/haosulab/ManiSkill

---

**Last Updated**: Based on successful setup on Windows 10/11 with Python 3.11, PyTorch 2.3.0, and NumPy 1.26.4