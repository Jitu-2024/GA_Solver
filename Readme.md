Ensure you have cmake installed

# In your GA_Solver directory
rm -r build
mkdir build
cd build

cmake ..


Ensure you have the nvidia toolkit installed and a working GPU available on your system
nvidia-smi
nvcc --version



Make the .venv virtual environment 



# Remove previous build artifacts
rm -rf build/
rm -rf *.egg-info/

# Reinstall in development mode
pip install -e .


# Remove previous build artifacts (PowerShell)
Remove-Item -Recurse -Force build -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force *.egg-info -ErrorAction SilentlyContinue

# Reinstall in development mode
pip install -e .