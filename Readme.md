Accidentally uploaded .venv and .vscode to this branch, just remove both also. 

Ensure you have cmake installed


Ensure you have the nvidia toolkit installed and a working GPU available on your system
nvidia-smi
nvcc --version


# Remove previous build artifacts
rm -rf .venv
rm -rf .vscode
rm -rf build/
rm -rf *.egg-info/

OR

# Remove previous build artifacts (PowerShell)
Remove-Item -Recurse -Force .venv -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force .vscode -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force build -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force *.egg-info -ErrorAction SilentlyContinue


# Make the .venv virtual environment (in proper root directory)
python -m venv .venv

source .venv/bin/activate
(For cronus server is it source .venv/bin/activate.csh (due to the gentoo and specific shell will has downloaded))

OR

.venv/Scripts/activate



# Reinstall in development mode
pip install -e .
