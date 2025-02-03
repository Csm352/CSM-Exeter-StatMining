PyMCubes Installation and Setup Guide

Prerequisites

Before installing PyMCubes, ensure you have the following:

Anaconda installed

Microsoft Visual C++ Build Tools (Windows users)

CMake installed (for compiling C extensions)

Git installed

Step 1: Create and Activate a Conda Environment

conda create --name geology-env python=3.9 -y
conda activate geology-env

Replace geology-env with your preferred environment name if needed.

Step 2: Install Git and Clone the Repository

conda install git -y
git --version  # Verify Git installation
git clone https://github.com/pmneila/PyMCubes.git
cd PyMCubes

Step 3: Install Dependencies

pip install --upgrade pip setuptools wheel
conda install numpy cython -y

Step 4: Install C++ Build Tools and CMake (Windows Users)
download and install from website

Step 4: Verify C++ Build Tools and CMake (Windows Users)

Open the Developer Command Prompt for Visual Studio and run:

cl  # Check for MSVC Compiler Toolset
dir "C:\Program Files (x86)\Windows Kits\10"  # Check for Windows 10 SDK
cmake --version  # Verify CMake installation

Step 5: Build and Install PyMCubes
Make sure you are in the PyMCubes directory in the environment
python -m pip install .

This will compile and install PyMCubes, including its Cython extensions.

Step 6: Verify Installation

python -c "import mcubes; print(mcubes.__version__)"

Make sure to install packages: scipy, plotly, and ezdxf

Now you're ready to use PyMCubes!

====================================================
Troubleshooting (just in case)

Reinstall PyMCubes

python -m pip uninstall PyMCubes -y
python -m pip install .

Manually Build the C Extension

python setup.py build_ext --inplace
python -m pip install .

Verify C Compiler

cl  # Should return the compiler version

Optional: Install Precompiled Binaries

If compilation issues persist, check for precompiled .whl files and install manually:

python -m pip install path_to_wheel_file.whl

Install scipy, plotly, and ezdxf Python libraries into your envoronment (geology-env in this case)

Store the GridWithRocktypes.csv data file at the environment location

Now you're ready to use PyMCubes!
