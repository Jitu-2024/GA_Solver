from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from pybind11.setup_helpers import Pybind11Extension, build_ext
import sys
import os
import subprocess
# import pybind11
import setuptools.command.develop

class BuildDevelop(setuptools.command.develop.develop):
    def run(self):
        try:
            self.run_command("build_ext")
        except Exception as e:
            print(f"Warning: build_ext failed with {e}")
        super().run()

class CMakeExtension(Extension):
    def __init__(self, name):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath("")

class CMakeBuild(build_ext):
    """Handles building with CMake"""
    def build_extension(self, ext):
        # Ensure pybind11 is installed before running CMake
        try:
            import pybind11
        except ImportError:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pybind11'])
            import pybind11

        # Use a fixed build directory in the project
        self.build_temp = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'build')#, 'temp')
        self.build_lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'build', 'lib')
        
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        if not os.path.exists(self.build_lib):
            os.makedirs(self.build_lib)

        # Get pybind11 cmake directory
        pybind11_cmake_dir = pybind11.get_cmake_dir()

        # Debug output
        print(f"+++++++++++++++++++++++++++++++++++++++Build directory: {self.build_temp}")
        print(f"+++++++++++++++++++++++++++++++++++++++Source directory: {ext.sourcedir}")
        print(f"+++++++++++++++++++++++++++++++++++++++pybind11 directory: {pybind11_cmake_dir}")
        
        print(f"+++++++++++++++++++++++++++++++++++++++sys.executable: {sys.executable}")
        print(f"+++++++++++++++++++++++++++++++++++++++os.path.abspath(self.build_lib): {os.path.abspath(self.build_lib)}")
        print(f"+++++++++++++++++++++++++++++++++++++++os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name))): {os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))}")

        # cmake_args = [
        #     f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))}",
        #     f"-DPYTHON_EXECUTABLE={sys.executable}",
        #     "-DCMAKE_BUILD_TYPE=Release",
        #     f"-Dpybind11_DIR={pybind11_cmake_dir}",
        # ]
        
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={os.path.abspath(self.build_lib)}", #C:\Users\justi\Documents\ELE-CPE Research\Gen_AI_Consultancy\GA-tool\GA_Solver
            # f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-Dpybind11_DIR={pybind11.get_cmake_dir()}",
            "-DPYTHON_MODULE_NAME=_core"
        ]

        build_args = ['--config', 'Release']
        
        env = os.environ.copy()
        env['CXXFLAGS'] = f"{env.get('CXXFLAGS', '')} -DVERSION_INFO=\\\"{self.distribution.get_version()}\\\""

        print(f"+++++++++++++++++++++++++++++++++++++++cmake_args: {cmake_args}")
        
        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args,
            cwd=self.build_temp,
            env=env
        )

        print(f"+++++++++++++++++++++++++++++++++++++++build_args: {build_args}")
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args,
            cwd=self.build_temp,
            env=env
        )

        print(f"+++++++++++++++++++++++++++++++++++++++check_call completion")
        # subprocess.check_call(
        #     ["cmake", ext.sourcedir] + cmake_args,
        #     cwd=self.build_temp
        # )
        # subprocess.check_call(
        #     ["cmake", "--build", "."] + build_args,
        #     cwd=self.build_temp
        # )
        
        
# ext_modules = [
#     Pybind11Extension("gasolver._core", ["gasolver/python_bindings.cpp"])
# ]


setup(
    name="gasolver",
    version="0.1",
    author="Harishjitu Saseendran",
    description="CUDA-accelerated GA Solver for TSP",
    long_description="",
    packages=find_packages(["gasolver"]),
    #package_dir={"gasolver": "src/python"},
    ext_modules=[CMakeExtension("gasolver._core")],
    cmdclass={
        "build_ext": CMakeBuild,
        "develop": BuildDevelop,
    },
    zip_safe=False,
    python_requires=">=3.6",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.0.0",
        "pybind11>=2.10.0"
    ]
)

# from setuptools import setup, Extension
# from setuptools.command.build_ext import build_ext
# import sys
# import os

# class CustomBuildExt(build_ext):
#     def build_extensions(self):
#         cuda_flags = ['-x', 'cuda', '--cuda-gpu-arch=sm_60']
#         for ext in self.extensions:
#             ext.extra_compile_args += cuda_flags
#         build_ext.build_extensions(self)

# # Get all CUDA source files
# cuda_sources = [
#     os.path.join('src/cuda', f) 
#     for f in os.listdir('src/cuda') 
#     if f.endswith('.cu')
# ]

# ext_modules = [
#     Extension(
#         "gasolver._core",
#         ["src/python/python_bindings.cpp"] + cuda_sources,
#         include_dirs=[
#             "src/include",
#             "src/cuda",
#             "/usr/local/cuda/include",
#             "pybind11/include"
#         ],
#         library_dirs=["/usr/local/cuda/lib64"],
#         libraries=["cudart"],
#         extra_compile_args=["-std=c++11"],
#         extra_link_args=["-std=c++11"],
#         language="c++"
#     ),
# ]

# setup(
#     name="gasolver",
#     version="0.1",
#     packages=['gasolver'],
#     package_dir={'gasolver': 'src/python'},
#     ext_modules=ext_modules,
#     cmdclass={"build_ext": CustomBuildExt},
#     install_requires=[
#         'numpy>=1.20.0',
#         'pandas>=1.0.0',
#     ],
#     python_requires=">=3.6",
# ) 
