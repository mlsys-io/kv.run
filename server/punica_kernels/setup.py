import contextlib
import datetime
import itertools
import os
import pathlib
import platform
import re
import subprocess

import setuptools
import torch
import torch.utils.cpp_extension as torch_cpp_ext

root = pathlib.Path(__name__).parent


def glob(pattern):
    return [str(p) for p in root.glob(pattern)]


def remove_unwanted_pytorch_nvcc_flags():
    REMOVE_NVCC_FLAGS = [
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
    ]
    for flag in REMOVE_NVCC_FLAGS:
        with contextlib.suppress(ValueError):
            torch_cpp_ext.COMMON_NVCC_FLAGS.remove(flag)


def get_local_version_suffix() -> str:
    if not (root / ".git").is_dir():
        return ""
    now = datetime.datetime.now()
    git_hash = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=root, text=True
    ).strip()
    commit_number = subprocess.check_output(
        ["git", "rev-list", "HEAD", "--count"], cwd=root, text=True
    ).strip()
    dirty = ".dirty" if subprocess.run(["git", "diff", "--quiet"]).returncode else ""
    return f"+c{commit_number}.d{now:%Y%m%d}.{git_hash}{dirty}"


def get_version() -> str:
    return "1.1.0"


def get_cuda_version() -> tuple[int, int]:
    if torch_cpp_ext.CUDA_HOME is None:
        nvcc = "nvcc"
    else:
        nvcc = os.path.join(torch_cpp_ext.CUDA_HOME, "bin/nvcc")
    txt = subprocess.check_output([nvcc, "--version"], text=True)
    major, minor = map(int, re.findall(r"release (\d+)\.(\d+),", txt)[0])
    return major, minor


def generate_build_meta() -> None:
    d = {}
    version = get_version()
    d["cuda_major"], d["cuda_minor"] = get_cuda_version()
    d["torch"] = torch.__version__
    d["python"] = platform.python_version()
    d["TORCH_CUDA_ARCH_LIST"] = os.environ.get("TORCH_CUDA_ARCH_LIST", None)
    with open(root / "_build_meta.py", "w") as f:
        f.write(f"__version__ = {version!r}\n")
        f.write(f"build_meta = {d!r}")


if __name__ == "__main__":
    remove_unwanted_pytorch_nvcc_flags()
    generate_build_meta()

    ext_modules = []
    ext_modules.append(
        torch_cpp_ext.CUDAExtension(
            name="punica.ops._kernels",
            sources=[
                "csrc/punica_ops.cc",
                "csrc/bgmv/bgmv_all.cu",
                "csrc/rms_norm/rms_norm_cutlass.cu",
                "csrc/sgmv/sgmv_cutlass.cu",
                "csrc/sgmv_flashinfer/sgmv_all.cu",
            ],
            include_dirs=[
                str(root.resolve() / "../third_party/cutlass/include"),
                str(root.resolve() / "../third_party/flashinfer/include"),
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3"],
            },
        )
    )

    setuptools.setup(
        version=get_version(),
        ext_modules=ext_modules,
        cmdclass={"build_ext": torch_cpp_ext.BuildExtension},
    )
