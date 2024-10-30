import ctypes
import os
import sys
from pathlib import Path


# Load the library
def _load_shared_library(lib_base_name: str, base_path: Path = None):
    # Determine the file extension based on the platform
    if sys.platform.startswith("linux"):
        lib_ext = ".so"
    elif sys.platform == "darwin":
        lib_ext = ".dylib"
    elif sys.platform == "win32":
        lib_ext = ".dll"
    else:
        raise RuntimeError("Unsupported platform")

    # Construct the paths to the possible shared library names
    if base_path is None:
        _base_path = Path(__file__).parent.parent.resolve()
    else:
        print(f"Using base path: {base_path}")
        _base_path = base_path
    _lib_paths = [
        _base_path / f"lib{lib_base_name}{lib_ext}",
        _base_path / f"{lib_base_name}{lib_ext}",
    ]

    # Add the library directory to the DLL search path on Windows (if needed)
    if sys.platform == "win32" and sys.version_info >= (3, 8):
        os.add_dll_directory(str(_base_path))

    # Try to load the shared library, handling potential errors
    for _lib_path in _lib_paths:
        print(f"Trying to load shared library '{_lib_path}'")
        if _lib_path.exists():
            try:
                return ctypes.CDLL(str(_lib_path))
            except Exception as e:
                print(f"Failed to load shared library '{_lib_path}': {e}")

    raise FileNotFoundError(
        f"Shared library with base name '{lib_base_name}' not found"
    )


# Specify the base name of the shared library to load
_lib_base_name = "hf-qwen2-audio_shared"
base_path = (
    Path(__file__).parent.parent.parent.resolve()
    / "build"
    / "examples"
    / "qwen2-audio"
)

# Load the library
_lib = _load_shared_library(_lib_base_name, base_path)

#   conda config --add channels conda-forge
#   conda update libstdcxx-ng


# struct omni_context_params
# {
#     char *model;
#     char *mmproj;
#     char *file;
#     char *prompt;
#     int32_t n_gpu_layers;
# };
class omni_context_params(ctypes.Structure):
    _fields_ = [
        ("model", ctypes.c_char_p),
        ("mmproj", ctypes.c_char_p),
        ("file", ctypes.c_char_p),
        ("prompt", ctypes.c_char_p),
        ("n_gpu_layers", ctypes.c_int32),
    ]


omni_context_params_p = ctypes.POINTER(omni_context_params)

omni_context_p = ctypes.c_void_p


# OMNI_AUDIO_API omni_context_params omni_context_default_params();
def omni_context_default_params() -> omni_context_params:
    return _lib.omni_context_default_params()


_lib.omni_context_default_params.argtypes = []
_lib.omni_context_default_params.restype = omni_context_params


# OMNI_AUDIO_API struct omni_context *omni_init_context(omni_context_params &params);
def omni_init_context(params: omni_context_params_p) -> omni_context_p:  # type: ignore
    return _lib.omni_init_context(params)


_lib.omni_init_context.argtypes = [omni_context_params_p]
_lib.omni_init_context.restype = omni_context_p


# OMNI_AUDIO_API void omni_process_full(
#     struct omni_context *ctx_omni,
#     omni_context_params &params
# );
def omni_process_full(ctx: omni_context_p, params: omni_context_params_p):  # type: ignore
    return _lib.omni_process_full(ctx, params)


_lib.omni_process_full.argtypes = [omni_context_p, omni_context_params_p]
_lib.omni_process_full.restype = None


# OMNI_AUDIO_API void omni_free(struct omni_context *ctx_omni);
def omni_free(ctx: omni_context_p):
    return _lib.omni_free(ctx)


_lib.omni_free.argtypes = [omni_context_p]
_lib.omni_free.restype = None