# Build the pyquick Python extension via f2py.
# Enabled with -DPYTHON=TRUE at the cmake configure step.

# ---------------------------------------------------------------------------
# 1. Locate the Python interpreter.
#
#    Honour an explicit -DPython3_EXECUTABLE=... hint if provided.
#    Otherwise resolve "python3" from PATH so that conda/virtualenv
#    environments are found before any system Python.
# ---------------------------------------------------------------------------
if(NOT Python3_EXECUTABLE)
    find_program(Python3_EXECUTABLE
        NAMES python3 python
        DOC "Python interpreter used to run f2py"
        NO_CMAKE_SYSTEM_PATH   # skip CMake's built-in system search first
    )
    if(NOT Python3_EXECUTABLE)
        find_program(Python3_EXECUTABLE NAMES python3 python
            DOC "Python interpreter used to run f2py")
    endif()
endif()

if(NOT Python3_EXECUTABLE)
    message(FATAL_ERROR
        "Python interpreter not found. Set -DPython3_EXECUTABLE=/path/to/python3 "
        "or activate the correct conda/virtualenv environment before running cmake.")
endif()

message(STATUS "pyquick: using Python interpreter: ${Python3_EXECUTABLE}")

# ---------------------------------------------------------------------------
# 2. Verify numpy and f2py are available via the located interpreter
# ---------------------------------------------------------------------------
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import numpy"
    RESULT_VARIABLE _numpy_result
    OUTPUT_QUIET ERROR_QUIET)
if(NOT _numpy_result EQUAL 0)
    message(FATAL_ERROR
        "numpy not importable by ${Python3_EXECUTABLE}. "
        "Install numpy or reconfigure without -DPYTHON=TRUE.")
endif()

execute_process(
    COMMAND ${Python3_EXECUTABLE} -m numpy.f2py -h
    RESULT_VARIABLE _f2py_result
    OUTPUT_QUIET
    ERROR_QUIET)
if(NOT _f2py_result EQUAL 0)
    message(FATAL_ERROR
        "f2py not found (numpy.f2py unavailable for ${Python3_EXECUTABLE}). "
        "Install numpy to build the Python extension, or reconfigure without -DPYTHON=TRUE.")
endif()

# f2py is driven with its meson backend (see the build command below), which
# needs the meson and ninja executables on PATH. Check now for a clear error.
find_program(MESON_EXECUTABLE meson)
find_program(NINJA_EXECUTABLE ninja)
if(NOT MESON_EXECUTABLE OR NOT NINJA_EXECUTABLE)
    message(FATAL_ERROR
        "pyquick requires 'meson' and 'ninja' for f2py's meson build backend, "
        "but they were not found on PATH. Install them for the build Python, e.g.\n"
        "    ${Python3_EXECUTABLE} -m pip install meson ninja\n"
        "then reconfigure (or pass -DPYTHON=FALSE to skip the Python extension).")
endif()

# ---------------------------------------------------------------------------
# 3. Compute the platform-specific extension suffix
#    e.g. .cpython-313-x86_64-linux-gnu.so
# ---------------------------------------------------------------------------
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c
        "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'), end='')"
    OUTPUT_VARIABLE PYTHON_EXT_SUFFIX)

set(_PYQUICK_SO "${CMAKE_CURRENT_BINARY_DIR}/_pyquick${PYTHON_EXT_SUFFIX}")

# ---------------------------------------------------------------------------
# 4. Select the libquick flavour to bind against: serial (CPU) or GPU.
#
#    pyquick.f90 carries GPU code inside #if defined(GPU) guards (mirroring
#    main.f90). The GPU macro is what src/CMakeLists.txt defines on
#    libquick_${QUICK_GPU_TARGET_NAME}, and it covers BOTH the cuda and hip
#    backends -- one source, two build recipes. So a GPU pyquick differs from
#    the CPU one only in these three things:
#
#      * -DGPU on the Fortran compile, to activate the guarded blocks;
#      * the .mod directory (quick/cuda or quick/hip, not quick/serial), which
#        is where the gpu_* interfaces live;
#      * linking libquick_cuda / libquick_hip instead of the serial libquick.
#
#    -cpp and -I<src/util> are unconditional: the guards must be *evaluated*
#    even on the CPU build (where they evaluate to nothing), and preprocessing
#    means "#include util.fh" now has to be resolved by cpp rather than by
#    gfortran's own include handling.
# ---------------------------------------------------------------------------
set(_PYQUICK_F90FLAGS "-cpp -I${CMAKE_SOURCE_DIR}/src/util")
set(_PYQUICK_EXTRA_LINK_ARGS "")

if(CUDA)
    # CUDA=TRUE covers both backends; HIP=TRUE additionally flips
    # QUICK_GPU_TARGET_NAME to "hip" and QUICK_GPU_PLATFORM to "HIP".
    set(_PYQUICK_LIBTARGET libquick_${QUICK_GPU_TARGET_NAME})
    set(_PYQUICK_MODDIR "${CMAKE_BINARY_DIR}/amber-modules/quick/${QUICK_GPU_TARGET_NAME}")
    set(_PYQUICK_LIBNAME "quick_${QUICK_GPU_TARGET_NAME}")
    string(APPEND _PYQUICK_F90FLAGS " -DGPU -D${QUICK_GPU_PLATFORM}")

    if(NOT HIP)
        # libquick_cuda is a shared library, so cublas/cusolver/cudart are
        # normally pulled in transitively. Name them anyway: f2py links with
        # gfortran, not the CUDA toolchain, so nothing guarantees the device
        # runtime is on the link line. FindCUDA gives full paths; f2py's meson
        # backend wants -L/-l, hence the split.
        foreach(_cudalib "${CUDA_CUDART_LIBRARY}" "${CUDA_cublas_LIBRARY}" "${CUDA_cusolver_LIBRARY}")
            if(_cudalib)
                get_filename_component(_cudalibdir "${_cudalib}" DIRECTORY)
                list(APPEND _PYQUICK_EXTRA_LINK_ARGS "-L${_cudalibdir}")
            endif()
        endforeach()
        list(REMOVE_DUPLICATES _PYQUICK_EXTRA_LINK_ARGS)
        list(APPEND _PYQUICK_EXTRA_LINK_ARGS -lcudart -lcublas -lcusolver)
    endif()

    message(STATUS "pyquick: binding against GPU libquick (${_PYQUICK_LIBNAME})")
else()
    set(_PYQUICK_LIBTARGET libquick)
    set(_PYQUICK_MODDIR "${CMAKE_BINARY_DIR}/amber-modules/quick/serial")
    set(_PYQUICK_LIBNAME "quick")
    message(STATUS "pyquick: binding against serial libquick (CPU)")
endif()

# ---------------------------------------------------------------------------
# 5. Build the .so via f2py as a custom command
#
#    Include path: build-tree .mod files for the selected libquick target,
#    set by config_module_dirs(...) in src/CMakeLists.txt and fully populated
#    once that library has compiled (hence DEPENDS ${_PYQUICK_LIBTARGET}).
#
#    Link path: resolved at build time via $<TARGET_FILE_DIR:...>, so -L always
#    points at the actual library location regardless of CMAKE_INSTALL_PREFIX.
#
#    Backend: force the meson backend. f2py's legacy distutils backend is
#    removed on Python 3.12+ and is already broken on 3.9 with modern numpy
#    (numpy.distutils imports distutils.msvccompiler, which no longer exists),
#    so meson is the portable choice. Requires meson + ninja (checked above).
# ---------------------------------------------------------------------------
add_custom_command(
    OUTPUT  "${_PYQUICK_SO}"
    COMMAND ${Python3_EXECUTABLE} -m numpy.f2py
                --backend meson
                -c "${CMAKE_SOURCE_DIR}/src/pyquick/pyquick.f90"
                "--f90flags=${_PYQUICK_F90FLAGS}"
                "-I${_PYQUICK_MODDIR}"
                "-L$<TARGET_FILE_DIR:${_PYQUICK_LIBTARGET}>"
                "-l${_PYQUICK_LIBNAME}"
                ${_PYQUICK_EXTRA_LINK_ARGS}
                -m _pyquick
    DEPENDS ${_PYQUICK_LIBTARGET} "${CMAKE_SOURCE_DIR}/src/pyquick/pyquick.f90"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    COMMENT "Building _pyquick Python extension via f2py"
    VERBATIM)

# ---------------------------------------------------------------------------
# 6. Named target included in the default ALL build
# ---------------------------------------------------------------------------
add_custom_target(pyquick ALL DEPENDS "${_PYQUICK_SO}")

# ---------------------------------------------------------------------------
# 7. Install rules
#    Both the compiled extension and the Python package glue go into
#    <prefix>/lib/pyquick/ so that after "source quick.rc" (which prepends
#    <prefix>/lib to PYTHONPATH) the package is importable as "import pyquick".
# ---------------------------------------------------------------------------
install(FILES "${_PYQUICK_SO}"
        DESTINATION lib/pyquick)
install(FILES "${CMAKE_SOURCE_DIR}/src/pyquick/__init__.py"
        DESTINATION lib/pyquick)
