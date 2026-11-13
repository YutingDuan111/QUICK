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
# 2. Verify numpy (which ships f2py) is available via the located interpreter
# ---------------------------------------------------------------------------
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import numpy; print(numpy.__version__)"
    RESULT_VARIABLE _numpy_result
    OUTPUT_VARIABLE _numpy_version
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
if(NOT _numpy_result EQUAL 0)
    message(FATAL_ERROR
        "numpy not importable by ${Python3_EXECUTABLE}. "
        "Install numpy or reconfigure without -DPYTHON=TRUE.")
endif()

# numpy < 2.0: f2py's meson backend silently drops --f90flags, so -cpp (and
# -DGPU) never reach the compiler. The #if defined(GPU) blocks in pyquick.f90
# then compile into a CPU build and the extension fails to import with
# undefined gpu_* symbols (or, for a GPU build, the GPU code is left out).
if(_numpy_version VERSION_LESS 2.0)
    message(FATAL_ERROR
        "pyquick requires numpy >= 2.0, but ${Python3_EXECUTABLE} has numpy "
        "${_numpy_version}: older f2py drops the --f90flags pyquick needs. "
        "Upgrade numpy (python -m pip install \"numpy>=2.0\") or reconfigure "
        "without -DPYTHON=TRUE.")
endif()
message(STATUS "pyquick: using numpy ${_numpy_version}")

# ---------------------------------------------------------------------------
# 3. Compute the platform-specific extension suffix
#    e.g. .cpython-313-x86_64-linux-gnu.so
# ---------------------------------------------------------------------------
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c
        "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'), end='')"
    OUTPUT_VARIABLE PYTHON_EXT_SUFFIX)

# ---------------------------------------------------------------------------
# 4. The engines.  pyquick.f90 is compiled once per QUICK library the build
#    makes, and the import a script uses picks one (src/pyquick/__init__.py):
#
#      serial   libquick             always            import pyquick
#      mpi      libquick_mpi         -DMPI=TRUE        import pyquick.mpi
#      gpu      libquick_cuda/_hip   -DCUDA/-DHIP      import pyquick.gpu
#      gpu_mpi  libquick_mpi_cuda    both              import pyquick.gpu_mpi
#
#    pyquick.f90 carries its GPU and MPI code inside #if guards (mirroring
#    main.f90), so an engine differs from the serial one only in its defines,
#    the .mod directory of its library (where the gpu_* and MPI interfaces
#    live), the library it links, and the CUDA/MPI link arguments.  GPU covers
#    both the cuda and hip backends: HIP=TRUE flips QUICK_GPU_TARGET_NAME to
#    "hip" and QUICK_GPU_PLATFORM to "HIP".
#
#    -cpp and -I<src/util> are unconditional: the guards must be *evaluated*
#    even on the CPU build (where they evaluate to nothing), and preprocessing
#    means "#include util.fh" now has to be resolved by cpp rather than by
#    gfortran's own include handling.
# ---------------------------------------------------------------------------
set(_PYQUICK_BASE_F90FLAGS "-cpp -I${CMAKE_SOURCE_DIR}/src/util")

# Extra f2py arguments (-I include dirs, -L/-l link flags) shared by the engines below.
macro(_pyquick_add_cuda_link_args)
    if(NOT HIP)
        # libquick_cuda is a shared library, so cublas/cusolver/cudart are
        # normally pulled in transitively. Name them anyway: f2py links with
        # gfortran, not the CUDA toolchain, so nothing guarantees the device
        # runtime is on the link line. FindCUDA gives full paths; f2py's meson
        # backend wants -L/-l, hence the split.
        foreach(_cudalib "${CUDA_CUDART_LIBRARY}" "${CUDA_cublas_LIBRARY}" "${CUDA_cusolver_LIBRARY}")
            if(_cudalib)
                get_filename_component(_cudalibdir "${_cudalib}" DIRECTORY)
                list(APPEND _PYQUICK_EXTRA_ARGS "-L${_cudalibdir}")
            endif()
        endforeach()
        list(REMOVE_DUPLICATES _PYQUICK_EXTRA_ARGS)
        list(APPEND _PYQUICK_EXTRA_ARGS -lcudart -lcublas -lcusolver)
    endif()
endmacro()

macro(_pyquick_add_mpi_args)
    # pyquick.f90's own MPI blocks sit behind #ifdef MPIV and call MPI directly
    # (use mpi), so the compile needs the MPI Fortran module directory and the
    # link the MPI Fortran libraries. Python owns MPI through mpi4py (see _bind
    # in _api.py), so mpi4py must be importable -- and built against this
    # same MPI library.
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import mpi4py"
        RESULT_VARIABLE _mpi4py_result
        OUTPUT_QUIET ERROR_QUIET)
    if(NOT _mpi4py_result EQUAL 0)
        message(FATAL_ERROR
            "pyquick with MPI=TRUE requires mpi4py for ${Python3_EXECUTABLE}, built "
            "against the same MPI library as QUICK. Install it from source:\n"
            "    MPICC=<path to mpicc> ${Python3_EXECUTABLE} -m pip install --no-binary=mpi4py mpi4py\n"
            "or reconfigure with -DPYTHON=FALSE.")
    endif()

    # FindMPI gives full library paths; f2py's meson backend wants -L/-l.
    # f2py turns each -l<name> into a meson variable named <name>, so a name
    # that is not a valid identifier (Open MPI 4's open-rte, open-pal) breaks
    # meson setup. Those are internal libraries libmpi itself depends on and
    # loads, so they are left out; the Fortran bindings (mpi_mpifh, ...) and
    # libmpi are what pyquick.f90 calls.
    foreach(_mpiinc ${MPI_Fortran_INCLUDE_DIRS} ${MPI_Fortran_MODULE_DIR})
        list(APPEND _PYQUICK_EXTRA_ARGS "-I${_mpiinc}")
    endforeach()
    foreach(_mpilib ${MPI_Fortran_LIBRARIES})
        get_filename_component(_mpilibdir "${_mpilib}" DIRECTORY)
        get_filename_component(_mpilibname "${_mpilib}" NAME_WE)
        string(REGEX REPLACE "^lib" "" _mpilibname "${_mpilibname}")
        if(_mpilibname MATCHES "^[A-Za-z_][A-Za-z0-9_]*$")
            list(APPEND _PYQUICK_EXTRA_ARGS "-L${_mpilibdir}" "-l${_mpilibname}")
        endif()
    endforeach()
    list(REMOVE_DUPLICATES _PYQUICK_EXTRA_ARGS)
endmacro()

# ---------------------------------------------------------------------------
# 5. One engine: build its .so via f2py and install it into
#    <prefix>/lib/pyquick/_engines/<mode>/.
#
#    Include path: build-tree .mod files for the engine's libquick target,
#    set by config_module_dirs(...) in src/CMakeLists.txt and fully populated
#    once that library has compiled (hence DEPENDS <library target>).
#
#    Link path: resolved at build time via $<TARGET_FILE_DIR:...>, so -L always
#    points at the actual library location regardless of CMAKE_INSTALL_PREFIX.
#
#    Backend: force the meson backend. f2py's legacy distutils backend is
#    removed on Python 3.12+ and is already broken on 3.9 with modern numpy
#    (numpy.distutils imports distutils.msvccompiler, which no longer exists),
#    so meson is the portable choice. Requires meson + ninja (checked above).
#
#    Each engine builds in its own directory (f2py's meson build tree lives in
#    the working directory).  MAKEFLAGS is cleared because a parallel make
#    passes its jobserver to the ninja that f2py starts, without the file
#    descriptors that go with it ("Could not initialize jobserver").
# ---------------------------------------------------------------------------
set(_PYQUICK_ENGINE_SOS "")

function(_pyquick_add_engine mode libtarget libname moddir defines use_cuda use_mpi)
    set(_PYQUICK_EXTRA_ARGS "")
    if(use_cuda)
        _pyquick_add_cuda_link_args()
    endif()
    if(use_mpi)
        _pyquick_add_mpi_args()
    endif()
    set(_workdir "${CMAKE_CURRENT_BINARY_DIR}/${mode}")
    set(_so "${_workdir}/_pyquick${PYTHON_EXT_SUFFIX}")
    file(MAKE_DIRECTORY "${_workdir}")
    add_custom_command(
        OUTPUT  "${_so}"
        COMMAND ${CMAKE_COMMAND} -E env --unset=MAKEFLAGS --unset=MFLAGS
                ${Python3_EXECUTABLE} -m numpy.f2py
                    --backend meson
                    -c "${CMAKE_SOURCE_DIR}/src/pyquick/pyquick.f90"
                    "--f90flags=${_PYQUICK_BASE_F90FLAGS}${defines}"
                    "-I${CMAKE_BINARY_DIR}/amber-modules/quick/${moddir}"
                    "-L$<TARGET_FILE_DIR:${libtarget}>"
                    "-l${libname}"
                    ${_PYQUICK_EXTRA_ARGS}
                    -m _pyquick
        DEPENDS ${libtarget} "${CMAKE_SOURCE_DIR}/src/pyquick/pyquick.f90"
        WORKING_DIRECTORY "${_workdir}"
        COMMENT "Building the ${mode} pyquick engine via f2py"
        VERBATIM)
    install(FILES "${_so}" DESTINATION lib/pyquick/_engines/${mode})
    set(_PYQUICK_ENGINE_SOS ${_PYQUICK_ENGINE_SOS} "${_so}" PARENT_SCOPE)
    message(STATUS "pyquick: ${mode} engine, bound to lib${libname}")
endfunction()

_pyquick_add_engine(serial libquick quick serial "" FALSE FALSE)
if(MPI)
    # MPI (CPU): libquick_mpi, whose MPIV code path QUICK runs across all
    # ranks of MPI_COMM_WORLD.
    _pyquick_add_engine(mpi libquick_mpi quick_mpi mpi " -DMPIV" FALSE TRUE)
endif()
if(CUDA)
    _pyquick_add_engine(gpu libquick_${QUICK_GPU_TARGET_NAME} quick_${QUICK_GPU_TARGET_NAME}
        ${QUICK_GPU_TARGET_NAME} " -DGPU -D${QUICK_GPU_PLATFORM}" TRUE FALSE)
endif()
if(CUDA AND MPI)
    # GPU + MPI: one GPU per rank -- QUICK's MPIV_GPU path, as in
    # quick.cuda.MPI. libquick_mpi_cuda is a copy of libquick_mpi with the GPU
    # sources, so MPIV holds there too and pyquick's CPU-MPI blocks apply
    # unchanged; MPIV_GPU adds the per-rank device start-up and the uploads.
    _pyquick_add_engine(gpu_mpi libquick_mpi_${QUICK_GPU_TARGET_NAME}
        quick_mpi_${QUICK_GPU_TARGET_NAME} mpi_${QUICK_GPU_TARGET_NAME}
        " -DMPIV -DMPIV_GPU -D${QUICK_GPU_PLATFORM}_MPIV" TRUE TRUE)
endif()

# ---------------------------------------------------------------------------
# 6. Named target included in the default ALL build: every engine
# ---------------------------------------------------------------------------
add_custom_target(pyquick ALL DEPENDS ${_PYQUICK_ENGINE_SOS})

# ---------------------------------------------------------------------------
# 7. Install the Python package next to the engines, in <prefix>/lib/pyquick/,
#    so that after "source quick.rc" (which prepends <prefix>/lib to
#    PYTHONPATH) "import pyquick" works.
# ---------------------------------------------------------------------------
install(FILES
            "${CMAKE_SOURCE_DIR}/src/pyquick/__init__.py"
            "${CMAKE_SOURCE_DIR}/src/pyquick/_api.py"
            "${CMAKE_SOURCE_DIR}/src/pyquick/serial.py"
            "${CMAKE_SOURCE_DIR}/src/pyquick/mpi.py"
            "${CMAKE_SOURCE_DIR}/src/pyquick/gpu.py"
            "${CMAKE_SOURCE_DIR}/src/pyquick/gpu_mpi.py"
        DESTINATION lib/pyquick)
