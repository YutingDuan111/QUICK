module pyquick

#include "util.fh"
    use iso_fortran_env, only : output_unit
    use quick_molspec_module, only : quick_molspec, natom, xyz, alloc
    use quick_calculated_module, only : quick_qm_struct
    use quick_method_module, only : quick_method
    use quick_files_module, only : iOutFile, outFileName, inFileName, isTemplate, &
                                   set_quick_files, print_quick_io_file, &
                                   write_molden, iMoldenFile
    use quick_molden_module, only : quick_molden
    use quick_api_module, only : quick_api
    use quick_constants_module, only : SYMBOL, SYMBOL_MAX, A_TO_BOHRS
    use quick_basis_module, only : nbasis, NBSuse, quick_basis, itype
#ifdef MPIV
    use quick_basis_module, only : mpi_jshell, mpi_nbasis
#endif
    use quick_cutoff_module, only : schwarzoff
    use quick_eri_cshell_module, only : getEriPrecomputables
    use quick_sad_guess_module, only : getSadGuess
    use quick_oei_module, only : kineticO
    use quick_gridpoints_module, only : quick_dft_grid, deform_dft_grid
    use quick_grad_cshell_module, only : cshell_gradient
    use quick_grad_oshell_module, only : oshell_gradient
    use quick_optimizer_module, only : lopt
    use quick_exception_module, only : get_exception_message
    use quick_molsurface_module, only : generate_MKS_surfaces
    use quick_oeproperties_module, only : compute_esp, compute_ESP_charge, esp_charges
    use quick_mpi_module, only : master, quick_comm_rank, quick_comm_size, libMPIMode

    implicit none

    private
    public :: set_calc, set_basis, set_method, clear_methods, read_geom, &
              had_error, error_message, print_input, input_string, &
              set_point_charges, clear_point_charges, &
              job_run, job_destroy, job_set_output, &
              job_open, job_step, job_close, &
              job_active, &
              job_total_energy, job_e_core, job_e_electronic, &
              job_e_1e, job_e_kinetic, job_e_xc, job_e_disp, job_e_charge, &
              job_dipole, job_has_dipole, &
              job_has_dispersion, job_has_extcharge, job_has_gradient, &
              job_has_mulliken, job_has_lowdin, job_has_mo_energies, job_has_density_matrix, &
              job_get_mulliken, job_get_lowdin, job_get_mo_energies, job_get_density_matrix, &
              job_get_gradients, job_get_geometry, job_exception_message, &
              job_has_esp, job_get_esp_charges, &
              job_has_ptchg_gradient, job_get_point_charge_gradient, &
              job_nbasis, job_nmo, job_nelec_alpha, job_nelec_beta, &
              job_get_mo_coefficients, job_get_beta_mo_coefficients, &
              job_get_overlap, job_get_ao_info, &
              job_has_trajectory, job_trajectory_nsteps, job_trajectory_natoms, &
              job_get_trajectory_energies, job_get_trajectory_coordinates, &
              job_has_beta, job_get_beta_mo_energies, job_get_beta_density_matrix, &
              job_has_optimized, job_opt_converged, job_get_optimized_geometry, &
              set_mpi, get_build_info

    integer, parameter :: KEYWORD_LEN = 300
    integer, parameter :: INPUT_LEN   = 10000

    character(len=8) :: calc_keyword = ''
    character(len=:), allocatable :: basis_token
    integer, allocatable          :: geom_atnum(:)      ! atomic numbers, length geom_natom
    double precision, allocatable :: geom_coords(:,:)   ! Angstrom coordinates, shape (3, geom_natom)

    logical :: has_calc  = .false.
    logical :: has_basis = .false.
    logical :: has_geom  = .false.

    logical            :: had_error     = .false.
    character(len=512) :: error_message = ''

    character(len=INPUT_LEN) :: input_string = ''

    ! atom count stored by read_geom so job_run knows natom before alloc
    integer :: geom_natom = 0

    ! external point charges for the following runs, set from Python:
    ! pc_data(1:3,i) = Angstrom coordinates, pc_data(4,i) = charge
    integer :: pc_n = 0
    double precision, allocatable :: pc_data(:,:)

    ! output file stem (default 'pyquick_job')
    character(len=80) :: output_stem = 'pyquick_job'

    ! job state
    logical :: job_active = .false.

    ! a persistent job (job_open .. repeated job_step .. job_close) is set up:
    ! the molecule and basis are allocated once and the density is reused across
    ! steps (only coordinates change), until job_close frees everything.
    logical :: job_open_flag = .false.
    integer :: job_nsteps = 0        ! steps run on the open persistent job

    ! .true. while QUICK itself is running (from initialize1 to the end of a
    ! job_run / job_open / job_step). Under MPI, a failure in that window may
    ! hit one rank only while the others wait in a collective call, so fail()
    ! aborts every rank instead of returning (see fail).
    logical :: engine_running = .false.

    ! scalar results (always available after a successful job_run)
    ! job_e_core is QUICK's ECore: the core-core NUCLEAR REPULSION energy
    double precision :: job_total_energy   = 0.0d0
    double precision :: job_e_core         = 0.0d0
    double precision :: job_e_electronic   = 0.0d0
    double precision :: job_e_1e           = 0.0d0
    double precision :: job_e_kinetic      = 0.0d0
    double precision :: job_e_xc           = 0.0d0
    double precision :: job_e_disp         = 0.0d0
    double precision :: job_e_charge       = 0.0d0   ! ECharge; never part of ETot

    ! dipole moment vector (Debye), available when the DIPOLE routine ran
    double precision :: job_dipole(3) = 0.0d0
    logical :: job_has_dipole = .false.

    ! whether dispersion (D* keyword) / EXTCHARGES were active for the last run
    logical :: job_has_dispersion = .false.
    logical :: job_has_extcharge  = .false.

    ! whether a nuclear gradient was computed for the last run
    logical :: job_has_gradient = .false.
    logical :: job_has_ptchg_gradient = .false.

    ! sizes for the MO, overlap and AO getters, set by harvest_results.
    ! job_nelec_alpha is the number of doubly occupied MOs on a restricted run.
    integer :: job_nbasis = 0
    integer :: job_nmo = 0
    integer :: job_nelec_alpha = 0
    integer :: job_nelec_beta = 0

    ! optimization trajectory, copied out of QUICK's Molden storage (see
    ! trajectory_begin). DEFAULT_OPT_CYCLES caps a geo_opt with no max_cycles: the
    ! storage needs a known size, and QUICK's optimizers do not check it.
    integer, parameter :: DEFAULT_OPT_CYCLES = 2000
    logical :: job_has_trajectory = .false.
    integer :: job_trajectory_nsteps = 0
    integer :: job_trajectory_natoms = 0
    double precision, allocatable :: traj_coords(:,:,:)   ! (3, natom, nsteps), Angstrom
    double precision, allocatable :: traj_energies(:)     ! (nsteps), Hartree

    ! geometry optimization: whether the last run was an OPT job, and whether it
    ! reached convergence (as opposed to stopping at the cycle limit)
    logical :: job_has_optimized  = .false.
    logical :: job_opt_converged  = .false.

    ! availability flags for array results
    logical :: job_has_mulliken      = .false.
    logical :: job_has_esp           = .false.
    logical :: job_has_lowdin        = .false.
    logical :: job_has_mo_energies   = .false.
    logical :: job_has_density_matrix = .false.
    logical :: job_has_beta           = .false.   ! unrestricted run: Eb / denseb hold beta

    type :: method_entry
        character(len=:), allocatable :: keyword
        character(len=:), allocatable :: arg
    end type method_entry

    type(method_entry), allocatable :: method_list(:)

contains

    ! -----------------------------------------------------------------------
    ! Backward-compatible input-assembly API
    ! -----------------------------------------------------------------------

    subroutine set_calc(keyword)
        character(len=*), intent(in) :: keyword
        character(len=:), allocatable :: normalized

        normalized = uppercase(trim(keyword))

        select case (trim(normalized))
        case ('HF', 'UHF', 'DFT', 'UDFT')
            calc_keyword = normalized
            has_calc = .true.
            call rebuild_input()
        case default
            call fail('set_calc: keyword must be HF, UHF, DFT or UDFT')
        end select
    end subroutine set_calc

    subroutine set_basis(basis_name)
        character(len=*), intent(in) :: basis_name

        if (len_trim(basis_name) == 0) then
            call fail('set_basis: basis name must be non-empty')
            return
        end if

        basis_token = 'BASIS=' // trim(adjustl(basis_name))
        has_basis = .true.
        call rebuild_input()
    end subroutine set_basis

    subroutine set_method(keyword, arg)
        character(len=*), intent(in) :: keyword
        character(len=*), intent(in), optional :: arg
        character(len=:), allocatable :: uname
        integer :: i

        if (len_trim(keyword) == 0) then
            call fail('set_method: keyword name must be non-empty')
            return
        end if

        uname = uppercase(trim(adjustl(keyword)))

        if (allocated(method_list)) then
            do i = 1, size(method_list)
                if (trim(method_list(i)%keyword) == trim(uname)) then
                    if (present(arg) .and. len_trim(arg) > 0) then
                        method_list(i)%arg = trim(adjustl(arg))
                    else
                        method_list(i)%arg = ''
                    end if
                    call rebuild_input()
                    return
                end if
            end do
        end if

        call append_method(uname, arg)
        call rebuild_input()
    end subroutine set_method

    subroutine read_geom(input)
        character(len=*), intent(in) :: input
        character(len=:), allocatable :: line
        integer :: start, len_input, nl, atom_count, ios, k, z
        character(len=4) :: sym
        double precision :: cx, cy, cz

        ! discard any previous geometry so a second call replaces the first
        if (allocated(geom_atnum))  deallocate(geom_atnum)
        if (allocated(geom_coords)) deallocate(geom_coords)
        geom_natom = 0
        has_geom   = .false.

        ! --- first pass: count non-empty lines to know how many atoms ---
        atom_count = 0
        start      = 1
        len_input  = len(input)

        do while (start <= len_input)
            nl = index(input(start:), new_line('a'))
            if (nl == 0) then
                if (len_trim(input(start:)) > 0) atom_count = atom_count + 1
                exit
            else
                if (len_trim(input(start:start+nl-2)) > 0) atom_count = atom_count + 1
                start = start + nl
            end if
        end do

        if (atom_count == 0) then
            call fail('read_geom: geometry must contain at least one atom')
            return
        end if

        allocate(geom_atnum(atom_count))
        allocate(geom_coords(3, atom_count))

        ! --- second pass: parse and validate each line ---
        atom_count = 0
        start      = 1

        do while (start <= len_input)
            nl = index(input(start:), new_line('a'))
            if (nl == 0) then
                line = trim(adjustl(input(start:)))
            else
                line = trim(adjustl(input(start:start+nl-2)))
                start = start + nl
            end if

            if (len_trim(line) == 0) then
                if (nl == 0) exit
                cycle
            end if

            ! parse symbol and three coordinates; accepts decimal and scientific notation
            sym = ''
            read(line, *, iostat=ios) sym, cx, cy, cz
            if (ios /= 0) then
                call fail('read_geom: cannot parse line (expected: SYMBOL X Y Z): ' // &
                          trim(line))
                return
            end if

            ! validate symbol against the QUICK element table
            z = 0
            do k = 1, SYMBOL_MAX
                if (trim(uppercase(sym)) == trim(uppercase(SYMBOL(k)))) then
                    z = k
                    exit
                end if
            end do
            if (z == 0) then
                call fail('read_geom: unknown element symbol "' // trim(sym) // &
                          '" in line: ' // trim(line))
                return
            end if

            atom_count = atom_count + 1
            geom_atnum(atom_count)     = z
            geom_coords(1, atom_count) = cx
            geom_coords(2, atom_count) = cy
            geom_coords(3, atom_count) = cz

            if (nl == 0) exit
        end do

        geom_natom = atom_count
        has_geom   = .true.
        call rebuild_input()

    end subroutine read_geom

    subroutine print_input()
        write(output_unit, '(A)') trim(input_string)
    end subroutine print_input

    subroutine rebuild_input()
        character(len=:), allocatable :: text
        character(len=64) :: atom_line
        integer :: i

        if (has_calc) then
            text = trim(calc_keyword)
        else
            text = ''
        end if

        if (allocated(method_list)) then
            do i = 1, size(method_list)
                if (len_trim(text) > 0) then
                    if (len_trim(method_list(i)%arg) > 0) then
                        text = trim(text) // ' ' // trim(method_list(i)%keyword) &
                               // '=' // trim(method_list(i)%arg)
                    else
                        text = trim(text) // ' ' // trim(method_list(i)%keyword)
                    end if
                else
                    if (len_trim(method_list(i)%arg) > 0) then
                        text = trim(method_list(i)%keyword) // '=' // trim(method_list(i)%arg)
                    else
                        text = trim(method_list(i)%keyword)
                    end if
                end if
            end do
        end if

        if (has_basis) then
            if (len_trim(text) > 0) then
                text = trim(text) // ' ' // trim(basis_token)
            else
                text = trim(basis_token)
            end if
        end if

        if (has_geom) then
            do i = 1, geom_natom
                write(atom_line, '(A2, 3(1X, F12.6))') &
                    trim(SYMBOL(geom_atnum(i))), &
                    geom_coords(1,i), geom_coords(2,i), geom_coords(3,i)
                text = trim(text) // new_line('a') // trim(atom_line)
            end do
        end if

        input_string = ''
        if (len_trim(text) > 0) then
            input_string(1:min(len_trim(text), INPUT_LEN)) = &
                text(1:min(len_trim(text), INPUT_LEN))
        end if
    end subroutine rebuild_input

    ! -----------------------------------------------------------------------
    ! Job execution API
    ! -----------------------------------------------------------------------

    subroutine job_set_output(stem)
        character(len=*), intent(in) :: stem
        if (len_trim(stem) == 0) then
            call fail('job_set_output: stem must be non-empty')
            return
        end if
        output_stem = trim(stem)
    end subroutine job_set_output

    subroutine job_run(jobtype, do_log, max_cycles)
        ! jobtype: 0 = single-point energy
        !          1 = energy + nuclear gradient
        !          2 = geometry optimization
        ! do_log:  1 = write QUICK's log to <output_stem>.out (default behaviour),
        !          0 = send it to the platform null device, leaving no file.
        ! max_cycles: for jobtype 2 only -- <= 0 uses DEFAULT_OPT_CYCLES; > 0 emits
        !          OPTIMIZE=<n> to cap the number of optimization cycles.
#if defined(GPU) || defined(MPIV_GPU)
        ! GPU builds only: the basis arrays the uploads need. Compiled out on the
        ! CPU build (needs the build to preprocess this file: -cpp, and -DGPU for
        ! GPU builds).
        use quick_basis_module, only : aexp, cutprim, dcoeff, jbasis, jshell, maxcontract, &
                                       ncontract, nprim, nshell, Ycutoff
        use quick_method_module, only : upload
#endif
        integer, intent(in) :: jobtype
        integer, intent(in) :: do_log
        integer, intent(in) :: max_cycles
        ! External free subroutines in libquick.so
        external :: initialize1, read_Job_and_Atom, getMol, getEnergy, dipole
        external :: finalize, dl_find

        character(len=:), allocatable :: keyword_line
        character(len=10) :: kwlen_str
        character(len=12) :: cyc_str
        character(len=1) :: open_mode
        integer :: ierr, i, j, k
        integer :: natm_type
        integer :: atm_type_id(geom_natom)
        logical :: new_type

        ierr = 0
        job_has_trajectory = .false.

        ! --- decide the log open mode before finalize clears job_active ---
        ! first run replaces the file; re-runs append to it
        if (job_active) then
            open_mode = 'A'
        else
            open_mode = 'R'
        end if

        ! --- if a previous run is still active, finalize it before re-running ---
        ! This deallocates all basis/MO/density arrays sized for the previous run
        ! so they can be reallocated at the correct dimensions for the new run.
        if (job_active) then
            call finalize(iOutFile, ierr, 1)
            if (ierr /= 0) then
                call fail('job_run: finalize of previous run failed')
                return
            end if
            job_active = .false.
        end if

        ! --- validate prerequisites ---
        if (.not. has_calc) then
            call fail('job_run: call set_calc before run')
            return
        end if
        if (.not. has_basis) then
            call fail('job_run: call set_basis before run')
            return
        end if
        if (.not. has_geom) then
            call fail('job_run: call read_geom before run')
            return
        end if

        ! --- build keyword line ---
        keyword_line = build_keyword_line()

        ! for a gradient job, add the GRADIENT keyword so read_Job_and_Atom sets
        ! quick_method%grad before getMol allocates quick_qm_struct%gradient
        if (jobtype == 1) keyword_line = trim(keyword_line) // ' GRADIENT'

        ! for an optimization, OPTIMIZE sets quick_method%opt (and %grad). The cycle
        ! cap is always explicit: it sizes the trajectory storage (a bare OPTIMIZE
        ! would leave iopt=0). The optimizer is QUICK's default (DL-Find) unless the
        ! user added the LOPT keyword (Cartesian).
        if (jobtype == 2) then
            if (max_cycles > 0) then
                write(cyc_str, '(I0)') max_cycles
            else
                write(cyc_str, '(I0)') DEFAULT_OPT_CYCLES
            end if
            keyword_line = trim(keyword_line) // ' OPTIMIZE=' // trim(cyc_str)
        end if

        ! --- guard against silent Fortran truncation on assignment to quick_api%Keywd ---
        if (len_trim(keyword_line) > KEYWORD_LEN) then
            write(kwlen_str, '(I0)') KEYWORD_LEN
            call fail('job_run: keyword line exceeds maximum length of ' // &
                      trim(kwlen_str) // ' characters')
            return
        end if

        ! --- configure quick_api for keyword injection ---
        quick_api%apiMode  = .true.
        quick_api%hasKeywd = .true.
        quick_api%Keywd    = trim(keyword_line)

        ! --- configure file names; isTemplate suppresses coord read in getMol ---
        inFileName = trim(output_stem) // '.in'
        isTemplate = .true.

        ! --- QUICK call chain (follows main.f90) ---
        engine_running = .true.
        call initialize1(ierr)
        if (ierr /= 0) then
            call fail('job_run: initialize1 failed')
            return
        end if
        call reset_engine_state()

        ! --- QUICK's diagnostic log: rank 0 only, as in main.f90. MPI worker
        ! ranks get the null device, so a stray write never creates a file ---
        if (master) then
            call open_log('job_run', do_log, open_mode)
        else
            call open_null_log('job_run')
        end if
        if (had_error) return

#if defined(GPU)
        ! --- GPU: create context and pick a device (mirrors main.f90) ---
        call gpu_new(ierr)
        call gpu_init_device(ierr)
        call gpu_write_info(iOutFile, ierr)
#elif defined(MPIV_GPU)
        ! --- multi-GPU: one device per rank (main.f90's MPIV_GPU branch) ---
        call start_mgpu('job_run')
        if (had_error) return
#endif

        ! reads keyword from quick_api%Keywd; skips coordinate read (apiMode)
        call read_Job_and_Atom(ierr)
        if (ierr /= 0) then
            call fail('job_run: read_Job_and_Atom failed')
            return
        end if

        ! set natom (module-level target) BEFORE alloc uses it
        natom = geom_natom
        call prepare_point_charges()

        call alloc(quick_molspec, quick_method%readxyz, ierr)
        if (ierr /= 0) then
            call fail('job_run: alloc(quick_molspec) failed')
            return
        end if
        call inject_point_charges()

        ! --- inject geometry into QUICK module-level state ---

        ! build atom type list (deduplicate by atomic number)
        natm_type = 0
        atm_type_id = 0
        do i = 1, geom_natom
            new_type = .true.
            do k = 1, natm_type
                if (atm_type_id(k) == geom_atnum(i)) then
                    new_type = .false.
                    exit
                end if
            end do
            if (new_type) then
                natm_type = natm_type + 1
                atm_type_id(natm_type) = geom_atnum(i)
            end if
        end do

        quick_molspec%iAtomType = natm_type
        quick_molspec%atom_type_sym = '  '   ! drop entries left by the previous molecule
        do i = 1, natm_type
            quick_molspec%atom_type_sym(i) = SYMBOL(atm_type_id(i))
        end do

        ! inject atomic numbers and coordinates (convert Angstrom -> Bohr)
        do i = 1, geom_natom
            quick_molspec%iattype(i) = geom_atnum(i)
            do j = 1, 3
                xyz(j, i) = geom_coords(j, i) * A_TO_BOHRS
            end do
        end do
        quick_molspec%xyz => xyz

        ! --- initial guess ---
        if (quick_method%SAD) then
            call getSadGuess(ierr)
            if (ierr /= 0) then
                call fail('job_run: getSadGuess failed')
                return
            end if
        end if

        ! --- build molecular orbital / basis information ---
        call getMol(ierr)
        if (ierr /= 0) then
            call fail('job_run: getMol failed')
            return
        end if

#if defined(GPU) || defined(MPIV_GPU)
        ! --- GPU: allocate scratch, upload method + molecule/coords ---
        call gpu_allocate_scratch(quick_method%grad .or. quick_method%opt)
        call upload(quick_method, ierr)
        if (.not. quick_method%opt) then
            call gpu_setup(natom, nbasis, quick_molspec%nElec, quick_molspec%imult, &
                           quick_molspec%molchg, quick_molspec%iAtomType)
            call gpu_upload_xyz(xyz)
            call gpu_upload_atom_and_chg(quick_molspec%iattype, quick_molspec%chg)
        end if
#endif

        ! --- ERI precomputables and cutoff screening ---
        call getEriPrecomputables()
        call schwarzoff()

#if defined(GPU) || defined(MPIV_GPU)
        ! --- GPU: upload basis + Schwarz cutoffs (needs the arrays just built) ---
        if (.not. quick_method%opt) then
            call gpu_upload_basis(nshell, nprim, jshell, jbasis, maxcontract, &
                ncontract, itype, aexp, dcoeff, &
                quick_basis%first_basis_function, quick_basis%last_basis_function, &
                quick_basis%first_shell_basis_function, quick_basis%last_shell_basis_function, &
                quick_basis%ncenter, quick_basis%kstart, quick_basis%katom, &
                quick_basis%ktype, quick_basis%kprim, quick_basis%kshell, quick_basis%Ksumtype, &
                quick_basis%Qnumber, quick_basis%Qstart, quick_basis%Qfinal, &
                quick_basis%Qsbasis, quick_basis%Qfbasis, &
                quick_basis%gccoeff, quick_basis%cons, quick_basis%gcexpo, quick_basis%KLMN)
            call gpu_upload_cutoff_matrix(Ycutoff, cutPrim)
            call gpu_upload_oei(quick_molspec%nExtAtom, quick_molspec%extxyz, &
                                quick_molspec%extchg, ierr)
        end if
#endif

        ! QUICK sets Exc only on the DFT path; reset it so an HF run after a DFT
        ! run in the same process does not report the previous run's value
        quick_qm_struct%Exc = 0.0d0

        ! --- energy, gradient, or optimization ---
        ! quick_method%opt/%grad come from the OPTIMIZE/GRADIENT keywords appended
        ! above. Each branch runs its own SCF, so getEnergy is only called for a
        ! plain energy job (this mirrors quick_api_module / main.f90).
        if (quick_method%opt) then
            ! DL-Find is QUICK's default optimizer; the LOPT keyword selects the
            ! Cartesian optimizer instead. DL-Find crashes on molecules with fewer
            ! than 3 atoms, so refuse that combination with a clear message rather
            ! than segfaulting (the Python layer guards this too).
            if (quick_method%usedlfind) then
                if (geom_natom < 3) then
                    call fail('geo_opt: DL-Find does not support molecules with ' // &
                              'fewer than 3 atoms; use the Cartesian optimizer ' // &
                              '(add the LOPT keyword)')
                    return
                end if
                call trajectory_begin()
                call dl_find(ierr, master)
            else
                call trajectory_begin()
                call lopt(ierr)
            end if
            call trajectory_end()
            if (ierr /= 0) then
                call fail('job_run: geometry optimization failed')
                return
            end if
        else if (quick_method%grad) then
            if (quick_method%unrst) then
                call oshell_gradient(ierr)
            else
                call cshell_gradient(ierr)
            end if
            if (ierr /= 0) then
                call fail('job_run: gradient calculation failed')
                return
            end if
        else
            call getEnergy(.false., ierr)
            if (ierr /= 0) then
                call fail('job_run: getEnergy failed')
                return
            end if
        end if

        ! ESP charges after an energy or gradient job, as main.f90 does
        if (.not. quick_method%opt) call run_esp_charges()

        ! --- post-SCF charges/dipole + harvest all results ---
        call harvest_results(quick_method%grad)

#if defined(GPU) || defined(MPIV_GPU)
        ! --- GPU: free device scratch and context ---
        call gpu_deallocate_scratch(quick_method%grad .or. quick_method%opt)
#if defined(MPIV_GPU)
        call delete_mgpu_setup(ierr)
#endif
        call gpu_delete(ierr)
#endif

        job_active = .true.
        engine_running = .false.

    end subroutine job_run

    ! -----------------------------------------------------------------------
    ! Persistent job: set up once (job_open), run many geometries of the same
    ! molecule (job_step, reusing the density), then free everything (job_close).
    ! -----------------------------------------------------------------------

    subroutine job_open(do_log)
        ! One-time setup for a fixed molecule. Prerequisites (as for job_run):
        ! set_calc, set_basis and read_geom (the initial geometry) must be done.
        ! do_log: 1 = write <stem>.out, 0 = null device.
#if defined(GPU) || defined(MPIV_GPU)
        use quick_method_module, only : upload
#endif
        integer, intent(in) :: do_log
        external :: initialize1, read_Job_and_Atom, getMol

        character(len=:), allocatable :: keyword_line
        character(len=10) :: kwlen_str
        integer :: ierr, i, j, k, natm_type
        integer :: atm_type_id(geom_natom)
        logical :: new_type

        ierr = 0
        job_has_trajectory = .false.
        if (job_open_flag) then
            call fail('job_open: a job is already open; call job_close first')
            return
        end if
        if (.not. has_calc)  then; call fail('job_open: call set_calc before'); return; end if
        if (.not. has_basis) then; call fail('job_open: call set_basis before'); return; end if
        if (.not. has_geom)  then; call fail('job_open: call read_geom before'); return; end if

        ! Append GRADIENT so getMol sizes quick_qm_struct%gradient; individual
        ! steps then opt into a gradient without re-running getMol.
        keyword_line = trim(build_keyword_line()) // ' GRADIENT'
        if (len_trim(keyword_line) > KEYWORD_LEN) then
            write(kwlen_str, '(I0)') KEYWORD_LEN
            call fail('job_open: keyword line exceeds ' // trim(kwlen_str) // ' characters')
            return
        end if

        quick_api%apiMode  = .true.
        quick_api%hasKeywd = .true.
        quick_api%Keywd    = trim(keyword_line)
        inFileName = trim(output_stem) // '.in'
        isTemplate = .true.

        engine_running = .true.
        call initialize1(ierr)
        if (ierr /= 0) then; call fail('job_open: initialize1 failed'); return; end if
        call reset_engine_state()

        ! log on rank 0 only; MPI workers write to the null device (see job_run)
        if (master) then
            call open_log('job_open', do_log, 'R')
        else
            call open_null_log('job_open')
        end if
        if (had_error) return

#if defined(GPU)
        ! --- GPU: create context and pick a device (mirrors job_run) ---
        call gpu_new(ierr)
        call gpu_init_device(ierr)
        call gpu_write_info(iOutFile, ierr)
#elif defined(MPIV_GPU)
        ! --- multi-GPU: one device per rank (main.f90's MPIV_GPU branch) ---
        call start_mgpu('job_open')
        if (had_error) return
#endif

        call read_Job_and_Atom(ierr)
        if (ierr /= 0) then; call fail('job_open: read_Job_and_Atom failed'); return; end if

        natom = geom_natom
        call prepare_point_charges()
        call alloc(quick_molspec, quick_method%readxyz, ierr)
        if (ierr /= 0) then; call fail('job_open: alloc(quick_molspec) failed'); return; end if
        call inject_point_charges()

        ! atom types (deduplicate by atomic number) + initial coordinates
        natm_type = 0
        atm_type_id = 0
        do i = 1, geom_natom
            new_type = .true.
            do k = 1, natm_type
                if (atm_type_id(k) == geom_atnum(i)) then; new_type = .false.; exit; end if
            end do
            if (new_type) then; natm_type = natm_type + 1; atm_type_id(natm_type) = geom_atnum(i); end if
        end do
        quick_molspec%iAtomType = natm_type
        quick_molspec%atom_type_sym = '  '   ! drop entries left by the previous molecule
        do i = 1, natm_type
            quick_molspec%atom_type_sym(i) = SYMBOL(atm_type_id(i))
        end do
        do i = 1, geom_natom
            quick_molspec%iattype(i) = geom_atnum(i)
            do j = 1, 3
                xyz(j, i) = geom_coords(j, i) * A_TO_BOHRS
            end do
        end do
        quick_molspec%xyz => xyz

        if (quick_method%SAD) then
            call getSadGuess(ierr)
            if (ierr /= 0) then; call fail('job_open: getSadGuess failed'); return; end if
        end if
        call getMol(ierr)
        if (ierr /= 0) then; call fail('job_open: getMol failed'); return; end if

#if defined(GPU) || defined(MPIV_GPU)
        ! --- GPU: allocate persistent scratch (always gradient-capable -- job_step
        ! toggles want_gradient per call, so this can't be sized in advance, unlike
        ! job_run's one-shot `quick_method%grad .or. quick_method%opt`) and upload
        ! the method settings once for the life of this job. ---
        call gpu_allocate_scratch(.true.)
        call upload(quick_method, ierr)
#endif

        job_open_flag = .true.
        job_active    = .true.
        job_nsteps    = 0
        engine_running = .false.
    end subroutine job_open

    subroutine job_step(want_gradient)
        ! Run one geometry of the open job. read_geom must have set the new
        ! coordinates (same atoms). want_gradient: 1 = also compute the gradient.
        ! getMol is NOT called, so quick_qm_struct%dense from the previous step
        ! survives and seeds this SCF.
#if defined(GPU) || defined(MPIV_GPU)
        use quick_basis_module, only : aexp, cutprim, dcoeff, jbasis, jshell, maxcontract, &
                                       ncontract, nprim, nshell, Ycutoff
        use quick_method_module, only : upload
#endif
        integer, intent(in) :: want_gradient
        integer :: ierr, i, j

        ierr = 0
        if (.not. job_open_flag) then
            call fail('job_step: no job is open; call job_open first')
            return
        end if
        if (geom_natom /= natom) then
            call fail('job_step: geometry has a different number of atoms than the open job')
            return
        end if
        engine_running = .true.

        ! update coordinates only (atoms are fixed for the job)
        do i = 1, natom
            do j = 1, 3
                xyz(j, i) = geom_coords(j, i) * A_TO_BOHRS
            end do
        end do

        ! DFT: free the previous step's grid before getEnergy builds one for the
        ! new geometry, as quick_api_module's run_quick does between MD steps.
        ! Otherwise the grid is rebuilt over the old arrays; once its size changes
        ! with the geometry, the GPU build corrupts the heap (found on job close).
        if (job_nsteps > 0 .and. quick_method%DFT) call deform_dft_grid(quick_dft_grid)

        call getEriPrecomputables()
        call schwarzoff()

#if defined(GPU) || defined(MPIV_GPU)
        ! --- GPU: re-upload geometry + basis for this step. gpu_upload_basis and
        ! gpu_upload_oei also bake in geometry-derived quantities (interatomic
        ! distances, Gaussian product centers), so this must repeat every step,
        ! not just once in job_open -- see quick_api_module's run_quick, which
        ! does the same full re-upload unconditionally on every call. ---
        call gpu_setup(natom, nbasis, quick_molspec%nElec, quick_molspec%imult, &
                       quick_molspec%molchg, quick_molspec%iAtomType)
        call gpu_upload_xyz(xyz)
        call gpu_upload_atom_and_chg(quick_molspec%iattype, quick_molspec%chg)
        call gpu_upload_basis(nshell, nprim, jshell, jbasis, maxcontract, &
            ncontract, itype, aexp, dcoeff, &
            quick_basis%first_basis_function, quick_basis%last_basis_function, &
            quick_basis%first_shell_basis_function, quick_basis%last_shell_basis_function, &
            quick_basis%ncenter, quick_basis%kstart, quick_basis%katom, &
            quick_basis%ktype, quick_basis%kprim, quick_basis%kshell, quick_basis%Ksumtype, &
            quick_basis%Qnumber, quick_basis%Qstart, quick_basis%Qfinal, &
            quick_basis%Qsbasis, quick_basis%Qfbasis, &
            quick_basis%gccoeff, quick_basis%cons, quick_basis%gcexpo, quick_basis%KLMN)
        call gpu_upload_cutoff_matrix(Ycutoff, cutPrim)
        call gpu_upload_oei(quick_molspec%nExtAtom, quick_molspec%extxyz, &
                            quick_molspec%extchg, ierr)
#endif

        quick_qm_struct%Exc = 0.0d0   ! see job_run: QUICK sets Exc only for DFT

        if (want_gradient /= 0) then
            if (quick_method%unrst) then
                call oshell_gradient(ierr)
            else
                call cshell_gradient(ierr)
            end if
            if (ierr /= 0) then; call fail('job_step: gradient calculation failed'); return; end if
        else
            call getEnergy(.false., ierr)
            if (ierr /= 0) then; call fail('job_step: getEnergy failed'); return; end if
        end if
        call run_esp_charges()

#if defined(GPU) || defined(MPIV_GPU)
        ! --- GPU: intentionally NOT calling gpu_cleanup() here. ---
        ! quick_api_module's run_quick calls it at this point to free the step's
        ! geometry-dependent device arrays, but here it segfaults on its first
        ! call: gdb shows it deleting gpu->gpu_cutoff->cutMatrix, whose object is
        ! already corrupted. That object is damaged in QUICK's normal GPU SCF path
        ! (freeing the previous iteration's cutMatrix in gpu_upload_cutoff_ breaks
        ! even the plain quick.cuda binary), so the fault is a host-memory
        ! corruption in the GPU code, not in pyquick. Until it is found, each step
        ! leaks its device arrays (~570 KiB/step for benzene/6-31G*), reclaimed at
        ! job_close/job_destroy by gpu_delete's cudaDeviceReset.
#endif

        call harvest_results(want_gradient /= 0)
        job_nsteps = job_nsteps + 1
        engine_running = .false.
    end subroutine job_step

    subroutine job_close()
        external :: finalize
        integer :: ierr
        ierr = 0
        if (job_active) then
            ! Only tear down the GPU context if THIS call opened one (job_open_flag).
            ! job_active is also left .true. by the one-shot job_run, which already
            ! tore its own GPU context down (gpu_delete) before returning -- calling
            ! job_gpu_teardown() again there would double-free the (still-dangling,
            ! gpu_delete_ never nulls the global `gpu` pointer) device context.
            if (job_open_flag) call job_gpu_teardown()
            call finalize(iOutFile, ierr, 1)
        end if
        job_active    = .false.
        job_open_flag = .false.
    end subroutine job_close

    subroutine job_gpu_teardown()
        ! Shared GPU-context teardown for job_close and job_destroy. Either one
        ! can be the last call on an open persistent job -- job_destroy also
        ! backstops __init__.py's PyQuick.__del__ -- so both must free the
        ! device context, not just job_close. Mirrors quick_api_module's
        ! delete_quick_job, including the libxc GPU cleanup that job_run's
        ! one-shot teardown skips (harmless there for non-DFT jobs, but a
        ! persistent job can run DFT).
#if defined(GPU) || defined(MPIV_GPU)
        use quick_method_module, only : delete
        integer :: ierr
        ierr = 0
        call delete(quick_method, ierr)
        call gpu_deallocate_scratch(.true.)
#if defined(MPIV_GPU)
        call delete_mgpu_setup(ierr)
#endif
        call gpu_delete(ierr)
#endif
    end subroutine job_gpu_teardown

    subroutine harvest_results(did_gradient)
        ! Copy energies/properties out of QUICK's module state after a run.
#ifdef MPIV
        use mpi
        use quick_mpi_module, only : quick_comm
        integer :: ierr_mpi
#endif
        external :: dipole
        logical, intent(in) :: did_gradient

        if (quick_method%dipole .and. master) call dipole

        job_total_energy   = quick_qm_struct%ETot
        job_e_core         = quick_qm_struct%ECore
        job_e_electronic   = quick_qm_struct%EEl
        job_e_1e           = quick_qm_struct%E1e
#ifdef MPIV
        ! under MPI each rank builds part of the one-electron matrix, so its E1e is
        ! partial. QUICK reduces Eel but not E1e; rank 0 needs the sum.
        call MPI_Reduce(quick_qm_struct%E1e, job_e_1e, 1, MPI_DOUBLE_PRECISION, MPI_SUM, &
                        0, quick_comm, ierr_mpi)
#endif
        job_e_xc           = quick_qm_struct%Exc
        if (master) call compute_kinetic_energy()
        job_e_disp         = quick_qm_struct%Edisp
        job_e_charge       = quick_qm_struct%ECharge
        job_has_dispersion = quick_method%edisp
        job_has_extcharge  = quick_method%extcharges

        job_has_dipole = quick_method%dipole
        if (quick_method%dipole) job_dipole = quick_qm_struct%dipole

        job_has_mulliken       = quick_method%dipole
        job_has_esp            = quick_method%esp_charge .and. allocated(esp_charges)
        job_has_lowdin         = quick_method%dipole
        job_has_mo_energies    = allocated(quick_qm_struct%E)
        job_has_density_matrix = allocated(quick_qm_struct%dense)
        ! gate on unrst, not on allocation: QUICK allocates denseb for closed-shell
        ! DFT too, and arrays can outlive the run that filled them
        job_has_beta           = quick_method%unrst .and. allocated(quick_qm_struct%Eb) &
                                 .and. allocated(quick_qm_struct%denseb)
        job_has_gradient       = did_gradient .and. allocated(quick_qm_struct%gradient)
        job_nbasis = nbasis
        job_nmo    = NBSuse
        if (quick_method%unrst) then
            job_nelec_alpha = quick_molspec%nElec
            job_nelec_beta  = quick_molspec%nElecb
        else
            job_nelec_alpha = quick_molspec%nElec / 2
            job_nelec_beta  = 0
        end if
        job_has_ptchg_gradient = did_gradient .and. quick_molspec%nextatom > 0 .and. &
                                 allocated(quick_qm_struct%ptchg_gradient)

        job_has_optimized = quick_method%opt
        job_opt_converged = quick_qm_struct%opt_converged
    end subroutine harvest_results

    subroutine job_destroy()
        external :: finalize
        integer :: ierr
        ierr = 0
        if (job_active) then
            ! See job_close: only tear down the GPU context if a persistent job
            ! (job_open) actually left one open.
            if (job_open_flag) call job_gpu_teardown()
            call finalize(iOutFile, ierr, 1)
            job_active    = .false.
            job_open_flag = .false.
        end if
    end subroutine job_destroy

    ! -----------------------------------------------------------------------
    ! Array result getters
    ! Each checks the availability flag and sets had_error if not computed.
    ! -----------------------------------------------------------------------

    subroutine run_esp_charges()
        ! ESP-fitted charges (ESP_CHARGE keyword). Calls the steps of QUICK's
        ! compute_oeprop directly -- surface, ESP on it, charge fit -- but skips
        ! writing the <stem>.vdw surface file, so log=False leaves nothing on
        ! disk (quick_open cannot target the null device: it renames an
        ! existing file with mv first).
#ifdef MPIV
        use mpi
        use quick_mpi_module, only : quick_comm
#endif
        double precision, allocatable :: esp_on_points(:)
#ifdef MPIV
        integer :: ierr
#endif

        if (allocated(esp_charges)) deallocate(esp_charges)
        if (.not. quick_method%esp_charge) return

#ifdef MPIV
        ! as compute_oeprop: the master builds the surface, every rank gets the
        ! points; compute_esp splits the shells over ranks and reduces onto the
        ! master; compute_ESP_charge (the fit) runs on the master only
        if (master) call generate_MKS_surfaces()
        call MPI_Bcast(quick_molspec%nvdwpoint, 1, MPI_INTEGER, 0, quick_comm, ierr)
        if (.not. master) allocate(quick_molspec%vdwpointxyz(3, quick_molspec%nvdwpoint))
        call MPI_Bcast(quick_molspec%vdwpointxyz, 3 * quick_molspec%nvdwpoint, &
                       MPI_DOUBLE_PRECISION, 0, quick_comm, ierr)
#else
        call generate_MKS_surfaces()
#endif
        allocate(esp_on_points(quick_molspec%nvdwpoint))
        call compute_esp(quick_molspec%nvdwpoint, quick_molspec%vdwpointxyz, esp_on_points)
        if (master) call compute_ESP_charge(quick_molspec%nvdwpoint, quick_molspec%vdwpointxyz, esp_on_points)
        deallocate(esp_on_points)
        deallocate(quick_molspec%vdwpointxyz)
    end subroutine run_esp_charges

    subroutine job_get_esp_charges(charges, n)
        !f2py intent(out) charges, n
        integer, intent(out) :: n
        double precision, intent(out) :: charges(10000)
        charges = 0.0d0
        n = 0
        if (.not. job_has_esp) then
            call fail("ESP charges were not computed; request properties=['esp_charges']")
            return
        end if
        n = size(esp_charges)
        charges(1:n) = esp_charges
    end subroutine job_get_esp_charges

    subroutine job_get_mulliken(charges, n)
        ! f2py cannot use external module variables as C array bounds, so we
        ! use a literal upper bound and return the actual count in n.
        !f2py intent(out) charges, n
        integer, intent(out) :: n
        double precision, intent(out) :: charges(10000)
        charges = 0.0d0
        if (.not. job_has_mulliken) then
            call fail("'mulliken' charges were not computed; " // &
                      "include DIPOLE in the keyword line via set_method('DIPOLE')")
            n = 0
            return
        end if
        n = natom
        charges(1:natom) = quick_qm_struct%Mulliken(1:natom)
    end subroutine job_get_mulliken

    subroutine job_get_lowdin(charges, n)
        !f2py intent(out) charges, n
        integer, intent(out) :: n
        double precision, intent(out) :: charges(10000)
        charges = 0.0d0
        if (.not. job_has_lowdin) then
            call fail("'lowdin' charges were not computed; " // &
                      "include DIPOLE in the keyword line via set_method('DIPOLE')")
            n = 0
            return
        end if
        n = natom
        charges(1:natom) = quick_qm_struct%Lowdin(1:natom)
    end subroutine job_get_lowdin

    subroutine job_get_mo_energies(energies, n)
        !f2py intent(out) energies, n
        integer, intent(out) :: n
        double precision, intent(out) :: energies(10000)
        energies = 0.0d0
        if (.not. job_has_mo_energies) then
            call fail("'mo_energies' were not computed; run() must complete successfully")
            n = 0
            return
        end if
        n = NBSuse
        energies(1:NBSuse) = quick_qm_struct%E(1:NBSuse)
    end subroutine job_get_mo_energies

    subroutine job_get_density_matrix(dm, nr, nc)
        ! Returns QUICK's dense -- the TOTAL density on restricted runs, the ALPHA
        ! density on unrestricted ones -- as a 1D (row-major) array of length
        ! nr*nc = nbasis*nbasis.  Reshape in Python: dm.reshape(nr, nc).
        ! We cap at 3000*3000 = 9_000_000 elements; large basis sets are rare.
        !f2py intent(out) dm, nr, nc
        integer, intent(out) :: nr, nc
        double precision, intent(out) :: dm(9000000)
        integer :: i, j, idx
        dm = 0.0d0
        if (.not. job_has_density_matrix) then
            call fail("'density_matrix' was not computed; run() must complete successfully")
            nr = 0
            nc = 0
            return
        end if
        nr = nbasis
        nc = nbasis
        do j = 1, nbasis
            do i = 1, nbasis
                idx = (i - 1) * nbasis + j
                dm(idx) = quick_qm_struct%dense(i, j)
            end do
        end do
    end subroutine job_get_density_matrix

    subroutine job_get_beta_mo_energies(energies, n)
        ! Beta MO energies (unrestricted runs only); see job_get_mo_energies.
        !f2py intent(out) energies, n
        integer, intent(out) :: n
        double precision, intent(out) :: energies(10000)
        energies = 0.0d0
        n = 0
        if (.not. job_has_beta) then
            call fail("'beta_mo_energies' exist only for unrestricted (open-shell) runs")
            return
        end if
        n = NBSuse
        energies(1:NBSuse) = quick_qm_struct%Eb(1:NBSuse)
    end subroutine job_get_beta_mo_energies

    subroutine job_get_beta_density_matrix(dm, nr, nc)
        ! Beta density matrix (unrestricted runs only), row-major like
        ! job_get_density_matrix.
        !f2py intent(out) dm, nr, nc
        integer, intent(out) :: nr, nc
        double precision, intent(out) :: dm(9000000)
        integer :: i, j
        dm = 0.0d0
        nr = 0
        nc = 0
        if (.not. job_has_beta) then
            call fail("'beta_density_matrix' exists only for unrestricted (open-shell) runs")
            return
        end if
        nr = nbasis
        nc = nbasis
        do j = 1, nbasis
            do i = 1, nbasis
                dm((i - 1) * nbasis + j) = quick_qm_struct%denseb(i, j)
            end do
        end do
    end subroutine job_get_beta_density_matrix

    subroutine compute_kinetic_energy()
        ! E_kinetic = Tr(P*T). QUICK keeps only T + V (oneElecO), so rebuild T with
        ! its own kineticO. That routine writes the lower triangle of the Fock array
        ! o, which later persistent-job steps still need, so save and restore it.
        double precision, allocatable :: saved_o(:,:)
        double precision :: p
        integer :: i, j

        saved_o = quick_qm_struct%o
        quick_qm_struct%o = 0.0d0
        do i = 1, nbasis
            call kineticO(i)
        end do
        job_e_kinetic = 0.0d0
        do i = 1, nbasis
            do j = i, nbasis
                p = quick_qm_struct%dense(j, i)
                if (quick_method%unrst) p = p + quick_qm_struct%denseb(j, i)
                if (j /= i) p = 2.0d0 * p
                job_e_kinetic = job_e_kinetic + p * quick_qm_struct%o(j, i)
            end do
        end do
        quick_qm_struct%o = saved_o
    end subroutine compute_kinetic_energy

    subroutine reset_engine_state()
        ! State that outlives a calculation in QUICK, which normally runs one per
        ! process, and that initialize1 and finalize leave alone. Without this, after
        ! a LOPT run every later optimization stays on LOPT; and under MPI basis.f90
        ! allocates its load-balancing tables once, sized by the first molecule, so
        ! a later larger molecule overruns them.
        quick_method%usedlfind = .true.
#ifdef MPIV
        if (allocated(mpi_jshell)) deallocate(mpi_jshell)
        if (allocated(mpi_nbasis)) deallocate(mpi_nbasis)
#endif
    end subroutine reset_engine_state

    subroutine trajectory_begin()
        ! Both optimizers and the SCF record each step in quick_molden when
        ! write_molden is on, so the trajectory needs no QUICK source change. Do
        ! what initialize_molden does, minus its two problems for pyquick: it sizes
        ! the arrays from iopt and iscf without margin (nscf_snapshots even by the
        ! SCF limit, not the step count), and it opens a real .molden file. UHF also
        ! writes MOs to that unit, so give it an unnamed scratch file (the null device
        ! may already be the log, and gfortran refuses it on a second unit). Done on
        ! every rank: workers reach some of the same code.
        integer :: nsnap, ios

        job_trajectory_nsteps = 0
        nsnap = quick_method%iopt + 2
        call trajectory_free_storage()
        allocate(quick_molden%nscf_snapshots(nsnap))
        allocate(quick_molden%e_snapshots(quick_method%iscf + 2, nsnap))
        allocate(quick_molden%xyz_snapshots(3, natom, nsnap))
        quick_molden%nscf_snapshots = 0
        quick_molden%e_snapshots    = 0.0d0
        quick_molden%natom            = natom
        quick_molden%opt              = .true.
        quick_molden%iexport_snapshot = 1
        quick_molden%iMoldenFile      = iMoldenFile
        open(unit=iMoldenFile, status='SCRATCH', form='FORMATTED', iostat=ios)
        if (ios /= 0) call fail('geo_opt: could not open a scratch file')
        write_molden = .true.
    end subroutine trajectory_begin

    subroutine trajectory_end()
        ! Copy the steps out and stop the Molden path, so finalize leaves it alone.
        integer :: i, n

        write_molden = .false.
        close(iMoldenFile)
        if (allocated(traj_coords))   deallocate(traj_coords)
        if (allocated(traj_energies)) deallocate(traj_energies)
        if (master) then
            n = min(quick_molden%iexport_snapshot - 1, size(quick_molden%nscf_snapshots))
            allocate(traj_coords(3, natom, n), traj_energies(n))
            do i = 1, n
                traj_coords(:, :, i) = quick_molden%xyz_snapshots(:, :, i) / A_TO_BOHRS
                traj_energies(i) = quick_molden%e_snapshots(quick_molden%nscf_snapshots(i), i)
            end do
            job_trajectory_nsteps = n
            job_trajectory_natoms = natom
            job_has_trajectory    = n > 0
        end if
        call trajectory_free_storage()
    end subroutine trajectory_end

    subroutine trajectory_free_storage()
        ! reord_mo_vec is QUICK's, sized by nbasis once: free it for the next molecule
        if (allocated(quick_molden%nscf_snapshots)) deallocate(quick_molden%nscf_snapshots)
        if (allocated(quick_molden%e_snapshots))    deallocate(quick_molden%e_snapshots)
        if (allocated(quick_molden%xyz_snapshots))  deallocate(quick_molden%xyz_snapshots)
        if (allocated(quick_molden%reord_mo_vec))   deallocate(quick_molden%reord_mo_vec)
    end subroutine trajectory_free_storage

    subroutine job_get_trajectory_energies(energies, n)
        ! Energy of each optimization step (Eel + Ecore, as QUICK records it).
        integer, intent(in) :: n
        double precision, intent(out) :: energies(n)
        energies = traj_energies(1:n)
    end subroutine job_get_trajectory_energies

    subroutine job_get_trajectory_coordinates(coords, n)
        ! Flat, row-major (step, atom, xyz) in Angstrom, n = nsteps*natoms*3.
        ! Reshape in Python: coords.reshape(nsteps, natoms, 3).
        integer, intent(in) :: n
        double precision, intent(out) :: coords(n)
        integer :: i, j, k, idx
        idx = 0
        do k = 1, job_trajectory_nsteps
            do i = 1, job_trajectory_natoms
                do j = 1, 3
                    idx = idx + 1
                    coords(idx) = traj_coords(j, i, k)
                end do
            end do
        end do
    end subroutine job_get_trajectory_coordinates

    subroutine job_get_mo_coefficients(c, n)
        ! AO -> MO coefficients, flat and row-major, n = nbasis*nmo.
        ! Reshape in Python: c.reshape(nbasis, nmo); columns are MOs.
        integer, intent(in) :: n
        double precision, intent(out) :: c(n)
        c = reshape(transpose(quick_qm_struct%co(1:nbasis, 1:NBSuse)), [n])
    end subroutine job_get_mo_coefficients

    subroutine job_get_beta_mo_coefficients(c, n)
        ! As job_get_mo_coefficients, for the beta orbitals of an unrestricted run.
        integer, intent(in) :: n
        double precision, intent(out) :: c(n)
        c = reshape(transpose(quick_qm_struct%cob(1:nbasis, 1:NBSuse)), [n])
    end subroutine job_get_beta_mo_coefficients

    subroutine job_get_overlap(s, n)
        ! AO overlap matrix, flat, n = nbasis*nbasis (symmetric, so order is free).
        integer, intent(in) :: n
        double precision, intent(out) :: s(n)
        s = reshape(quick_qm_struct%s(1:nbasis, 1:nbasis), [n])
    end subroutine job_get_overlap

    subroutine job_get_ao_info(centers, powers, n)
        ! For each of the n = nbasis AOs: its atom (1-based) and the Cartesian
        ! powers (lx, ly, lz) of its function, flat as (lx1,ly1,lz1, lx2,...).
        integer, intent(in) :: n
        integer, intent(out) :: centers(n), powers(3 * n)
        integer :: i
        do i = 1, n
            centers(i) = quick_basis%ncenter(i)
            powers(3 * i - 2:3 * i) = itype(1:3, i)
        end do
    end subroutine job_get_ao_info

    subroutine job_get_point_charge_gradient(g, n)
        ! Gradient on the external point charges, flat and row-major per charge:
        ! (dx1,dy1,dz1, dx2,...), length 3*n, Hartree/Bohr. Reshape: g[:3*n].reshape(n, 3).
        !f2py intent(out) g, n
        integer, intent(out) :: n
        double precision, intent(out) :: g(300000)
        g = 0.0d0
        n = 0
        if (.not. job_has_ptchg_gradient) then
            call fail("the point-charge gradient was not computed; " // &
                      "use get_grad() with point_charges")
            return
        end if
        n = quick_molspec%nextatom
        g(1:3*n) = quick_qm_struct%ptchg_gradient(1:3*n)
    end subroutine job_get_point_charge_gradient

    subroutine job_get_gradients(g, n)
        ! Returns the nuclear gradient as a flat, row-major (per-atom) array of
        ! length 3*n = 3*natom: (dx1,dy1,dz1, dx2,dy2,dz2, ...), in Hartree/Bohr.
        ! Reshape in Python: g[:3*n].reshape(n, 3).
        !f2py intent(out) g, n
        integer, intent(out) :: n
        double precision, intent(out) :: g(30000)
        integer :: i
        g = 0.0d0
        if (.not. job_has_gradient) then
            call fail("'gradient' was not computed; use get_grad() (not get_energy())")
            n = 0
            return
        end if
        n = natom
        do i = 1, 3 * natom
            g(i) = quick_qm_struct%gradient(i)
        end do
    end subroutine job_get_gradients

    subroutine job_get_optimized_geometry(coords, n)
        ! Returns the geometry after a successful OPT job: QUICK optimizes
        ! quick_molspec%xyz in place, in Bohr, so convert back to Angstrom here
        ! to match the units read_geom accepts and job_get_geometry returns.
        ! Flat, row-major per atom: (x1,y1,z1, x2,y2,z2, ...); reshape (n, 3).
        !f2py intent(out) coords, n
        integer, intent(out) :: n
        double precision, intent(out) :: coords(30000)
        integer :: i, j
        coords = 0.0d0
        if (.not. job_has_optimized) then
            call fail("'optimized_coordinates' were not computed; use geo_opt()")
            n = 0
            return
        end if
        n = natom
        do i = 1, natom
            do j = 1, 3
                coords((i - 1) * 3 + j) = xyz(j, i) / A_TO_BOHRS
            end do
        end do
    end subroutine job_get_optimized_geometry

    subroutine job_get_geometry(atnums, coords, n)
        ! Returns QUICK's own parsed geometry: atomic numbers and Angstrom
        ! coordinates as stored by read_geom (no re-parsing on the Python side).
        ! coords is flattened row-major per atom: (x1,y1,z1, x2,y2,z2, ...);
        ! reshape in Python: coords[:3*n].reshape(n, 3).
        !f2py intent(out) atnums, coords, n
        integer, intent(out) :: n
        integer, intent(out) :: atnums(10000)
        double precision, intent(out) :: coords(30000)
        integer :: i, j
        atnums = 0
        coords = 0.0d0
        if (.not. has_geom) then
            call fail('job_get_geometry: no geometry set; call read_geom first')
            n = 0
            return
        end if
        n = geom_natom
        do i = 1, geom_natom
            atnums(i) = geom_atnum(i)
            do j = 1, 3
                coords((i - 1) * 3 + j) = geom_coords(j, i)
            end do
        end do
    end subroutine job_get_geometry

    ! -----------------------------------------------------------------------
    ! Private helpers
    ! -----------------------------------------------------------------------

    subroutine set_point_charges(charges, n)
        ! Set the external point charges for the following runs (replaces any
        ! previous set): charges(1:3,i) in Angstrom, charges(4,i) the charge.
        integer, intent(in) :: n
        double precision, intent(in) :: charges(4, n)

        if (allocated(pc_data)) deallocate(pc_data)
        allocate(pc_data(4, n))
        pc_data = charges
        pc_n = n
    end subroutine set_point_charges

    subroutine clear_point_charges()
        if (allocated(pc_data)) deallocate(pc_data)
        pc_n = 0
    end subroutine clear_point_charges

    subroutine prepare_point_charges()
        ! Before alloc(quick_molspec), which sizes extxyz/extchg from nextatom.
        ! initialize1 resets nextatom to 0 and isTemplate skips QUICK's own
        ! reading of the charges, so pyquick sets the count itself. Drop any
        ! arrays left from an earlier run so alloc sizes them for this one.
        if (allocated(quick_molspec%extxyz)) deallocate(quick_molspec%extxyz)
        if (allocated(quick_molspec%extchg)) deallocate(quick_molspec%extchg)
        quick_molspec%nextatom = pc_n
    end subroutine prepare_point_charges

    subroutine inject_point_charges()
        ! After alloc(quick_molspec): copy the charges in (Angstrom -> Bohr),
        ! as set_quick_molspecs in quick_api_module does for the AMBER API.
        integer :: i

        do i = 1, pc_n
            quick_molspec%extxyz(1:3, i) = pc_data(1:3, i) * A_TO_BOHRS
            quick_molspec%extchg(i)      = pc_data(4, i)
        end do
    end subroutine inject_point_charges

    function build_keyword_line() result(text)
        character(len=:), allocatable :: text
        integer :: i

        text = trim(calc_keyword)

        if (allocated(method_list)) then
            do i = 1, size(method_list)
                if (len_trim(method_list(i)%arg) > 0) then
                    text = trim(text) // ' ' // trim(method_list(i)%keyword) &
                           // '=' // trim(method_list(i)%arg)
                else
                    text = trim(text) // ' ' // trim(method_list(i)%keyword)
                end if
            end do
        end if

        if (has_basis) text = trim(text) // ' ' // trim(basis_token)

        ! EXTCHARGES makes QUICK report ECharge; the charges themselves are
        ! injected by prepare_/inject_point_charges, not read from an input file
        if (pc_n > 0) text = trim(text) // ' EXTCHARGES'

    end function build_keyword_line

    subroutine clear_methods()
        if (allocated(method_list)) deallocate(method_list)
        call rebuild_input()
    end subroutine clear_methods

    subroutine append_method(uname, arg)
        character(len=*), intent(in) :: uname
        character(len=*), intent(in), optional :: arg
        type(method_entry), allocatable :: tmp(:)
        integer :: n

        if (.not. allocated(method_list)) then
            allocate(method_list(1))
            n = 1
        else
            n = size(method_list) + 1
            allocate(tmp(n))
            tmp(1:n-1) = method_list
            call move_alloc(tmp, method_list)
        end if

        method_list(n)%keyword = trim(uname)
        if (present(arg) .and. len_trim(arg) > 0) then
            method_list(n)%arg = trim(adjustl(arg))
        else
            method_list(n)%arg = ''
        end if
    end subroutine append_method

    subroutine fail(message)
        ! Record an error; Python's _checked() raises it as RuntimeError.
        ! MPI with more than one rank: once QUICK is running (engine_running),
        ! a failure may be local to one rank while the others wait in a
        ! collective call, so returning would leave them hanging. Abort every
        ! rank instead, as QUICK's own quick_exit does. Errors before QUICK
        ! starts (setup, geometry) occur on every rank alike and stay catchable.
#ifdef MPIV
        use mpi
        use iso_fortran_env, only : error_unit
        integer :: ierr_mpi
#endif
        character(len=*), intent(in) :: message
        had_error     = .true.
        error_message = trim(message)
#ifdef MPIV
        if (engine_running .and. quick_comm_size > 1) then
            write(error_unit, '(A,I0,A)') 'pyquick: rank ', quick_comm_rank, ': ' // &
                trim(message) // ' -- aborting all MPI ranks'
            flush(error_unit)
            call MPI_Abort(MPI_COMM_WORLD, 1, ierr_mpi)
        end if
#endif
    end subroutine fail

    ! -----------------------------------------------------------------------
    ! MPI (libquick_mpi builds). Python owns MPI through mpi4py; QUICK borrows
    ! it in library mode, exactly as Amber does through setQuickMPI.
    ! Both routines exist in every build -- only their bodies are guarded --
    ! because f2py scans this file before preprocessing and would otherwise
    ! wrap a symbol that a serial build never compiles.
    ! -----------------------------------------------------------------------

    subroutine set_mpi(rank, nprocs)
        ! Give QUICK its own copy of MPI_COMM_WORLD before its first job, as
        ! setQuickMPI does for Amber. libMPIMode makes initialize1 skip MPI_Init
        ! (mpi4py already called it); finalize(..., 1), used throughout this
        ! module, already skips MPI_Finalize. No-op outside MPI builds.
#ifdef MPIV
        use mpi
        use quick_mpi_module, only : quick_comm, mpi_world_rank, mpi_world_size
#endif
        integer, intent(in) :: rank, nprocs
#ifdef MPIV
        integer :: ierr_mpi

        call MPI_Comm_dup(MPI_COMM_WORLD, quick_comm, ierr_mpi)
        quick_comm_rank = rank
        quick_comm_size = nprocs
        mpi_world_rank  = rank
        mpi_world_size  = nprocs
        libMPIMode      = .true.
        master          = (rank == 0)
#endif
    end subroutine set_mpi

    subroutine get_build_info(is_mpi, is_gpu)
        ! is_mpi: built against an MPI libquick (libquick_mpi, libquick_mpi_cuda).
        ! is_gpu: built against a GPU libquick (libquick_cuda/_hip, or the
        !         multi-GPU libquick_mpi_cuda/_hip).
        !f2py intent(out) is_mpi, is_gpu
        logical, intent(out) :: is_mpi, is_gpu
#ifdef MPIV
        is_mpi = .true.
#else
        is_mpi = .false.
#endif
#if defined(GPU) || defined(MPIV_GPU)
        is_gpu = .true.
#else
        is_gpu = .false.
#endif
    end subroutine get_build_info

    subroutine start_mgpu(caller)
        ! Multi-GPU builds (MPIV_GPU): give every rank a GPU, as main.f90 does --
        ! rank r takes visible device r % count, so ranks share GPUs when there
        ! are more ranks than GPUs. Errors go through fail() instead of
        ! SAFE_CALL's exit(); with more than one rank, fail() aborts every rank.
        ! Empty outside MPIV_GPU builds.
#if defined(MPIV_GPU)
        use quick_mpi_module, only : quick_comm, mpi_world_rank, mgpu_id, mgpu_ids
#endif
        character(len=*), intent(in) :: caller
#if defined(MPIV_GPU)
        external :: gpu_new, mgpu_query, mgpu_setup, mgpu_write_info, mgpu_init_device
        integer :: ierr
        character(len=12) :: num

        ierr = 0
        call gpu_new(quick_comm_rank, ierr)
        call mgpu_query(mpi_world_rank, mgpu_id, ierr)
        if (ierr /= 0) then
            write(num, '(I0)') ierr
            call fail(caller // ': no usable GPU for this rank (QUICK error ' // trim(num) // ')')
            return
        end if
        call mgpu_setup()                  ! master gathers every rank's device id
        if (master) call mgpu_write_info(iOutFile, quick_comm_size, mgpu_ids, ierr)
        call mgpu_init_device(quick_comm, quick_comm_rank, quick_comm_size, mgpu_id, ierr)
#endif
    end subroutine start_mgpu

    subroutine open_log(caller, do_log, open_mode)
        ! Rank 0 (and every serial run): set QUICK's file names, open its log
        ! (<output_stem>.out, or the null device when do_log == 0) and write the
        ! header, as main.f90 does on its master. open_mode: 'R' replaces the
        ! file, 'A' appends to it. Errors are reported through fail().
#ifdef MPIV
        use quick_mpi_module, only : print_quick_mpi
#endif
        external :: outputCopyright, PrtDate, quick_open
        character(len=*), intent(in) :: caller
        integer, intent(in) :: do_log
        character(len=1), intent(in) :: open_mode
        integer :: ierr
        character(len=256) :: note

        ierr = 0
        call set_quick_files(.true., ierr)
        if (ierr /= 0) then; call fail(caller // ': set_quick_files failed'); return; end if

        if (do_log /= 0) then
            ! a real <output_stem>.out file, as set_quick_files derived it;
            ! quick_open backs up any pre-existing file to '<file>~'
            call quick_open(iOutFile, outFileName, 'U', 'F', open_mode, .false., ierr)
            if (ierr /= 0) then; call fail(caller // ': quick_open failed'); return; end if
        else
            call open_null_log(caller)
            if (had_error) return
        end if

        call outputCopyright(iOutFile, ierr)
        if (ierr /= 0) then; call fail(caller // ': outputCopyright failed'); return; end if
        note = 'TASK STARTS ON:'
        call PrtDate(iOutFile, note, ierr)
        if (ierr /= 0) then; call fail(caller // ': PrtDate failed'); return; end if
        call print_quick_io_file(iOutFile, ierr)
        if (ierr /= 0) then; call fail(caller // ': print_quick_io_file failed'); return; end if
#ifdef MPIV
        call print_quick_mpi(iOutFile, ierr)
#endif
    end subroutine open_log

    subroutine open_null_log(caller)
        ! Send QUICK's log to the platform null device (log=False, and every MPI
        ! worker rank) so the run leaves nothing on disk. Opened directly rather
        ! than through quick_open, which would try to back up the device with
        ! 'mv'. The device name is chosen at runtime (NUL on Windows).
        character(len=*), intent(in) :: caller
        character(len=:), allocatable :: null_dev
        logical :: unit_open
        integer :: ios

        null_dev = null_device()
        outFileName = null_dev
        inquire(unit=iOutFile, opened=unit_open)
        if (unit_open) close(iOutFile)
        open(unit=iOutFile, file=null_dev, status='UNKNOWN', form='FORMATTED', &
             action='WRITE', iostat=ios)
        if (ios /= 0) call fail(caller // ': could not open null output device')
    end subroutine open_null_log

    subroutine job_exception_message(code, msg)
        ! Expose QUICK's error text for a code, so Python can report/inspect it.
        !f2py intent(out) msg
        integer, intent(in) :: code
        character(len=200), intent(out) :: msg
        call get_exception_message(code, msg)
    end subroutine job_exception_message

    function uppercase(text) result(upper)
        character(len=*), intent(in) :: text
        character(len=len(text)) :: upper
        integer :: i

        upper = text
        do i = 1, len(text)
            select case (upper(i:i))
            case ('a':'z')
                upper(i:i) = achar(iachar(upper(i:i)) - 32)
            end select
        end do
    end function uppercase

    function null_device() result(dev)
        ! Platform-appropriate null device for discarding QUICK's log.
        ! Detected at runtime so the choice does not depend on preprocessor
        ! flags or a specific Fortran compiler's predefined macros:
        !   Windows sets OS=Windows_NT and defines %WINDIR% -> use 'NUL'
        !   Unix/macOS -> use '/dev/null'
        character(len=:), allocatable :: dev
        character(len=64) :: val
        integer :: length, stat

        call get_environment_variable('OS', val, length, stat)
        if (stat == 0 .and. length > 0) then
            if (index(uppercase(val(1:min(length, len(val)))), 'WINDOWS') > 0) then
                dev = 'NUL'
                return
            end if
        end if

        call get_environment_variable('WINDIR', val, length, stat)
        if (stat == 0 .and. length > 0) then
            dev = 'NUL'
            return
        end if

        dev = '/dev/null'
    end function null_device

end module pyquick
