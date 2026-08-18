# pyquick GPU — delorean status (session handoff)

Written 2026-08-18 from the Mac session. Read together with
`ai-content/pyquick_gpu_approach.md` (the plan) and `ai-content/pyquick_progress_summary.md`
(what pyquick is). This file is only "where we got to on the box".

## The box

`delorean` — standalone RHEL 9 workstation, NVIDIA GPU, no job scheduler (run things
directly, no srun). Home is `/server-home1/yud042`. Repo at `~/QUICK`, branch `API-dev-GPU`.
GCC/gfortran 11.5.0, binutils 2.35.2.

## Workflow decided

- VS Code Remote-SSH for editing; **Claude Code runs as the terminal TUI** on delorean.
  The VS Code sidebar extension never activated (`command 'claude-vscode.editor.openLast'
  not found`, `~/.claude/ide/` never created) and was abandoned — not worth more time.
- **Long builds run in tmux, piped to a log**, never as an agent tool call:
  `tmux new -s build; make -j8 2>&1 | tee build.log`. A dropped SSH session then costs
  nothing and the agent reads `build.log` afterwards.
- Delorean is the working copy for GPU work; the Mac pulls and re-runs the CPU pytest
  suite to confirm the serial path still builds (the `-cpp` change touches both branches).

## Python environment — RESOLVED THE HARD WAY, do not repeat

Two separate failures, in order:

1. **System python3.9 + numpy 1.23.5**: f2py's `--backend meson` flag did not exist before
   numpy 1.26.0, and `src/pyquick/CMakeLists.txt` hardcodes it. Fix = numpy >= 1.26.
2. **venv from the system python still failed**: meson died with
   `Run-time dependency python found: NO (tried pkgconfig and sysconfig)` →
   `meson.build:12:12: ERROR: Python dependency not found`. Cause: **no Python development
   headers** on the box (`python3-devel` not installed, no root to install it). A venv
   inherits headers from its base interpreter, so it cannot fix this.

Resolution: use an interpreter that ships its own headers + pkg-config — a `module load`
Python if one exists, otherwise miniforge. Python **3.11**, not 3.9 (meson already warns
3.9 is EOL and meson 1.12 will drop it; nothing in pyquick needs 3.9 — that was a Homebrew
constraint on the Mac only). Pass that interpreter as `-DPython3_EXECUTABLE=...` and use it
for every build and test run. rdkit is only needed for the notebooks, not for the build or
the pytest suite.

## Where we actually are

Steps refer to the runbook in `pyquick_gpu_approach.md`.

- [x] Code on the box, branch `API-dev-GPU`.
- [x] `src/pyquick/CMakeLists.txt` GPU branch written (CPU-verified on the Mac: 47/47 pytest).
- [ ] **Step 2 — NOT DONE, this is the next thing.** CUDA toolchain build with
      `-DPYTHON=FALSE`, then run `quick.cuda` on `test/ene_H2O_rhf_sto3g.in` and confirm the
      device block appears in the `.out`. This step involves no Python at all and is
      completely independent of the environment mess above.
- [ ] Step 3 — reconfigure with `-DPYTHON=TRUE`; configure must print
      `pyquick: binding against GPU libquick (quick_cuda)`.
- [ ] Step 4 — verify pyquick used the device: `gpu_write_info` block in the `.out` AND
      energy matching the CPU reference -74.947863811 Ha to ~1e-6 (looser than the CPU
      test's tolerance; GPU cutoffs and reduction order differ).

## Unverified risk to check at step 3

f2py's meson backend has historically dropped `--f90flags`. If it does, `-DGPU` never reaches
gfortran, the guarded block in `pyquick.f90` compiles out, and you get a **CPU-only `.so`
that builds and links cleanly and looks like it works**. Check the generated `meson.build`
for `-cpp -DGPU` in `fortran_args` before trusting a green build. Fallback if the flags are
dropped: rename to `pyquick.F90` (capital F forces gfortran to preprocess unconditionally).
The step-4 runtime check is the backstop.
