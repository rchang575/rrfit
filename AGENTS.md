# rrfit Agent Guide

## Project Overview

- `rrfit` is a small Python package for fitting microwave resonator measurements.
- The codebase is numerics-heavy and used for scientific analysis, so correctness and reproducibility matter more than broad refactors.
- Packaging uses Poetry. Source code lives under `rrfit/`.

## Working Priorities

- Prefer small, verifiable changes over large rewrites.
- Preserve numerical behavior unless the task is explicitly to fix a bug.
- When changing fit logic, add or update tests first whenever practical.
- Focus on reliability improvements: input validation, tests, clearer failure modes, and removal of hidden side effects.

## Safe Change Guidelines

- Do not change model equations, parameter conventions, or return shapes without calling that out clearly.
- Avoid in-place mutation of caller-provided arrays unless the API explicitly documents it.
- Keep plotting concerns separate from core computation where possible.
- Replace unconditional `print(...)` behavior with optional verbosity or structured errors when touching public APIs.

## Testing Expectations

- Add automated tests for new behavior and bug fixes.
- Prioritize coverage for:
  - `rrfit.circlefit`
  - `rrfit.delayfit`
  - `rrfit.magfit`
  - `rrfit.hangerfit`
  - `rrfit.dataio`
- Use small synthetic datasets where possible so tests stay fast and deterministic.

## Style Notes

- Match the existing lightweight style, but improve docstrings and type hints when editing code.
- Keep public function signatures stable unless the user asks for an API change.
- Prefer explicit error messages over silent failure or stdout-only reporting.

## Good Starter Tasks

- Add a minimal `pytest` suite for core fitting helpers.
- Fix reliability issues in `rrfit/dataio.py` defaults and loading behavior.
- Remove unintended input mutation in `rrfit/hangerfit.py`.
- Improve README usage examples once tests exist.
