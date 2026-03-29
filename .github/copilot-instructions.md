# Copilot Instructions for AlgebraicABMs.jl

## Overview

AlgebraicABMs.jl implements agent-based models (ABMs) using algebraic rewriting on attributed C-sets (ACSets). Models are defined as collections of stochastic rewrite rules with hazard-based timers, simulated via a competing-clocks (Gillespie-style) algorithm. The package builds on the AlgebraicJulia ecosystem — primarily Catlab (categorical data structures) and AlgebraicRewriting (DPO/SPO rewrite rules with incremental homomorphism search).

## Build and Test

```bash
# Run full test suite
julia --project=test test/runtests.jl

# Run a single test file
julia --project=test -e 'using Test; @testset "ABMs" begin include("test/ABMs.jl") end'

# Build and serve documentation locally
julia --project=docs -e 'using AlgebraicABMs, LiveServer; servedocs(literate_dir=joinpath("docs","literate"), skip_dir=joinpath("docs","src","examples"))'
```

CI uses shared AlgebraicJulia workflows (`.github/workflows/julia_ci.yml` → `AlgebraicJulia/.github/.github/workflows/julia_ci.yml@main`). There is also a Buildkite pipeline for HPC testing with 32 threads.

## Architecture

### Core Type Hierarchy

The simulation is built around these key abstractions in `src/ABMs.jl`:

- **`ABM`** — A collection of `ABMRule`s and optional `ABMFlow`s (continuous ODE dynamics, WIP). Rules are named and indexed.
- **`ABMRule`** — Pairs an AlgebraicRewriting `Rule` (L ← I → R span) with a timer/hazard. Has an auto-classified `PatternType` and an optional `basis` subobject for controlling match granularity.
- **Timer hierarchy** (`AbsTimer`):
  - `DiscreteHazard` / `ContinuousHazard` — Wrap `Distributions.jl` distributions. Shorthand: `DiscreteHazard(1.)` creates a Dirac delta, `ContinuousHazard(1.5)` creates an Exponential.
  - `FullClosure` — `(ACSetTransformation, clocktime) → hazard_rate` (state + time dependent)
  - `ClosureState` — `ACSetTransformation → hazard_rate` (state dependent only)
  - `ClosureTime` — `clocktime → hazard_rate` (time dependent only)
- **`PatternType`** — Auto-detected optimization for match enumeration:
  - `EmptyP` — Rule pattern L is empty (one trivial match, e.g. creation rules)
  - `RepresentableP` — L is a coproduct of representables; matches are sampled randomly without full homomorphism search (requires exponential timer)
  - `RegularP` — General case; uses incremental homomorphism search (`IncHomSet`)
- **`RuntimeABM`** — Mutable simulation state: current ACSet, incremental hom-sets (`AbsHomSet`), a `CompetingClocks.FirstToFire` sampler, and ODE problem. Created from an `ABM` + initial state.
- **`Traj`** — Trajectory output: initial state, event log `(time, rule_id, name, saved_data)`, and a history of rewrite spans.

### Simulation Loop (`run!`)

`run!(abm, init; maxevent, maxtime, save)` creates a `RuntimeABM` then loops:
1. Pop the next firing event(s) from the competing-clocks sampler (simultaneous events are batched)
2. For each event: find/sample a match morphism, execute the rewrite, record the span
3. After all events: update every rule's incremental hom-set via `deletion!`/`addition!`, disable invalidated clocks, enable new ones
4. Re-enable any fired matches that are still valid

### Module Organization

- `src/AlgebraicABMs.jl` — Top-level module, re-exports submodules via `@reexport`
- `src/ABMs.jl` — Core ABM types, rule classification, runtime, simulation loop
- `src/Upstream.jl` — Utilities intended for upstream contribution: `make_partial` (epi-mono factorization for spans), `IncHomSet_basis`, and `pop!`/`pops!` extensions for `CompetingClocks.FirstToFire`
- `src/Distributions.jl` — `weibullpar()` helper (Weibull shape/scale from mean/variance)
- `src/Visualization.jl` — `view(::Traj)` renders SVG timeline via Graphviz
- `ext/PetriInterface.jl` — Converts `AbstractPetriNet` to `ABM` (weak dep on AlgebraicPetri)
- `ext/MakieExt.jl` — Optional Makie-based visualization (weak dep on Makie)

### Extension Pattern

Package extensions use Julia's weak-dependency mechanism (`[weakdeps]` + `[extensions]` in Project.toml). The main module defines stub functions (e.g. `function PetriNetCSet end`) that extensions implement. Extensions import from internal submodules (e.g. `using AlgebraicABMs.ABMs: AbsDynamics`).

## Key Conventions

- **Rewrite rules as spans**: Rules are DPO or SPO rewrite rules expressed as spans `L ← I → R` using Catlab's `ACSetTransformation`. Constructors like `id()`, `create()`, `delete()`, `homomorphism()` from Catlab/AlgebraicRewriting build these morphisms.
- **`@struct_hash_equal`**: Most data types use `StructEquality.@struct_hash_equal` for structural equality and hashing instead of manual `==`/`hash` definitions.
- **`@reexport`**: The top-level module re-exports all submodules so users only need `using AlgebraicABMs`.
- **Type piracy in Upstream.jl**: `Upstream.jl` intentionally extends methods from Catlab and CompetingClocks. This is documented and meant for eventual upstreaming.
- **Test modules**: Each test file wraps its content in a module (e.g. `module TestABMs ... end`) to isolate namespace.
- **Literate examples**: Documentation examples live in `docs/literate/*.jl` as executable Julia scripts processed by Literate.jl into markdown.
- **Code quality**: `test/aqua.jl` runs Aqua.jl checks (ambiguities, unbound args, etc.).
