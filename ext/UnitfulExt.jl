module UnitfulExt

using Unitful
using Unitful: 𝐓, NoDims, dimension, ustrip, unit
using Distributions: Exponential, Distribution, Univariate, Continuous, Discrete
using AlgebraicABMs
using AlgebraicABMs.ABMs: AbsTimer, AbsHazard, ContinuousHazard, DiscreteHazard,
                          ClosureTime, ClosureState, FullClosure, Maybe

import AlgebraicABMs.ABMs: validate_units, strip_units

"""
    validate_units(rate, expected_dim=𝐓^-1)

Validate that a rate quantity has the expected dimensions.
For hazard rates, the expected dimension is inverse time (𝐓⁻¹).
Returns `true` if valid, throws `DimensionError` if not.

# Example
```julia
using Unitful
validate_units(0.1u"1/d")  # true
validate_units(0.1u"m")    # DimensionError
```
"""
function validate_units(rate::Unitful.Quantity, expected_dim=𝐓^-1)
  d = dimension(rate)
  d == expected_dim || throw(Unitful.DimensionError(expected_dim, d))
  return true
end

validate_units(::Number, args...) = true

"""
    strip_units(hazard::ContinuousHazard)
    strip_units(hazard::DiscreteHazard)
    strip_units(val::Unitful.Quantity)

Strip units from a hazard rate or quantity, returning a dimensionless value.
Useful for passing unitful rates into the simulation engine which operates
on plain Float64 values.

# Example
```julia
using Unitful
strip_units(0.1u"1/d")  # 0.1
strip_units(ContinuousHazard(Exponential(ustrip(u"d", 10u"d"))))
```
"""
strip_units(val::Unitful.Quantity) = ustrip(val)
strip_units(val::Number) = val

"""
    ContinuousHazard(rate::Unitful.Quantity{<:Real, 𝐓^-1})

Construct a ContinuousHazard from a unitful rate. The rate must have dimensions
of inverse time (e.g., `0.1u"1/d"`, `5.0u"1/hr"`). Units are stripped and
the underlying value is used as the Exponential rate parameter.

# Example
```julia
using Unitful
h = ContinuousHazard(0.1u"1/d")  # Exponential with mean 10 (days)
```
"""
function ContinuousHazard(rate::Unitful.Quantity)
  validate_units(rate, 𝐓^-1)
  ContinuousHazard(Exponential(1.0 / ustrip(rate)))
end

"""
    DiscreteHazard(time::Unitful.Quantity{<:Real, 𝐓})

Construct a DiscreteHazard from a unitful time. The time must have dimensions
of time (e.g., `1.0u"d"`, `24.0u"hr"`). Units are stripped.

# Example
```julia
using Unitful
h = DiscreteHazard(1.0u"d")  # Dirac delta at t=1 (in days)
```
"""
function DiscreteHazard(time::Unitful.Quantity)
  validate_units(time, Unitful.𝐓)
  DiscreteHazard(ustrip(time))
end

"""
    check_time_units(maxtime, dt=nothing)

Validate and strip time units from simulation parameters.
Returns (maxtime_stripped, dt_stripped) as plain numbers.

# Example
```julia
using Unitful
mt, d = check_time_units(100u"d", 0.1u"d")  # (100.0, 0.1)
```
"""
function check_time_units(maxtime::Unitful.Quantity, dt::Union{Unitful.Quantity,Nothing}=nothing)
  validate_units(maxtime, Unitful.𝐓)
  mt = ustrip(maxtime)
  d = isnothing(dt) ? nothing : begin
    validate_units(dt, Unitful.𝐓)
    ustrip(dt)
  end
  return (mt, d)
end

check_time_units(maxtime::Number, dt=nothing) = (maxtime, dt)

end # module
