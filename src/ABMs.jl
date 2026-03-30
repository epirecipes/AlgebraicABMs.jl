
module ABMs

export ABM, ABMRule, ABMSchedule, run!, DiscreteHazard, ContinuousHazard, FullClosure, 
       ClosureState, ClosureTime, ClosureParams, FullClosureParams, ClosureHistory,
       RawODE, ABMFlow, filter, push!, copy, length, run_scenarios,
       Observable, Traj, TiePolicy, TieBreak, TieRandom, TieError,
       RuntimeABM, Intervention, refresh_clocks!,
       infer_schema, validate_schema,
       networkify, shortest_distance,
       resolve_match, match_equal,
       validate_units, strip_units,
       within_radius, pairwise_distances, positions,
       log_likelihood, particle_filter, abc_reject

using Distributions, CompetingClocks, Random
using DataStructures: DefaultDict
using DifferentialEquations: ODEProblem, solve, Tsit5
using StructEquality

using Catlab, AlgebraicRewriting
using AlgebraicRewriting.Incremental.Algorithms: connected_acset_components, pull_back
using AlgebraicRewriting.Rewrite.Migration: repr_dict
using Catlab.CategoricalAlgebra.Chase: extend_morphism_constraints
using AlgebraicRewriting.Rewrite.Utils: get_pmap, get_rmap, get_expr_binding_map
import Catlab: left, right
import AlgebraicRewriting: get_match, ruletype, addition!, deletion!, get_matches

import ..Upstream: pattern, pops!, IncHomSet_basis

# Timers
########

"""
Something that can produce a ACSetTransformation × clocktime → hazard_rate
"""
abstract type AbsTimer end

abstract type StateDependentTimer <: AbsTimer end

state_dep(t::AbsTimer) = t isa StateDependentTimer

"""
A closure which accepts a ACSetTransformation and returns a function of type
clocktime → hazard_rate
"""
struct FullClosure <: StateDependentTimer
  val::Function # ACSetTransformation → clocktime → hazard_rate
end

(c::FullClosure)(m::ACSetTransformation, t::Float64) = c.val(m,t)

"""
A closure which accepts a clocktime and returns a hazard_rate. This is a timer 
which cannot depend on the match data nor ACSet state.
"""
struct ClosureTime <: AbsTimer
  val::Function # clocktime → hazard_rate
end

(c::ClosureTime)(t::Float64) = c.val(t)

"""
A closure which accepts a match morphism and returns a hazard_rate. This is a 
timer which cannot depend on the absolute clock time.
"""
struct ClosureState <: StateDependentTimer
  val::Function # ACSetTransformation → hazard_rate
end

(c::ClosureState)(m::ACSetTransformation) = c.val(m)

"""
A closure which accepts a match morphism and a parameter dict, returning a 
hazard_rate. This is a timer which depends on the match state and exogenous 
parameters but not the absolute clock time.
"""
struct ClosureParams <: StateDependentTimer
  val::Function # (ACSetTransformation, params) → hazard_rate
end

(c::ClosureParams)(m::ACSetTransformation, params) = c.val(m, params)

"""
A closure which accepts a match morphism, clock time, and a parameter dict, 
returning a hazard_rate. The most general parameterized timer.
"""
struct FullClosureParams <: StateDependentTimer
  val::Function # (ACSetTransformation, clocktime, params) → hazard_rate
end

(c::FullClosureParams)(m::ACSetTransformation, t::Float64, params) = c.val(m, t, params)

"""
A closure which accepts a match morphism, clock time, and trajectory history,
returning a hazard_rate. This enables history-sensitive hazard rates where the
firing distribution depends on past events (e.g. time since infection, 
cumulative exposure, prior state transitions).

The trajectory is passed as a `Traj` object containing all events and rewrite
spans up to the current simulation time.
"""
struct ClosureHistory <: StateDependentTimer
  val::Function # (ACSetTransformation, clocktime, Traj) → hazard_rate
end

(c::ClosureHistory)(m::ACSetTransformation, t::Float64, traj) = c.val(m, t, traj)

abstract type AbsHazard <: AbsTimer end

@struct_hash_equal struct DiscreteHazard <: AbsHazard
  val::Distribution{Univariate, Discrete}
end

DiscreteHazard(t::Number) = DiscreteHazard(Dirac(t))

@struct_hash_equal struct ContinuousHazard <: AbsHazard
  val::Distribution{Univariate, Continuous}
end

"""Check if a hazard rate is a simple exponential"""
is_exp(h::ContinuousHazard) = h.val isa Distributions.Exponential
is_exp(h::AbsTimer) = false

ContinuousHazard(p::Number) = ContinuousHazard(Exponential(p))


# Rules 
#######
abstract type PatternType end

"""Empty patterns have (one) trivial pattern match"""
@struct_hash_equal struct EmptyP <: PatternType end

"""
Default case, where pattern matches should be found via (incremental) 
homomorphism search and represented explicitly, each with own events getting 
scheduled.
"""
@struct_hash_equal struct RegularP <: PatternType end

# """
# Special case of homsearch where no backtracking is needed. The only nonempty
# sets in L are those for objects with no outgoing homs. There may be attributes,
# however, so at runtime we must filter the sets before picking random elements.
# E.g. for labeled set L = {:a, :a, AttrVar(1)} we randomly pick two elements with
# label :a and one arbitrary element.

# WARNING: this is only viable if the timer associated with the rewrite rule is
# symmteric with respect to the discrete parts.
# """
# @struct_hash_equal struct DiscreteP <: PatternType
#   parts::Dict{Symbol, Int}
# end

"""
A pattern match from a coproduct of representables is just a choice of parts
in the codomain. E.g. matching L = •→• • •  is just a random choice of edge and
two random vertices.

The vector of ints refers to parts of L which are the counits of the left kan 
extensions that define the representables (usually this is just wherever the 
colimit leg sends 1, as there is often just one X part in the representable X).

WARNING: this is only viable if the timer associated with the rewrite rule is
symmteric with respect to the disjoint representables and has a simple
exponential timer.
"""
@struct_hash_equal struct RepresentableP <: PatternType
  parts::Dict{Symbol, Vector{Int}}
end

Base.keys(p::RepresentableP) = keys(p.parts)

multiplier(p::RepresentableP, X::ACSet) =
  prod(nparts(X, k)^length(v) for (k, v) in pairs(p.parts))

not_monic(b::Bool) = b === false 
not_monic(obs::AbstractVector{Symbol}) = isempty(obs)

"""
Analyze a pattern to find the most efficient pattern type for it.

Because ACSet types do not know their own equations, we may have to pass the 
schema as an argument in order to compute representables that would otherwise 
be infinite.

Even if the pattern is a coproduct of representables, we cannot use the 
efficient encoding unless the distribution is either an exponential 
(or a single dirac delta - not yet supported).
"""
function pattern_type(r::Rule, is_exp::Bool)
  p = pattern(r)
  
  # Check empty case
  isempty(p) && return EmptyP()

  # Determine if pattern is a coproduct of representables
  if is_exp && isempty(r.conditions) && not_monic(r.monic)
    repr_loc = DefaultDict{Symbol, Vector{Int}}(() -> Int[])
    reprs = repr_dict(typeof(p))
    ccs, iso′ = connected_acset_components(p)
    iso = invert_iso(iso′)
    for cc_leg in legs(ccs)
      found = false
      for (o, (repr, i)) in pairs(reprs)
        α = isomorphism(repr, dom(cc_leg)) 
        if !isnothing(α)
          push!(repr_loc[o], iso[o](cc_leg[o](α[o](i))))
          found = true
          break
        end
      end
      found || break
    end
    length(ccs) == sum(length.(values(repr_loc))) && return RepresentableP(repr_loc)
  end

  # Determine if pattern is discrete
  # all(ob(S)) do o 
  #   nparts(p, o) == 0  || isempty(homs(S, from=o))
  # end && return DiscreteP(Dict(o => nparts(p, o) for o in ob(S)))

  return RegularP() # no special case found
end

# Hazard rates depend on pattern type

get_hazard(::PatternType, m::ACSetTransformation, t::Float64, h::FullClosure; kw...) = h(m, t)

get_hazard(::PatternType, ::ACSetTransformation, t::Float64, h::ClosureTime; kw...) = h(t)

get_hazard(::PatternType, m::ACSetTransformation, ::Float64, h::ClosureState; kw...) = h(m)

get_hazard(::PatternType, ::ACSetTransformation, ::Float64, h::AbsHazard; kw...) = h.val

get_hazard(::PatternType, m::ACSetTransformation, ::Float64, h::ClosureParams; params=nothing, kw...) = 
  h(m, params)

get_hazard(::PatternType, m::ACSetTransformation, t::Float64, h::FullClosureParams; params=nothing, kw...) = 
  h(m, t, params)

get_hazard(::PatternType, m::ACSetTransformation, t::Float64, h::ClosureHistory; traj=nothing, kw...) = 
  h(m, t, traj)

function get_hazard(r::RepresentableP, f::ACSetTransformation, ::Float64, 
                    h::ContinuousHazard; kw...) 
   err = "Representable patterns must have simple exponential rules"
   X = codom(f)
   is_exp(h) ? Exponential(h.val.θ/multiplier(r,X)) : error(err)
end

const Maybe{T} = Union{Nothing, T}

"""
A stochastic rewrite rule with a dependent hazard rate

A basis is a subobject of the pattern of the rule for which we want a timer 
per match. By default, the basis ↣ pattern map is just id(pattern).

A context is an optional morphism `L ↪ Ctx` embedding the pattern in a larger 
context. When provided, state-dependent hazard rates receive a match morphism
`Ctx → X` (the extended context match) instead of just `L → X`. This lets 
hazard rates access neighborhood information beyond the pattern itself.

A dependency is an optional morphism `Dep ↪ Ctx` identifying the subset of 
the context that the hazard rate actually depends on.

A fix is an optional morphism `L_fix ↪ L` identifying the identity-constitutive
parts of a match. Two matches are considered "the same" if they agree on `fix`.
"""
@struct_hash_equal struct ABMRule
  rule::Rule
  timer::AbsTimer
  basis::Maybe{ACSetTransformation}
  name::Maybe{Symbol}
  pattern_type::PatternType
  context::Maybe{ACSetTransformation}     # L ↪ Ctx
  dependency::Maybe{ACSetTransformation}  # Dep ↪ Ctx
  fix::Maybe{ACSetTransformation}         # L_fix ↪ L
  ABMRule(r::Rule, t::AbsTimer; basis=nothing, name=nothing,
          context=nothing, dependency=nothing, fix=nothing) = 
    new(r, t, basis, name, pattern_type(r, is_exp(t)), context, dependency, fix)
end

# Give name as first arg rather than as kwarg
ABMRule(name::Maybe{Symbol}, r::Rule, t::AbsTimer; kw...) = 
  ABMRule(r, t; name, kw...)

getrule(r::ABMRule) = r.rule

Base.nameof(r::ABMRule) = r.name

pattern_type(r::ABMRule) = r.pattern_type

pattern(r::ABMRule) = pattern(getrule(r))

left(r::ABMRule) = left(getrule(r))
right(r::ABMRule) = right(getrule(r))

ruletype(r::ABMRule) = ruletype(getrule(r))

basis(r::ABMRule) = r.basis
context(r::ABMRule) = r.context
dependency(r::ABMRule) = r.dependency
fix(r::ABMRule) = r.fix

basis_pattern(r::ABMRule) = isnothing(r.basis) ? codom(left(r)) : dom(basis(r))

get_matches(r::ABMRule, args...; kw...) = 
  get_matches(getrule(r), args...; kw...)

(F::Migrate)(r::ABMRule) = 
  ABMRule(F(r.rule), r.timer; basis=F(r.basis), name=r.name,
          context=isnothing(r.context) ? nothing : F(r.context),
          dependency=isnothing(r.dependency) ? nothing : F(r.dependency),
          fix=isnothing(r.fix) ? nothing : F(r.fix))

"""
    resolve_match(rule::ABMRule, m::ACSetTransformation)

Given a match `m : L → X`, extend it through the context and/or dependency
morphisms to produce the match that the hazard rate function will receive.
"""
function resolve_match(rule::ABMRule, m::ACSetTransformation)
  ctx = context(rule)
  isnothing(ctx) && return m
  ctx_match = extend_morphism_constraints(m, ctx)
  dep = dependency(rule)
  extensions = homomorphisms(codom(ctx), codom(m); initial=ctx_match)
  isempty(extensions) && return m
  if isnothing(dep)
    length(extensions) == 1 && return only(extensions)
    rname = isnothing(rule.name) ? "<unnamed>" : string(rule.name)
    error("Ambiguous context extension for rule '$rname': found $(length(extensions)) valid context matches")
  end
  dep_extensions = map(ext -> dep ⋅ ext, extensions)
  first_dep = first(dep_extensions)
  all(ext -> ext == first_dep, dep_extensions) && return first_dep
  rname = isnothing(rule.name) ? "<unnamed>" : string(rule.name)
  error("Ambiguous dependency-restricted context extension for rule '$rname': found $(length(dep_extensions)) distinct dependency matches")
end

"""
    match_equal(rule::ABMRule, m1::ACSetTransformation, m2::ACSetTransformation)

Compare two matches for identity according to the rule's `fix` subobject.
If `fix` is set, compares `fix ⋅ m1 == fix ⋅ m2`. Otherwise `m1 == m2`.
"""
function match_equal(rule::ABMRule, m1::ACSetTransformation, m2::ACSetTransformation)
  f = fix(rule)
  isnothing(f) && return m1 == m2
  return (f ⋅ m1) == (f ⋅ m2)
end

"""
An ABM event driven by an AlgebraicRewriting Schedule rather than a single rule.

When the timer fires, the entire schedule is executed via `interpret!`. 
After execution, all hom-sets are rebuilt from scratch since the schedule 
may perform arbitrary sequences of rewrites.

Requires the initial state ACSet to use `MarkAsDeleted` part type.
"""
struct ABMSchedule
  name::Maybe{Symbol}
  schedule::Schedule
  timer::AbsTimer
end

ABMSchedule(schedule::Schedule, timer::AbsTimer) = ABMSchedule(nothing, schedule, timer)

Base.nameof(s::ABMSchedule) = s.name

"""
A type which implements AbsDynamics must be able to compiled to an ODE for some 
set of variables.
"""
abstract type AbsDynamics end 

"""Use raw Julia functions to define an ODE"""
@struct_hash_equal struct RawODE <: AbsDynamics 
  dynam::Vector{Function}
end

""" Continuous dynamics """
@struct_hash_equal struct ABMFlow 
  pat::ACSet
  dyn::AbsDynamics
  name::Maybe{Symbol}
  acs::Vector{Condition} # application conditions
  mapping::Vector{Pair{Symbol, Int}} # pair pat's variables w/ dyn quantities
end 

# Schema inference and validation
#################################

"""Extract the schema from an ABMRule's pattern (the codomain of left(rule))."""
rule_schema(r::ABMRule) = acset_schema(codom(left(getrule(r))))

"""Extract the schema from an ABMFlow's pattern."""
flow_schema(f::ABMFlow) = acset_schema(f.pat)

"""
    merge_schemas(schemas::Vector{BasicSchema{Symbol}})

Merge multiple schemas by identifying objects, homs, attrs, and attrtypes by name.
Conflicting hom/attr signatures (same name, different dom/codom) raise an error.
"""
function merge_schemas(schemas::Vector{<:Catlab.BasicSchema{Symbol}})
  isempty(schemas) && error("Cannot merge empty list of schemas")
  length(schemas) == 1 && return first(schemas)
  
  all_obs = Symbol[]
  all_homs = Tuple{Symbol,Symbol,Symbol}[]
  all_attrtypes = Symbol[]
  all_attrs = Tuple{Symbol,Symbol,Symbol}[]
  
  hom_sigs = Dict{Symbol, Tuple{Symbol,Symbol}}()
  attr_sigs = Dict{Symbol, Tuple{Symbol,Symbol}}()
  
  for S in schemas
    for o in objects(S)
      o ∉ all_obs && push!(all_obs, o)
    end
    for at in attrtypes(S)
      at ∉ all_attrtypes && push!(all_attrtypes, at)
    end
    for h in homs(S)
      name, d, c = h
      if haskey(hom_sigs, name)
        prev = hom_sigs[name]
        prev == (d, c) || error(
          "Schema conflict: hom '$name' has signature ($d→$c) in one rule " *
          "but ($(prev[1])→$(prev[2])) in another")
      else
        hom_sigs[name] = (d, c)
        push!(all_homs, h)
      end
    end
    for a in attrs(S)
      name, d, c = a
      if haskey(attr_sigs, name)
        prev = attr_sigs[name]
        prev == (d, c) || error(
          "Schema conflict: attr '$name' has signature ($d→$c) in one rule " *
          "but ($(prev[1])→$(prev[2])) in another")
      else
        attr_sigs[name] = (d, c)
        push!(all_attrs, a)
      end
    end
  end
  
  Catlab.BasicSchema{Symbol}(all_obs, all_homs, all_attrtypes, all_attrs, 
    Tuple{Union{Nothing, Symbol}, Symbol, Symbol, Tuple{Tuple{Vararg{Symbol}}, Tuple{Vararg{Symbol}}}}[])
end

"""
    infer_schema(rules::Vector{ABMRule}; dyn=[])

Infer the combined schema from a collection of ABM rules (and optional flows)
by merging (taking the colimit of) all rule/flow pattern schemas.

Raises an error if any two rules have conflicting schema elements 
(same name but different signatures).
"""
function infer_schema(rules::Vector{ABMRule}; dyn::Vector{ABMFlow}=ABMFlow[])
  schemas = Catlab.BasicSchema{Symbol}[rule_schema(r) for r in rules]
  for f in dyn
    push!(schemas, flow_schema(f))
  end
  merge_schemas(schemas)
end

"""
    is_subschema(sub::BasicSchema{Symbol}, sup::BasicSchema{Symbol})

Check whether `sub` is a sub-schema of `sup`: every object, hom, attr, and 
attrtype in `sub` must be present (with matching signature) in `sup`.

Returns `(ok::Bool, reason::String)`.
"""
function is_subschema(sub::Catlab.BasicSchema{Symbol}, sup::Catlab.BasicSchema{Symbol})
  sup_obs = Set(objects(sup))
  for o in objects(sub)
    o ∈ sup_obs || return (false, "Object '$o' not in target schema")
  end
  sup_ats = Set(attrtypes(sup))
  for at in attrtypes(sub)
    at ∈ sup_ats || return (false, "AttrType '$at' not in target schema")
  end
  sup_homs = Set(homs(sup))
  for h in homs(sub)
    h ∈ sup_homs || return (false, "Hom '$(h[1])' ($(h[2])→$(h[3])) not in target schema")
  end
  sup_attrs = Set(attrs(sup))
  for a in attrs(sub)
    a ∈ sup_attrs || return (false, "Attr '$(a[1])' ($(a[2])→$(a[3])) not in target schema")
  end
  return (true, "")
end

"""
    validate_schema(rules::Vector{ABMRule}, schema::BasicSchema; dyn=[])

Validate that all rules (and optional flows) have patterns whose schemas 
are sub-schemas of the declared `schema`.

Raises an error if any rule's schema is not compatible.
"""
function validate_schema(rules::Vector{ABMRule}, schema::Catlab.BasicSchema{Symbol};
                         dyn::Vector{ABMFlow}=ABMFlow[])
  for (i, r) in enumerate(rules)
    rs = rule_schema(r)
    ok, reason = is_subschema(rs, schema)
    ok || error("Rule $(something(nameof(r), i)): $reason")
  end
  for (i, f) in enumerate(dyn)
    fs = flow_schema(f)
    ok, reason = is_subschema(fs, schema)
    ok || error("Flow $(something(f.name, i)): $reason")
  end
  return schema
end

# Accessing an IncHomSet
const KeyType = Union{Pair{Int, Int},        # connected comp. homset
                      Vector{Pair{Int,Int}}} # multi-component homset

"""
Policy for handling simultaneous events (ties).

- `TieBreak`: Execute events in arbitrary order; later events may be invalidated 
  by earlier ones. This is the default and matches previous behavior.
- `TieRandom`: Shuffle event order randomly before sequential execution.
  Useful when the arbitrary ordering has systematic bias.
- `TieError`: Error if more than one event fires simultaneously.
  Useful for models where ties should never occur.
"""
@enum TiePolicy TieBreak TieRandom TieError

"""
An agent-based model, optionally parameterized.

Optionally accepts keyword arguments:
- `schema`: validates rules against it if provided, infers from rules if omitted.
- `tiepolicy`: how to handle simultaneous events (TieBreak, TieRandom, TieError).
- `params`: a NamedTuple or Dict of exogenous parameters for ClosureParams/FullClosureParams timers.
"""
@struct_hash_equal struct ABM
  rules::Vector{ABMRule}
  dyn::Vector{ABMFlow}
  names::Dict{Symbol, Int}
  tiepolicy::TiePolicy
  schema::Maybe{Catlab.BasicSchema{Symbol}}
  params::Any  # NamedTuple, Dict, or nothing
  schedules::Vector{ABMSchedule}
  function ABM(rules, dyn=[]; tiepolicy::TiePolicy=TieBreak, schema=nothing, 
               params=nothing, schedules::Vector{ABMSchedule}=ABMSchedule[]) 
    names = Dict(n=>i for (i,n) in enumerate(nameof.(rules)) if !isnothing(n))
    for (j, s) in enumerate(schedules)
      n = nameof(s)
      !isnothing(n) && (names[n] = length(rules) + j)
    end
    rs = collect(ABMRule, rules)
    ds = collect(ABMFlow, dyn)
    s = if !isnothing(schema)
      validate_schema(rs, schema; dyn=ds)
    elseif !isempty(rs) || !isempty(ds)
      infer_schema(rs; dyn=ds)
    else
      nothing
    end
    new(rs, ds, names, tiepolicy, s, params, schedules)
  end
end

additions(abm::ABM) = right.(abm.rules)

(F::Migrate)(abm::ABM) = ABM(F.(abm.rules), abm.dyn; tiepolicy=abm.tiepolicy, params=abm.params, schedules=abm.schedules)

Base.getindex(abm::ABM, i::Int) = abm.rules[i]
Base.getindex(abm::ABM, n::Symbol) = abm.rules[abm.names[n]]

Base.filter(f, abm::ABM) = ABM(filter(f, abm.rules); tiepolicy=abm.tiepolicy, params=abm.params, schedules=abm.schedules)

function Base.push!(abm::ABM, r::ABMRule; overwrite=false)
  if haskey(abm.names, r.name)
    overwrite || error("The ABM already has a rule with this name, set overwrite=true to replace")
    abm.rules[abm.names[r.name]] = r
  else
    push!(abm.rules, r)
    abm.names[r.name] = length(abm.rules)
  end
  abm
end

Base.copy(abm::ABM) = ABM(copy(abm.rules); tiepolicy=abm.tiepolicy, params=abm.params, schedules=copy(abm.schedules))
Base.length(abm::ABM) = length(abm.rules)

"""A collection of timers associated at runtime w/ an ABMRule"""
abstract type AbsHomSet end

@struct_hash_equal struct EmptyHomSet <: AbsHomSet end

@struct_hash_equal struct RepresentableHomSet <: AbsHomSet end

@struct_hash_equal struct ExplicitHomSet <: AbsHomSet val::IncHomSet end

Base.keys(h::ExplicitHomSet) = keys(h.val)

Base.haskey(h::ExplicitHomSet, k::KeyType) = haskey(h.val, k)

Base.haskey(::EmptyHomSet, k) = false

Base.haskey(::RepresentableHomSet, k) = false

Base.pairs(h::ExplicitHomSet) = pairs(h.val)

Base.getindex(h::ExplicitHomSet, i) = h.val[i]

deletion!(h::ExplicitHomSet, m; kw...) =  deletion!(h.val, m; kw...)

addition!(h::ExplicitHomSet, k, r, u) = addition!(h.val, k, r, u)

"""Initialize runtime hom-set given the rule and the initial state"""
function init_homset(rule::ABMRule, state::ACSet, 
                     additions::Vector{<:ACSetTransformation})
  p, sd = pattern_type(rule), state_dep(rule.timer)
  p == EmptyP() && return EmptyHomSet()
  (sd || p == RegularP()  
   ) && return ExplicitHomSet(IncHomSet_basis(getrule(rule), state,  additions; 
                                        basis=basis_pattern(rule)))
  @assert p isa RepresentableP  "$(typeof(p))"
  return RepresentableHomSet()
end 

const default_sampler = FirstToFire{
  Union{Pair{Int, Nothing},   # non-explicit homset
        Pair{Int, Pair{Int,Int}}, # explicit single cc homset
        Pair{Int, Vector{Pair{Int,Int}}}},  # explicit mc homset
  Float64}

"""
Data structure for maintaining simulation information while running an ABM
"""
mutable struct RuntimeABM
  state::ACSet
  const clocks::Vector{AbsHomSet}
  tnow::Float64
  nevent::Int
  const sampler::SSA # stochastic simulation algorithm
  const rng::Distributions.AbstractRNG
  const names::Dict{Symbol, Int}
  prob::ODEProblem
  probmap::Vector{Pair{Symbol, Int}}
  probdict::Dict{Symbol, Dict{Tuple{Symbol, Int}, Int}}

  function RuntimeABM(abm::ABM, init::T; sampler=default_sampler) where T<:ACSet
    # Create the runtime
    names = Dict(r => i for (i, r) in enumerate(nameof.(abm.rules))
                 if !isnothing(r))
    rt = new(init, init_homset.(abm.rules, Ref(init), Ref(additions(abm))), 
             0., 0, sampler(), Random.RandomDevice(), names, 
             mk_prob(abm, init)...)
    # Initialize the firing queue
    for (i, (pat,homset)) in enumerate(zip(pattern_type.(abm.rules), rt.clocks))
      kv = if homset isa ExplicitHomSet 
        pairs(homset) 
      else
        if pat isa EmptyP || all(>(0), nparts.(Ref(init), keys(pat)))
          [nothing => create(init)]
        else 
          []
        end
      end
      for (key, val) in kv
        val_resolved = resolve_match(abm.rules[i], val)
        haz = get_hazard(pat, val_resolved, 0., abm.rules[i].timer; params=abm.params)
        enable!(rt.sampler, i => key, haz, 0., 0., rt.rng)
      end
    end
    # Initialize schedule timers (indices > nrules)
    nrules = length(abm.rules)
    for (j, sched) in enumerate(abm.schedules)
      sched_idx = nrules + j
      haz = if sched.timer isa AbsHazard
        sched.timer.val
      elseif sched.timer isa ClosureTime
        sched.timer(0.)
      else
        error("ABMSchedule timers must not depend on match state")
      end
      enable!(rt.sampler, sched_idx => nothing, haz, 0., 0., rt.rng)
    end
    return rt
  end
end

state(r::RuntimeABM) = r.state

Base.haskey(rt::RuntimeABM, k::Pair) = haskey(rt.sampler.transition_entry, k)

Base.haskey(rt::RuntimeABM, k::Int) = 
  haskey(rt.sampler.transition_entry, k => nothing)

Base.getindex(rt::RuntimeABM, i::Int) = rt.clocks[i]
Base.getindex(rt::RuntimeABM, n::Symbol) = rt.clocks[rt.names[n]]

"""
    refresh_clocks!(rt::RuntimeABM, abm::ABM)

Rebuild the incremental hom-sets and re-initialize all clocks in the sampler
after external modification of `rt.state`. Call this after manually changing the
state between segmented `run!` calls.

# Example
```julia
rt = RuntimeABM(abm, init)
traj = run!(abm, rt, Traj(init); maxtime=10)
# Manually modify state
add_parts!(rt.state, :V, 5)
refresh_clocks!(rt, abm)
traj = run!(abm, rt, traj; maxtime=20)
```
"""
function refresh_clocks!(rt::RuntimeABM, abm::ABM)
  for i in eachindex(abm.rules)
    rt.clocks[i] = init_homset(abm.rules[i], rt.state, additions(abm))
  end
  for (key, _) in collect(rt.sampler.transition_entry)
    disable!(rt.sampler, key, rt.tnow)
  end
  for (i, (pat, homset)) in enumerate(zip(pattern_type.(abm.rules), rt.clocks))
    kv = if homset isa ExplicitHomSet
      pairs(homset)
    else
      if pat isa EmptyP || all(>(0), nparts.(Ref(rt.state), keys(pat)))
        [nothing => create(rt.state)]
      else
        []
      end
    end
    for (key, val) in kv
      val_resolved = resolve_match(abm.rules[i], val)
      haz = get_hazard(pat, val_resolved, rt.tnow, abm.rules[i].timer; params=abm.params)
      enable!(rt.sampler, i => key, haz, rt.tnow, rt.tnow, rt.rng)
    end
  end
  nrules = length(abm.rules)
  for (j, sched) in enumerate(abm.schedules)
    sched_idx = nrules + j
    haz = if sched.timer isa AbsHazard
      sched.timer.val
    elseif sched.timer isa ClosureTime
      sched.timer(rt.tnow)
    else
      error("ABMSchedule timers must not depend on match state")
    end
    enable!(rt.sampler, sched_idx => nothing, haz, rt.tnow, rt.tnow, rt.rng)
  end
  return rt
end

"""
    Intervention(time, action)
    Intervention(predicate, action; name=nothing)

A scheduled or conditional intervention applied during simulation.

- **Scheduled**: `Intervention(5.0, state -> add_parts!(state, :V, 10))`
  fires at t=5.0.
- **Conditional**: `Intervention(state -> nparts(state, :V) > 100, state -> ...)`  
  fires when the predicate becomes true (checked after each event).
"""
struct Intervention
  time::Maybe{Float64}
  predicate::Maybe{Function}
  action::Function
  name::Maybe{Symbol}
end

Intervention(time::Real, action::Function; name=nothing) = 
  Intervention(Float64(time), nothing, action, name)
Intervention(predicate::Function, action::Function; name=nothing) = 
  Intervention(nothing, predicate, action, name)

is_scheduled(iv::Intervention) = !isnothing(iv.time)
is_conditional(iv::Intervention) = !isnothing(iv.predicate)

"""
Construct an ODE for a given ACSet state. For each flow, find all matches of the 
flow's pattern in the state. Each match contributes ODE variables corresponding 
to the flow's `mapping`. The returned `probmap` tracks which (attribute, part_index) 
each ODE variable corresponds to, and `probdict` provides reverse lookup by
attribute type and part.
"""
function mk_prob(abm::ABM, state::ACSet)
  isempty(abm.dyn) && return (ODEProblem((_,_,_,_)->0, 0, (0.,1.)), [], Dict())
  
  probmap = Pair{Symbol, Int}[]       # ODE index → (attr_name, part_index)
  probdict = Dict{Symbol, Dict{Tuple{Symbol, Int}, Int}}()  # attr_type → (attr_name, part_index) → ODE index
  dynam_fns = Function[]              # dynamics function for each ODE variable

  S = acset_schema(state)
  # Build lookup: attr_type_symbol → [(attr_name, object_name)]
  attr_lookup = Dict{Symbol, Vector{Tuple{Symbol, Symbol}}}()
  for (aname, ob, atype) in attrs(S)
    push!(get!(attr_lookup, atype, Tuple{Symbol,Symbol}[]), (aname, ob))
  end

  for flow in abm.dyn
    matches = homomorphisms(flow.pat, state)
    for m in matches
      for (map_idx, (attr_sym, pat_idx)) in enumerate(flow.mapping)
        haskey(attr_lookup, attr_sym) || continue
        for (aname, ob) in attr_lookup[attr_sym]
          found = false
          for p in parts(flow.pat, ob)
            val = flow.pat[p, aname]
            if val isa AttrVar && val.val == pat_idx
              state_part = m[ob](p)
              if !haskey(probdict, attr_sym)
                probdict[attr_sym] = Dict{Tuple{Symbol, Int}, Int}()
              end
              lookup_key = (aname, state_part)
              if !haskey(probdict[attr_sym], lookup_key)
                push!(probmap, aname => state_part)
                probdict[attr_sym][lookup_key] = length(probmap)
                push!(dynam_fns, flow.dyn.dynam[map_idx])
              end
              found = true
              break
            end
          end
          found && break
        end
      end
    end
  end
  
  isempty(probmap) && return (ODEProblem((_,_,_,_)->0, 0, (0.,1.)), probmap, probdict)
  
  # Build initial condition from current state attribute values
  u0 = Float64[]
  for (attr_name, state_part) in probmap
    push!(u0, Float64(state[state_part, attr_name]))
  end
  
  # Build the ODE function
  fns = copy(dynam_fns)
  function ode_f!(du, u, p, t)
    for i in eachindex(du)
      du[i] = fns[i](u[i])
    end
  end
  
  prob = ODEProblem(ode_f!, u0, (0., Inf))
  return (prob, probmap, probdict)
end

"""
Write ODE solution values back into the ACSet state attributes.
"""
function write_ode_to_state!(state::ACSet, u::AbstractVector, 
                              probmap::Vector{Pair{Symbol, Int}})
  for (i, (attr_name, state_part)) in enumerate(probmap)
    set_subpart!(state, state_part, attr_name, u[i])
  end
end

"""
Rebuild the ODE problem after a rewrite event changes the state.
Returns updated (prob, probmap, probdict).
"""
function remake_prob(abm::ABM, state::ACSet, tnow::Float64)
  (prob, probmap, probdict) = mk_prob(abm, state)
  # Remap tspan to start from current time
  if prob.u0 isa Number && prob.u0 == 0
    return (prob, probmap, probdict)
  end
  prob = ODEProblem(prob.f, prob.u0, (tnow, Inf))
  return (prob, probmap, probdict)
end

"""
Check that RuntimeABM incremental hom sets have all valid homs.
"""
function validate(rt::RuntimeABM)
  for c in filter(c -> c isa IncHomSet, rt.clocks)
    c.state == rt.state || error("State mismatch")
    validate(c)
  end
end

"""Pop the next random event, advance the clock"""
function pops!(rt::RuntimeABM)::Vector{Pair{Int, Maybe{KeyType}}}
  rt.nevent += 1
  (rt.tnow, which) = pops!(rt.sampler, rt.rng, rt.tnow)
  return which
end


function get_match(pat::PatternType, L::ACSet, G::ACSet, timer::AbsHomSet, key; 
                   basis::Maybe{ACSetTransformation}) 
  isnothing(basis) && return get_match(pat, L, G, timer, key)
  m = get_match(pat, dom(basis), G, timer, key)
  initial = extend_morphism_constraints(m, basis)
  rand(homomorphisms(L, G; initial))
end

"""
Get match returns a randomly chosen morphism for the aggregate rule
"""
get_match(::EmptyP, L::ACSet, G::ACSet, ::EmptyHomSet, ::Nothing) = create(G)

function get_match(P::RepresentableP, L::T, G::ACSet, ::RepresentableHomSet, 
                   ::Nothing) where T<:ACSet
  initial = Dict(map(collect(pairs(P.parts))) do (o, idxs) 
    o => Dict(idx => rand(parts(G, o)) for idx in idxs)
  end)
  return homomorphism(L, G; initial)
end

get_match(::RegularP, ::ACSet, ::ACSet, hs::ExplicitHomSet, key::KeyType) = hs[key]

"""
An observable quantity: a pattern whose match count is tracked over time.

    Observable(name::Symbol, pattern::ACSet)

Creates an observable that counts the number of homomorphisms from `pattern` 
into the current state at each logging point.

# Example
```julia
obs_V = Observable(:vertices, @acset Graph begin V=1 end)
obs_E = Observable(:edges, @acset Graph begin V=2; E=1; src=1; tgt=2 end)
traj = run!(abm, init; observables=[obs_V, obs_E], save_every=1.0)
traj.snapshots  # Vector of (time, Dict(:vertices => count, :edges => count))
```
"""
struct Observable
  name::Symbol
  pattern::ACSet
end

"""Evaluate an observable by counting homomorphisms from pattern into state."""
(obs::Observable)(state::ACSet) = length(homomorphisms(obs.pattern, state))

"""
Refresh a match morphism's attribute bindings after ODE integration has changed 
attribute values in the state. Keeps the same combinatorial mapping, recomputes
the attribute component to match the current state.
"""
function refresh_match(m::ACSetTransformation, state::ACSet)
  S = acset_schema(dom(m))
  initial = NamedTuple(Dict(o => collect(m[o]) for o in ob(S)))
  homomorphism(dom(m), state; initial)
end
"""
A trajectory of an ABM: each event time and result of `save`.

# Fields
- `init::ACSet` — initial state
- `events` — vector of `(time, rule_id, rule_name, saved_data)` tuples
- `hist` — vector of rewrite spans (empty when `record_history=false`)
- `snapshots` — vector of `(time, Dict{Symbol,Int})` from observable evaluations
"""
@struct_hash_equal struct Traj
  init::ACSet
  events::Vector{Tuple{Float64, Int, String, Any}}
  hist::Vector{Span{<:ACSet}}
  snapshots::Vector{Tuple{Float64, Dict{Symbol, Int}}}
end

Traj(x::ACSet) = Traj(x, Tuple{Float64, Int, String, Any}[], Span{ACSet}[],
                       Tuple{Float64, Dict{Symbol, Int}}[])

function Base.push!(t::Traj, tup::Tuple{Float64,Int,String,Any,Span{<:ACSet}}) 
  (τ, rule, rulename, v, sp) = tup
  push!(t.events, (τ, rule, rulename, v))
  isempty(t.hist) || codom(left(sp)) == codom(right(last(t.hist))) || error(
    "Bad history \n$(codom(left(sp))) \n!= \n$(codom(right(last(t.hist))))"
  )
  push!(t.hist, sp)
end

function Base.push!(t::Traj, tup::Tuple{Float64,Int,String,Any,Nothing}) 
  (τ, rule, rulename, v, _) = tup
  push!(t.events, (τ, rule, rulename, v))
end

Base.isempty(t::Traj) = isempty(t.events)

Base.length(t::Traj) = length(t.events)

const MAXEVENT = 100

"""
Run an ABM, creating a fresh runtime + trajectory.

# Keyword arguments
- `save` — function applied to the ACSet state to produce data stored per event
- `maxevent` — maximum number of events before stopping (default: $MAXEVENT)
- `maxtime` — maximum simulation time (default: Inf)
- `observables` — vector of `Observable`s to evaluate at logging points
- `save_every` — if set, record observables at regular time intervals
- `record_history` — if `true`, store rewrite spans in trajectory (default: `true`)
- `dt` — timestep for checking discrete events when running ODE dynamics
- `interventions` — vector of `Intervention`s applied during simulation
"""
function run!(abm::ABM, init::T; save=_->nothing, maxevent=MAXEVENT, 
              maxtime=Inf, dt=0.1, observables::Vector{Observable}=Observable[],
              save_every::Maybe{Float64}=nothing, record_history::Bool=true,
              interventions::Vector{Intervention}=Intervention[],
              kw...) where T<:ACSet 
  run!(abm::ABM, RuntimeABM(abm, init; kw...), Traj(init); 
       save, maxtime, maxevent, dt, observables, save_every, record_history,
       interventions)
end

function run!(abm::ABM, rt::RuntimeABM, output::Traj;
              save=_->nothing, maxevent=MAXEVENT, maxtime=Inf, dt=0.1,
              observables::Vector{Observable}=Observable[],
              save_every::Maybe{Float64}=nothing, record_history::Bool=true,
              interventions::Vector{Intervention}=Intervention[])
  next_snapshot_time = isnothing(save_every) ? Inf : save_every
  has_observables = !isempty(observables)

  # Record initial snapshot if we have observables
  if has_observables
    snap = Dict(obs.name => obs(rt.state) for obs in observables)
    push!(output.snapshots, (rt.tnow, snap))
  end

  # Helper functions that automatically incorporate the runtime `rt`
  getname(rule::Int)::String = 
    string(isnothing(abm.rules[rule].name) ? rule : abm.rules[rule].name)
  function log!(rule::Int, sp::Span)
    logged_sp = record_history ? sp : nothing
    push!(output, (rt.tnow, rule, getname(rule), save(rt.state), logged_sp))
  end
  disable!′(key::Pair) = disable!(rt.sampler, key, rt.tnow)
  disable!′(i::Int) = disable!′(i => nothing)
  function enable!′(m::ACSetTransformation, rule_id::Int, key::Maybe{KeyType}=nothing) 
    rule = abm.rules[rule_id]
    m_resolved = resolve_match(rule, m)
    haz = get_hazard(pattern_type(rule), m_resolved, rt.tnow, rule.timer; params=abm.params, traj=output)
    enable!(rt.sampler, rule_id => key, haz, rt.tnow, rt.tnow, rt.rng)
  end

  # Track which interventions have fired (for scheduled ones)
  fired_interventions = Set{Int}()

  # Main loop
  while rt.nevent < maxevent && rt.tnow < maxtime
    # TODO: isempty(abm.dyn) should be check that all flows sum to 0 
    if length(rt.sampler) == 0 && isempty(abm.dyn)
      has_pending = any(enumerate(interventions)) do (i, iv)
        is_scheduled(iv) && i ∉ fired_interventions && iv.time <= maxtime
      end
      if !has_pending
        @info "Stochastic scheduling algorithm ran out of events"
        return output
      end
    end

    # Determine next stochastic event time (Inf if no stochastic events)
    new_time = length(rt.sampler) > 0 ? first(next(rt.sampler, rt.tnow, rt.rng)) : Inf

    # Check for scheduled interventions that fire before the next stochastic event
    intervention_fired = false
    for (i, iv) in enumerate(interventions)
      is_scheduled(iv) || continue
      i ∈ fired_interventions && continue
      if iv.time <= min(new_time, maxtime) && iv.time >= rt.tnow
        rt.tnow = iv.time
        iv.action(rt.state)
        push!(fired_interventions, i)
        iname = isnothing(iv.name) ? "intervention_$i" : string(iv.name)
        @debug "Intervention '$iname' applied at t=$(rt.tnow)"
        refresh_clocks!(rt, abm)
        intervention_fired = true
        break
      end
    end
    intervention_fired && continue

    if !isempty(abm.dyn) && rt.tnow + dt < new_time 
      # ODE integration branch: integrate continuous dynamics in steps of dt
      # until the next stochastic event fires (or maxtime is reached).
      # We keep ODE values in rt.prob.u0 and only write to state at the end,
      # to avoid invalidating stored match morphisms during integration.
      target_time = min(new_time, maxtime)
      while rt.tnow + dt < target_time && rt.tnow < maxtime
        t_start = rt.tnow
        t_end = min(rt.tnow + dt, target_time)
        if rt.prob.u0 isa AbstractVector && !isempty(rt.prob.u0)
          local sol = solve(
            ODEProblem(rt.prob.f, rt.prob.u0, (t_start, t_end)),
            Tsit5(); save_everystep=false
          )
          rt.prob = ODEProblem(rt.prob.f, sol.u[end], (t_end, Inf))
        end
        rt.tnow = t_end

        # Check if a stochastic event now fires sooner
        if length(rt.sampler) > 0
          new_time = first(next(rt.sampler, rt.tnow, rt.rng))
          new_time <= rt.tnow + dt && break
          target_time = min(new_time, maxtime)
        end
      end
      # Integrate the final segment up to the event time (or maxtime)
      t_final = min(new_time, maxtime)
      if rt.prob.u0 isa AbstractVector && !isempty(rt.prob.u0) && t_final - rt.tnow > 1e-12
        local sol = solve(
          ODEProblem(rt.prob.f, rt.prob.u0, (rt.tnow, t_final)),
          Tsit5(); save_everystep=false
        )
        rt.prob = ODEProblem(rt.prob.f, sol.u[end], (t_final, Inf))
      end
      rt.tnow = t_final
      # Write ODE values back to state
      if rt.prob.u0 isa AbstractVector && !isempty(rt.prob.u0)
        write_ode_to_state!(rt.state, rt.prob.u0, rt.probmap)
      end
      # If there are no stochastic events, just continue the loop
      length(rt.sampler) == 0 && continue
    else
      # Get next event + unpack data
      events::Vector{Pair{Int,Maybe{KeyType}}} = pops!(rt) # updates the clock time
      N = length(rt.sampler)

      nrules = length(abm.rules)
      s = length(events) > 1 ? "s" : ""
      function rname(e) 
        r = first(e)
        if r <= nrules
          n = abm.rules[r].name; isnothing(n) ? r : n
        else
          n = abm.schedules[r - nrules].name; isnothing(n) ? "sched_$(r-nrules)" : n
        end
      end
      @debug ("Step $(length(output)): Event$s $(join(string.(rname.(events)), ", "))"
              *" | Fired @ t = $(round(rt.tnow, digits=2)) ($N queued)")

      # TODO some sort of check that the events are consistent with each other
      # or a randomization of their order
      if length(events) > 1
        if abm.tiepolicy == TieError
          error("TieError policy: $(length(events)) simultaneous events at t=$(rt.tnow)")
        elseif abm.tiepolicy == TieRandom
          shuffle!(rt.rng, events)
        end
        # TieBreak: keep arbitrary order (default)
      end

      # Separate schedule events from rule events
      schedule_events = filter(e -> first(e) > nrules, events)
      rule_events = filter(e -> first(e) <= nrules, events)
      
      # Execute schedule events (full state replacement + refresh)
      schedule_fired = false
      for (event, _key) in schedule_events
        sched = abm.schedules[event - nrules]
        sname = something(nameof(sched), "schedule_$(event - nrules)")
        @debug "Executing schedule '$sname' at t=$(rt.tnow)"
        result = interpret!(sched.schedule, rt.state)
        rt.state = codom(result)
        schedule_fired = true
        push!(output, (rt.tnow, event, string(sname), save(rt.state), 
              Span(id(rt.state), id(rt.state))))
        # Re-enable schedule timer
        shaz = if sched.timer isa AbsHazard
          sched.timer.val
        elseif sched.timer isa ClosureTime
          sched.timer(rt.tnow)
        else
          error("ABMSchedule timers must not depend on match state")
        end
        enable!(rt.sampler, event => nothing, shaz, rt.tnow, rt.tnow, rt.rng)
      end
      
      # If any schedule fired, rebuild all rule hom-sets
      if schedule_fired
        for (i, (ruleᵢ, clocksᵢ)) in enumerate(zip(abm.rules, rt.clocks))
          clocksᵢ isa ExplicitHomSet || continue
          for k in collect(keys(clocksᵢ))
            disable!(rt.sampler, i => k, rt.tnow)
          end
        end
        rt.clocks .= init_homset.(abm.rules, Ref(rt.state), Ref(additions(abm)))
        for (i, (pat, homset)) in enumerate(zip(pattern_type.(abm.rules), rt.clocks))
          homset isa ExplicitHomSet || continue
          for (key, val) in pairs(homset)
            val_resolved = resolve_match(abm.rules[i], val)
            haz = get_hazard(pat, val_resolved, rt.tnow, abm.rules[i].timer; params=abm.params)
            enable!(rt.sampler, i => key, haz, rt.tnow, rt.tnow, rt.rng)
          end
        end
        isempty(rule_events) && continue
      end

      update_data = [] # use to update incremental hom sets afterwards
      # execute all the rule events
      for (event, key) in rule_events
        rule::ABMRule, clocks::AbsHomSet = abm.rules[event], rt.clocks[event]
        rule′::Rule, rule_type::Symbol = getrule(rule), ruletype(rule)
        # If RegularPattern, we have an explicit match, otherwise randomly pick one
        m = get_match(pattern_type(rule), pattern(rule), rt.state, clocks, key; 
                      basis=basis(rule))
        # Refresh attribute bindings if ODE integration changed attribute values
        if !isempty(abm.dyn)
          m = refresh_match(m, rt.state)
        end
        # bring the match 'up to speed' given the previous (simultanous) updates
        for (l, r) in first.(update_data)
          pb = pull_back(l, m)
          if isnothing(pb)
            @debug "Skipping event $(name(rule)): match invalidated by prior simultaneous event"
            m = nothing
            break
          end
          m = pb ⋅ r
        end
        isnothing(m) && continue
        dpo = rule_type == :DPO ? (left(rule′), m) : nothing
        # check if dangling condition is satisfied
        isnothing(dpo) || can_pushout_complement(ComposablePair(dpo...)) || continue
        # Excute rewrite rule and unpack results
        rw_result = (rule_type, rewrite_match_maps(rule′, m))
        rmap_ = get_rmap(rw_result...)
        xmap = get_expr_binding_map(rule′, m, rw_result[2])
        (lft, rght_) = get_pmap(rw_result...)
        rmap, rght = compose.([rmap_,rght_], Ref(xmap))
        pmap = Span(lft, rght)
        rt.state = codom(rmap) # update runtime state
        log!(event, pmap)      # record event result
        push!(update_data, (pmap, rmap, dpo, right(rule′)))
      end
      
      # if no event at this time was actionable, due to dangling condition
      isempty(update_data) && continue 

      # Track which (rule, key) pairs are enabled during the update phase,
      # so we can avoid duplicate re-enabling of fired matches at the end.
      updated_keys = Set{Pair{Int, Maybe{KeyType}}}()

      # All other rules can potentially update in response to the current event
      for (i, (ruleᵢ, clocksᵢ)) in enumerate(zip(abm.rules, rt.clocks))
        pt = pattern_type(ruleᵢ)
        if pt == EmptyP() && i ∈ first.(events)
          enable!′(create(rt.state), i)
        elseif pt == RegularP() # update explicit hom-set w/r/t span Xₙ ↩ • -> Xₙ₊₁
          for ((lft, rght), rmap, dpo, rule_right) in update_data
            del_invalid, del_new = deletion!(clocksᵢ, lft; dpo)

            for d in del_invalid # disable clocks which are invalidated
              (i=>d) ∈ events || disable!′(i => d) # (event,key) already diabled
            end

            for a in del_new
              enable!′(clocksᵢ[a], i, a) 
              push!(updated_keys, i => a)
            end
            add_invalid, add_new = addition!(clocksᵢ, rule_right, rmap, rght)

            for d in add_invalid # disable clocks which are invalidated
              (i=>d) ∈ (events) || disable!′(i => d) # (event,key) already diabled
            end
            for a in add_new
              # Check if this newly-discovered match duplicates one already
              # enabled (can occur when expression binding creates a new state
              # object that is isomorphic to the old one).
              new_match = clocksᵢ[a]
              is_dup = any(updated_keys) do rk
                first(rk) == i && haskey(clocksᵢ, last(rk)) &&
                  clocksᵢ[last(rk)] == new_match
              end
              if !is_dup
                enable!′(clocksᵢ[a], i, a)
              end
              push!(updated_keys, i => a)
            end
          end
        elseif pt isa RepresentableP
          relevant_obs = keys(pt)
          Xs = left(first(first(update_data))), right(first(last(update_data)))
          # we need to update current timer if # of parts has changed
          if i ∈ first.(events) && all(>(0), nparts.(Ref(rt.state), relevant_obs))
            enable!′(create(rt.state), i)
          elseif !all(ob -> allequal(nparts.(codom.(Xs), ob)), relevant_obs)
            currently_enabled = haskey(rt, i)
            currently_enabled && disable!′(i) # Disable if active
            # enable new timer if possible to apply rule
            if all(>(0), nparts.(Ref(rt.state), relevant_obs))
              enable!′(create(rt.state), i) 
            end
          end
        end
      end
      # If any of the matches that were fired are still preserved, re-enable,
      # but only if an equivalent match (per fix subobject) was not already enabled.
      for (event, key) in events
        if haskey(rt.clocks[event], key)
          match = rt.clocks[event][key]
          rule_ev = abm.rules[event]
          already_enabled = any(updated_keys) do rk
            first(rk) == event && haskey(rt.sampler.transition_entry, rk) &&
              haskey(rt.clocks[event], last(rk)) &&
              match_equal(rule_ev, rt.clocks[event][last(rk)], match)
          end
          if !already_enabled
            enable!′(match, event, key)
          end
        end
      end

      # Record observable snapshots at save_every intervals
      if has_observables
        while rt.tnow >= next_snapshot_time
          snap = Dict(obs.name => obs(rt.state) for obs in observables)
          push!(output.snapshots, (next_snapshot_time, snap))
          next_snapshot_time += save_every
        end
      end
      # Rebuild ODE problem if flows exist (state topology may have changed)
      if !isempty(abm.dyn)
        (rt.prob, rt.probmap, rt.probdict) = remake_prob(abm, rt.state, rt.tnow)
      end

      # Check conditional interventions after events execute
      for (i, iv) in enumerate(interventions)
        is_conditional(iv) || continue
        if iv.predicate(rt.state)
          iv.action(rt.state)
          iname = isnothing(iv.name) ? "cond_intervention_$i" : string(iv.name)
          @debug "Conditional intervention '$iname' triggered at t=$(rt.tnow)"
          refresh_clocks!(rt, abm)
        end
      end
    end
  end
  return output
end

"""
    run_scenarios(abm, init, scenarios; kw...)

Run an ABM under multiple parameter scenarios. Each scenario is a `NamedTuple` 
or `Dict{Symbol,Any}` that overrides `abm.params` for that run.

Returns a `Vector{Pair{<:Any, Traj}}` of (scenario, trajectory) pairs.
"""
function run_scenarios(abm::ABM, init::T, scenarios; kw...) where T<:ACSet
  map(scenarios) do scen
    abm_s = ABM(abm.rules, abm.dyn; tiepolicy=abm.tiepolicy, params=scen)
    traj = run!(abm_s, deepcopy(init); kw...)
    scen => traj
  end
end

# Networkification
##################

"""
    networkify(S::Presentation; loc_prefix=:loc_)

Extend schema `S` with graph structure, putting agents on a network.
Adds objects `V` (vertices) and `E` (edges), homs `src` and `tgt`, and
for each object `X` in `S`, a location hom `loc_prefix * X :: Hom(X, V)`.

Returns a new `Presentation`. Use with `@acset_type` or `AnonACSet` to
create instances.

# Example
```julia
@present SchSIR(FreeSchema) begin
  S::Ob; I::Ob; R::Ob
end
SchSIR_Net = networkify(SchSIR)
# Now has: S, I, R, V, E, src, tgt, loc_S, loc_I, loc_R
```
"""
function networkify(S::Presentation; loc_prefix::Symbol=:loc_)
  S_net = Presentation(FreeSchema)
  
  for g in generators(S, :Ob)
    add_generator!(S_net, g)
  end
  for g in generators(S, :AttrType)
    add_generator!(S_net, g)
  end
  
  add_generator!(S_net, Ob(FreeSchema, :V))
  add_generator!(S_net, Ob(FreeSchema, :E))
  
  for g in generators(S, :Hom)
    add_generator!(S_net, g)
  end
  
  add_generator!(S_net, Hom(:src, S_net[:E], S_net[:V]))
  add_generator!(S_net, Hom(:tgt, S_net[:E], S_net[:V]))
  
  for g in generators(S, :Attr)
    add_generator!(S_net, g)
  end
  
  for g in generators(S, :Ob)
    name = Symbol(loc_prefix, first(g))
    add_generator!(S_net, Hom(name, S_net[first(g)], S_net[:V]))
  end
  
  return S_net
end

"""
    networkify(S::BasicSchema{Symbol}; loc_prefix=:loc_)

Extend a `BasicSchema` with graph structure. Returns a new `BasicSchema`.
"""
function networkify(S::Catlab.BasicSchema{Symbol}; loc_prefix::Symbol=:loc_)
  obs = vcat(collect(objects(S)), [:V, :E])
  hom_list = vcat(
    collect(homs(S)),
    [(:src, :E, :V), (:tgt, :E, :V)],
    [Tuple{Symbol,Symbol,Symbol}((Symbol(loc_prefix, o), o, :V)) for o in objects(S)]
  )
  at_list = collect(attrtypes(S))
  attr_list = collect(attrs(S))
  Catlab.BasicSchema{Symbol}(obs, hom_list, at_list, attr_list,
    Tuple{Union{Nothing,Symbol},Symbol,Symbol,Tuple{Tuple{Vararg{Symbol}},Tuple{Vararg{Symbol}}}}[])
end

"""
    shortest_distance(state::ACSet, u::Int, v::Int; 
                      src_hom=:src, tgt_hom=:tgt, v_ob=:V)

Compute shortest path distance between vertices `u` and `v` in the graph
structure of an ACSet. Returns `typemax(Int)` if no path exists.

Treats edges as undirected (considers both src→tgt and tgt→src).
"""
function shortest_distance(state::ACSet, u::Int, v::Int;
                          src_hom::Symbol=:src, tgt_hom::Symbol=:tgt,
                          v_ob::Symbol=:V)
  u == v && return 0
  nv = nparts(state, v_ob)
  (u < 1 || u > nv || v < 1 || v > nv) && return typemax(Int)
  
  srcs = subpart(state, src_hom)
  tgts = subpart(state, tgt_hom)
  adj = [Int[] for _ in 1:nv]
  for (s, t) in zip(srcs, tgts)
    push!(adj[s], t)
    push!(adj[t], s)
  end
  
  dist = fill(typemax(Int), nv)
  dist[u] = 0
  queue = Int[u]
  while !isempty(queue)
    cur = popfirst!(queue)
    for nb in adj[cur]
      if dist[nb] == typemax(Int)
        dist[nb] = dist[cur] + 1
        nb == v && return dist[nb]
        push!(queue, nb)
      end
    end
  end
  return dist[v]
end

# Dimensional analysis stubs (methods added by UnitfulExt)
"""
    validate_units(rate, expected_dim)

Validate dimensional consistency. Methods added by UnitfulExt when Unitful is loaded.
"""
function validate_units end

"""
    strip_units(val)

Strip units from a quantity. Methods added by UnitfulExt when Unitful is loaded.
"""
function strip_units end

# Spatial utilities
###################

"""
    positions(state::ACSet, ob::Symbol, attrs::Vector{Symbol})

Extract position matrix from an ACSet. Returns a `D × N` matrix.
"""
function positions(state::ACSet, ob::Symbol, attrs::Vector{Symbol})
  n = nparts(state, ob)
  n == 0 && return Matrix{Float64}(undef, length(attrs), 0)
  hcat([Float64.(subpart(state, a)) for a in attrs]...)'
end

"""
    within_radius(state::ACSet, i::Int, r::Real, ob::Symbol, attrs::Vector{Symbol};
                  exclude_self=true)

Find all parts of type `ob` within Euclidean distance `r` of part `i`.
"""
function within_radius(state::ACSet, i::Int, r::Real, ob::Symbol, 
                       attrs::Vector{Symbol}; exclude_self::Bool=true)
  pos = positions(state, ob, attrs)
  n = size(pos, 2)
  (i < 1 || i > n) && return Int[]
  pi = pos[:, i]
  r2 = r * r
  result = Int[]
  for j in 1:n
    (exclude_self && j == i) && continue
    d2 = sum((pos[k, j] - pi[k])^2 for k in axes(pos, 1))
    d2 <= r2 && push!(result, j)
  end
  return result
end

"""
    pairwise_distances(state::ACSet, ob::Symbol, attrs::Vector{Symbol})

Compute pairwise Euclidean distance matrix for all parts of type `ob`.
"""
function pairwise_distances(state::ACSet, ob::Symbol, attrs::Vector{Symbol})
  pos = positions(state, ob, attrs)
  n = size(pos, 2)
  D = zeros(Float64, n, n)
  for i in 1:n, j in (i+1):n
    d = sqrt(sum((pos[k, i] - pos[k, j])^2 for k in axes(pos, 1)))
    D[i, j] = d
    D[j, i] = d
  end
  return D
end

# Calibration / inference
#########################

"""
    log_likelihood(traj::Traj, observations, observe_fn, log_obs_density; times=nothing)

Compute log-likelihood of observed data given a simulation trajectory.
"""
function log_likelihood(traj::Traj, observations, observe_fn::Function,
                        log_obs_density::Function; times=nothing)
  obs_times = isnothing(times) ? [e[1] for e in traj.events] : collect(times)
  length(obs_times) == length(observations) || 
    error("Number of observation times must match observations")
  ll = 0.0
  for (t, obs) in zip(obs_times, observations)
    state = _state_at_time(traj, t)
    sim = observe_fn(state, t)
    ll += log_obs_density(obs, sim)
  end
  return ll
end

function _state_at_time(traj::Traj, t::Float64)
  state = deepcopy(traj.init)
  for (i, (event_t, _, _, _)) in enumerate(traj.events)
    event_t > t && break
    if i <= length(traj.hist)
      state = codom(right(traj.hist[i]))
    end
  end
  return state
end

"""
    particle_filter(abm, init, observations, observe_fn, log_obs_density,
                    obs_times; nparticles=100, params_sampler=nothing, kw...)

Bootstrap particle filter for marginal likelihood estimation.
Returns `(log_marginal_likelihood, particles, weights)`.
"""
function particle_filter(abm::ABM, init::T, observations, observe_fn::Function,
                         log_obs_density::Function, obs_times;
                         nparticles::Int=100, params_sampler=nothing,
                         kw...) where T<:ACSet
  N = nparticles
  length(obs_times) == length(observations) || error("obs_times and observations must match")
  particles = [deepcopy(init) for _ in 1:N]
  log_ml = 0.0
  for (obs_t, obs) in zip(obs_times, observations)
    log_weights = zeros(N)
    for i in 1:N
      p = isnothing(params_sampler) ? nothing : params_sampler()
      abm_i = isnothing(p) ? abm : ABM(abm.rules, abm.dyn; tiepolicy=abm.tiepolicy, params=p)
      traj = run!(abm_i, deepcopy(particles[i]); maxtime=obs_t, save=_->nothing, kw...)
      particles[i] = isempty(traj.hist) ? deepcopy(traj.init) : deepcopy(codom(right(traj.hist[end])))
      sim = observe_fn(particles[i], obs_t)
      log_weights[i] = log_obs_density(obs, sim)
    end
    max_lw = maximum(log_weights)
    weights = exp.(log_weights .- max_lw)
    sum_w = sum(weights)
    log_ml += max_lw + log(sum_w) - log(N)
    weights ./= sum_w
    particles = _systematic_resample(particles, weights)
  end
  return (log_marginal_likelihood=log_ml, particles=particles, weights=fill(1.0/N, N))
end

function _systematic_resample(particles::Vector, weights::Vector{Float64})
  N = length(particles)
  cumw = cumsum(weights)
  u = rand() / N
  new_particles = similar(particles)
  j = 1
  for i in 1:N
    while cumw[j] < u
      j += 1
    end
    new_particles[i] = deepcopy(particles[j])
    u += 1.0 / N
  end
  return new_particles
end

"""
    abc_reject(abm, init, obs_summary, summary_fn, distance_fn;
               nsamples=100, threshold=Inf, params_sampler, kw...)

ABC rejection sampler. Returns vector of `(params, distance)` for accepted samples.
"""
function abc_reject(abm_template::ABM, init::T, obs_summary,
                    summary_fn::Function, distance_fn::Function;
                    nsamples::Int=100, threshold::Float64=Inf,
                    params_sampler::Function, kw...) where T<:ACSet
  accepted = Tuple{Any, Float64}[]
  for _ in 1:nsamples
    params = params_sampler()
    abm = ABM(abm_template.rules, abm_template.dyn; 
              tiepolicy=abm_template.tiepolicy, params=params)
    traj = run!(abm, deepcopy(init); kw...)
    sim_summary = summary_fn(traj)
    d = distance_fn(sim_summary, obs_summary)
    d <= threshold && push!(accepted, (params, d))
  end
  return accepted
end

end # module
