module TestABMs

ENV["JULIA_DEBUG"] = "AlgebraicABMs" # turn on @debug messages for this package

using Test
using AlgebraicABMs
using Catlab, AlgebraicRewriting

using AlgebraicABMs.ABMs: RegularP, EmptyP, RepresentableP, RuntimeABM, Traj, Intervention
using AlgebraicABMs.ABMs: _state_at_time, _systematic_resample
using AlgebraicRewriting.Incremental.IncrementalCC: match_vect
using Distributions: Exponential
import Distributions
using Catlab.CategoricalAlgebra.CSets: MarkAsDeleted
const HAS_UNITFUL = try
  @eval using Unitful
  true
catch
  false
end

# Top-level schema definitions for schema inference/validation tests
# (@acset_type generates `const` which cannot be used inside @testset on Julia 1.12)
@present SchATest(FreeSchema) begin A::Ob end
@acset_type ASet(SchATest)

@present SchBTest(FreeSchema) begin B::Ob; C::AttrType; val::Attr(B, C) end
@acset_type BSet(SchBTest){Int}

@present SchSmallTest(FreeSchema) begin V::Ob end
@acset_type SmallSet(SchSmallTest)

# MarkAsDeleted graph type for schedule tests
@present SchGrphMD(FreeSchema) begin
  V::Ob; E::Ob; src::Hom(E,V); tgt::Hom(E,V)
end
@acset_type GrphMD(SchGrphMD, part_type=MarkAsDeleted)

# Spatial schema for spatial utility tests
@present SchSpatial(FreeSchema) begin
  Agent::Ob
  Coord::AttrType
  px::Attr(Agent, Coord)
  py::Attr(Agent, Coord)
end
@acset_type SpatialSet(SchSpatial, part_type=BitSetParts)

# L = ∅, I = ∅, R = •↺
create_loop = ABMRule(
  :CreateLoop,
  Rule(id(Graph()),                  # l : I -> L
       create(ob(terminal(Graph)))), # r : I → R
  DiscreteHazard(1.)) # Dirac delta, indep. of clock time / state

# check that we know this rule has an empty pattern L
@test create_loop.pattern_type == EmptyP()

# • ← • → •↺
add_loop = ABMRule(Rule(id(Graph(1)), # 
                        delete(Graph(1))),  # r : I -> R
                   ContinuousHazard(1.5))
@test add_loop.pattern_type == RepresentableP(Dict(:V=>[1]))

# •↺ ⇽ • → •
rem_loop = ABMRule(:RemLoop, Rule(delete(Graph(1)), id(Graph(1))), DiscreteHazard(2))
@test rem_loop.pattern_type == RegularP()

# •→• ⇽ • → •
rem_edge = ABMRule(:RemEdge, Rule(homomorphism(Graph(2), path_graph(Graph, 2); initial=(V=1:2,)), 
                        id(Graph(2))), 
                   ContinuousHazard(1))
@test rem_edge.pattern_type == RepresentableP(Dict(:E=>[1]))

# Create initial state 
G = @acset Graph begin V=3; E=3; src=[1,1,1]; tgt=[1,1,2] end
to_graphviz(G)

# Assemble rules into ABM
abm = ABM([create_loop, add_loop, rem_loop, rem_edge])

# Test a few collection methods for ABMs
@test length(abm) == 4
abm = filter(r -> r.name != :RemEdge, abm)
@test length(abm) == 3
push!(abm, rem_edge)
@test length(abm) == 4

new_abm = copy(abm)
@test !(new_abm === abm)
for i in eachindex(abm.rules)
  @test abm.rules[i] == new_abm.rules[i]
end

do_nothing = ABMRule(
  :DoNothing,
  Rule{:DPO}(
    id(representable(Graph, :V)),
    id(representable(Graph, :V))
  ),
  ContinuousHazard(1)
)
push!(new_abm, do_nothing)
@test length(new_abm) == length(abm) + 1

@test_throws ErrorException push!(new_abm, rem_loop)

# 2 loops, so 2 cached homs for the only rule with an explicit hom set
@test length(only(match_vect(RuntimeABM(abm, G)[:RemLoop].val))) == 2

traj = run!(abm, G; maxevent=10);

traj = run!(ABM([rem_edge]), G);
@test length(traj) == 3


traj = run!(ABM([add_loop]), G);
@test length(traj) > 3 # after we add a loop, the match persists + is resampled

let traj = run!(ABM([create_loop]), Graph(); maxevent=2, maxtime=10.0)
  @test length(traj) == 2
  @test traj.events[end][1] == 2.0
end



# Test events in parallel
create_loop = ABMRule(Rule(id(Graph(1)), delete(Graph(1))), DiscreteHazard(1.));
create_vert = ABMRule(Rule(id(Graph()),  create(Graph(1))), DiscreteHazard(1.));
abm = ABM([create_loop, create_vert]);
traj = run!(abm, Graph(); maxtime=5);


# Basis
#######
using AlgebraicABMs, Catlab, AlgebraicRewriting

# Rule which fires once per vertex (it tries to add an edge to an 
# arbitrary other vertex) - rate is based on vertex pattern, not pattern for 
# the rule itself, which has two 
add_edge = ABMRule(Rule(id(Graph(2)), 
                        homomorphism(Graph(2), path_graph(Graph, 2); 
                                     any=true, monic=true)),
                   DiscreteHazard(1.),
                   basis=homomorphism(Graph(1), Graph(2); any=true))
abm = ABM([add_edge])
rt = RuntimeABM(abm, Graph(3))
traj = run!(abm, Graph(3); maxtime=3);

view(traj, graphviz_write)


# Tie-breaking policy
#######################

@testset "TiePolicy" begin
  # Two rules firing at t=1, both duplicating a vertex
  g1, g2 = Graph(1), Graph(2)
  r_map = homomorphism(g1, g2; initial=(V=[1],))
  dup1 = ABMRule(:dup1, Rule(id(g1), r_map), DiscreteHazard(1.))
  dup2 = ABMRule(:dup2, Rule(id(g1), r_map), DiscreteHazard(1.))

  @testset "TieBreak (default)" begin
    abm = ABM([dup1, dup2])
    @test abm.tiepolicy == TieBreak
    traj = run!(abm, Graph(1), maxevent=1)
    @test nparts(codom(right(last(traj.hist))), :V) > 1
  end

  @testset "TieRandom" begin
    abm = ABM([dup1, dup2]; tiepolicy=TieRandom)
    @test abm.tiepolicy == TieRandom
    traj = run!(abm, Graph(1), maxevent=1)
    @test nparts(codom(right(last(traj.hist))), :V) > 1
  end

  @testset "TieError" begin
    abm = ABM([dup1, dup2]; tiepolicy=TieError)
    @test abm.tiepolicy == TieError
    @test_throws ErrorException run!(abm, Graph(1), maxevent=1)
  end

  @testset "Single rule no error" begin
    abm = ABM([dup1]; tiepolicy=TieError)
    traj = run!(abm, Graph(1), maxevent=1)
    @test nparts(codom(right(last(traj.hist))), :V) == 2
  end
end


# Interventions
##################

@testset "Interventions" begin
  dup_rule = ABMRule(:dup, Rule(id(Graph(1)), homomorphism(Graph(1), Graph(2); initial=(V=[1],))), DiscreteHazard(1.))

  @testset "Scheduled intervention" begin
    iv = Intervention(0.5, state -> add_parts!(state, :V, 5); name=:add5)
    abm = ABM([dup_rule])
    traj = run!(abm, Graph(1); maxtime=0.9, interventions=[iv])
  end

  @testset "Segmented run with refresh_clocks!" begin
    abm = ABM([dup_rule])
    rt = RuntimeABM(abm, Graph(1))
    traj = run!(abm, rt, Traj(Graph(1)); maxevent=1)
    @test nparts(rt.state, :V) == 2
    add_parts!(rt.state, :V, 3)
    @test nparts(rt.state, :V) == 5
    refresh_clocks!(rt, abm)
    traj2 = run!(abm, rt, traj; maxevent=2)
    @test nparts(rt.state, :V) > 5
  end

  @testset "refresh_clocks! re-enables schedule timers" begin
    z = GrphMD()
    g1 = @acset GrphMD begin V=1 end
    add_v_rule = Rule(id(z), create(g1))
    sched = tryrule(RuleApp(:add_v, add_v_rule, z))
    abm_sched = ABM(ABMRule[]; schedules=[ABMSchedule(:add_v_sched, sched, DiscreteHazard(1.0))])

    init = @acset GrphMD begin V=1 end
    rt = RuntimeABM(abm_sched, init)
    traj = run!(abm_sched, rt, Traj(init); maxevent=1)
    @test nparts(rt.state, :V) == 2

    add_part!(rt.state, :V)
    refresh_clocks!(rt, abm_sched)
    traj2 = run!(abm_sched, rt, traj; maxevent=2)
    @test nparts(rt.state, :V) == 4
  end

  @testset "Conditional intervention" begin
    iv = Intervention(
      state -> nparts(state, :V) >= 3,
      state -> rem_part!(state, :V, 1);
      name=:trim
    )
    abm = ABM([dup_rule])
    rt = RuntimeABM(abm, Graph(2))
    traj = run!(abm, rt, Traj(Graph(2)); maxevent=2, interventions=[iv])
    @test nparts(rt.state, :V) >= 2
  end

  @testset "Scheduled intervention at exact time" begin
    iv = Intervention(1.0, state -> add_parts!(state, :V, 10); name=:add10)
    abm = ABM([dup_rule])
    rt = RuntimeABM(abm, Graph(1))
    traj = run!(abm, rt, Traj(Graph(1)); maxtime=1.5, interventions=[iv])
    @test nparts(rt.state, :V) > 10
  end

  @testset "Intervention preserves schedule timers" begin
    z = GrphMD()
    g1 = @acset GrphMD begin V=1 end
    add_v_rule = Rule(id(z), create(g1))
    sched = tryrule(RuleApp(:add_v, add_v_rule, z))
    abm_sched = ABM(ABMRule[]; schedules=[ABMSchedule(:add_v_sched, sched, DiscreteHazard(1.0))])

    init = @acset GrphMD begin V=1 end
    iv = Intervention(1.5, state -> add_part!(state, :V); name=:manual_add)
    traj = run!(abm_sched, init; maxtime=3.1, interventions=[iv])
    @test nparts(isempty(traj.hist) ? init : codom(right(traj.hist[end])), :V) == 5
  end
end


# ODEs
#####################
using AlgebraicABMs, AlgebraicRewriting, Catlab, DifferentialEquations
using AlgebraicABMs.ABMs: RuntimeABM

# State of world: a set of free-floating Float64s
@present SchLSet(FreeSchema) begin X::Ob; D::AttrType; f::Attr(X, D) end
@acset_type LSet(SchLSet){Float64}

# Pattern: a single vertex with attribute variable
v = @acset LSet begin X=1; D=1; f=[AttrVar(1)] end

# Dynamics: for an individual variable, it grows linearly w/ time
flow = ABMFlow(v, RawODE([_ -> 1.0]), :Grow, [], [(:D => 1)])

# Rule: copy a variable
v2 = @acset LSet begin X=2; D=1; f=[AttrVar(1), AttrVar(1)] end
dup_vertex = ABMRule(Rule(id(v), homomorphism(v, v2; initial=(X=[1],))), DiscreteHazard(1.))

@testset "Pure ODE" begin
  init = @acset LSet begin X=2; f=[1.0, 2.0] end
  abm_ode = ABM(ABMRule[], [flow])
  rt = RuntimeABM(abm_ode, deepcopy(init))
  run!(abm_ode, rt, AlgebraicABMs.ABMs.Traj(deepcopy(init)); maxtime=2.0, dt=0.5)
  # Linear growth rate 1.0 for 2 time units: values increase by 2.0
  @test abs(rt.state[:f][1] - 3.0) < 0.01
  @test abs(rt.state[:f][2] - 4.0) < 0.01
end

@testset "ODE accuracy (long)" begin
  init = @acset LSet begin X=1; f=[5.0] end
  abm_ode = ABM(ABMRule[], [flow])
  rt = RuntimeABM(abm_ode, deepcopy(init))
  run!(abm_ode, rt, AlgebraicABMs.ABMs.Traj(deepcopy(init)); maxtime=10.0, dt=1.0)
  @test abs(rt.state[:f][1] - 15.0) < 0.01
end

@testset "Hybrid ODE + Stochastic" begin
  init = @acset LSet begin X=2; f=[1.0, 2.0] end
  abm_hybrid = ABM([dup_vertex], [flow])
  rt = RuntimeABM(abm_hybrid, deepcopy(init))
  run!(abm_hybrid, rt, AlgebraicABMs.ABMs.Traj(deepcopy(init)); maxtime=1.5, dt=0.1)
  # At t=1: ODE grew values by 1.0, then duplication fires → more parts
  @test nparts(rt.state, :X) > 2
  # All values should have grown from ODE integration
  @test all(v -> v > 1.0, rt.state[:f])
end


# Schema inference and validation
##################################

using AlgebraicABMs.ABMs: infer_schema, validate_schema, rule_schema, is_subschema

@testset "Schema inference" begin
  # Infer schema from rules that share the same schema
  s = infer_schema([create_loop, add_loop, rem_loop, rem_edge])
  @test Set(objects(s)) == Set([:V, :E])
  @test length(homs(s)) == 2  # src, tgt

  # ABM constructor auto-infers schema
  abm_inferred = ABM([create_loop, add_loop])
  @test !isnothing(abm_inferred.schema)
  @test Set(objects(abm_inferred.schema)) == Set([:V, :E])
end

@testset "Schema validation" begin
  # Validate rules against a matching schema
  graph_schema = rule_schema(add_loop)
  abm_validated = ABM([create_loop, add_loop]; schema=graph_schema)
  @test abm_validated.schema == graph_schema

  # Validation should fail with a mismatched schema (LSet schema for Graph rules)
  wrong_schema = acset_schema(LSet())
  @test_throws ErrorException ABM([create_loop]; schema=wrong_schema)
end

@testset "Schema merge across different schemas" begin
  a1 = @acset ASet begin A=1 end
  a2 = @acset ASet begin A=2 end
  rule_a = ABMRule(:rA, Rule(id(a1), homomorphism(a1, a2; initial=(A=[1],))), DiscreteHazard(1.))

  b1 = @acset BSet begin B=1; C=1; val=[AttrVar(1)] end
  b2 = @acset BSet begin B=2; C=1; val=[AttrVar(1), AttrVar(1)] end
  rule_b = ABMRule(:rB, Rule(id(b1), homomorphism(b1, b2; initial=(B=[1],))), DiscreteHazard(1.))

  merged = infer_schema([rule_a, rule_b])
  @test Set(objects(merged)) == Set([:A, :B])
  @test length(attrs(merged)) == 1  # val
  @test length(attrtypes(merged)) == 1  # C
end

@testset "is_subschema" begin
  s_graph = rule_schema(add_loop)
  s_graph2 = rule_schema(rem_edge)
  @test is_subschema(s_graph, s_graph2)[1]  # same schema

  s_small = acset_schema(SmallSet())
  @test is_subschema(s_small, s_graph)[1]   # V ⊂ {V,E,src,tgt}
  @test !is_subschema(s_graph, s_small)[1]  # {V,E} ⊄ {V}
end


# Parameters and scenarios
###########################

@testset "Parameters" begin
  dup_p = ABMRule(:dup_p, Rule(id(Graph(1)), homomorphism(Graph(1), Graph(2); initial=(V=[1],))),
                  ClosureParams((m, params) -> Exponential(params.rate)))
  
  abm_p = ABM([dup_p]; params=(rate=0.5,))
  @test abm_p.params == (rate=0.5,)
  
  traj = run!(abm_p, Graph(1); maxevent=3)
  @test length(traj) >= 3
end

@testset "FullClosureParams" begin
  rule_fp = ABMRule(:fp, Rule(id(Graph(1)), homomorphism(Graph(1), Graph(2); initial=(V=[1],))),
                    FullClosureParams((m, t, p) -> Exponential(p.base_rate * (1 + t))))
  abm_fp = ABM([rule_fp]; params=(base_rate=1.0,))
  traj = run!(abm_fp, Graph(1); maxevent=2)
  @test length(traj) >= 2
end

@testset "run_scenarios" begin
  dup_s = ABMRule(:dup_s, Rule(id(Graph(1)), homomorphism(Graph(1), Graph(2); initial=(V=[1],))),
                  ClosureParams((m, params) -> Exponential(params.rate)))
  
  abm_s = ABM([dup_s]; params=(rate=1.0,))
  scenarios = [(rate=0.1,), (rate=1.0,), (rate=10.0,)]
  
  results = run_scenarios(abm_s, Graph(1), scenarios; maxevent=5)
  @test length(results) == 3
  @test all(r -> r isa Pair, results)
  @test results[1].first == (rate=0.1,)
  @test all(r -> length(r.second) >= 5, results)
end

@testset "Params backward compatibility" begin
  abm_no_p = ABM([create_loop, add_loop])
  @test isnothing(abm_no_p.params)
  traj = run!(abm_no_p, G; maxevent=5)
  @test length(traj) >= 5
end


# Schedules
##################

@testset "ABMSchedule basic" begin
  z = GrphMD()
  g1 = @acset GrphMD begin V=1 end
  add_v_rule = Rule(id(z), create(g1))
  ra = RuleApp(:add_v, add_v_rule, z)
  sched = tryrule(ra)
  
  abm_sched = ABMSchedule(:add_v_sched, sched, DiscreteHazard(1.))
  @test nameof(abm_sched) == :add_v_sched
  
  abm = ABM(ABMRule[]; schedules=[abm_sched])
  @test length(abm.schedules) == 1
  
  init = @acset GrphMD begin V=2 end
  traj = run!(abm, init; maxevent=5)
  @test length(traj) >= 3
end

@testset "ABMSchedule with rules" begin
  z = GrphMD()
  g1 = @acset GrphMD begin V=1 end
  g2 = @acset GrphMD begin V=2 end
  
  h = homomorphism(g1, g2; initial=(V=[1],))
  dup_rule = ABMRule(:dup, Rule(id(g1), h), ContinuousHazard(0.5))
  
  add_v_rule = Rule(id(z), create(g1))
  ra = RuleApp(:sched_add, add_v_rule, z)
  sched = tryrule(ra)
  abm_sched = ABMSchedule(:periodic_add, sched, DiscreteHazard(2.))
  
  abm = ABM([dup_rule]; schedules=[abm_sched])
  init = @acset GrphMD begin V=1 end
  traj = run!(abm, init; maxevent=5)
  @test length(traj) >= 3
end

# Networkify
############

@testset "networkify Presentation" begin
  @present SchSIR(FreeSchema) begin S::Ob; I::Ob; R::Ob end
  S_net = networkify(SchSIR)
  gen_names = [first(g) for g in generators(S_net)]
  @test :V ∈ gen_names && :E ∈ gen_names
  @test :loc_S ∈ gen_names && :loc_I ∈ gen_names && :loc_R ∈ gen_names
  @test length(generators(S_net, :Ob)) == 5
  @test length(generators(S_net, :Hom)) == 5
end

@testset "networkify BasicSchema" begin
  bs = Catlab.BasicSchema{Symbol}(
    [:Person], Tuple{Symbol,Symbol,Symbol}[], [:Status], [(:status, :Person, :Status)],
    Tuple{Union{Nothing,Symbol},Symbol,Symbol,Tuple{Tuple{Vararg{Symbol}},Tuple{Vararg{Symbol}}}}[])
  bs_net = networkify(bs)
  @test :V ∈ objects(bs_net) && :E ∈ objects(bs_net)
  @test (:loc_Person, :Person, :V) ∈ homs(bs_net)
  @test (:status, :Person, :Status) ∈ attrs(bs_net)
end

@testset "shortest_distance" begin
  g = @acset GrphMD begin V=5; E=8; src=[1,2,3,4,2,3,4,5]; tgt=[2,3,4,5,1,2,3,4] end
  @test shortest_distance(g, 1, 1) == 0
  @test shortest_distance(g, 1, 2) == 1
  @test shortest_distance(g, 1, 3) == 2
  @test shortest_distance(g, 1, 5) == 4
  g2 = @acset GrphMD begin V=4; E=2; src=[1,2]; tgt=[2,1] end
  @test shortest_distance(g2, 1, 2) == 1
  @test shortest_distance(g2, 1, 3) == typemax(Int)
end

# Phase 4-5 tests
##################

# ClosureHistory test: history-sensitive hazard rates (#22)
@present SchGrphT(FreeSchema) begin V::Ob; E::Ob; src::Hom(E,V); tgt::Hom(E,V) end
@acset_type GrphT(SchGrphT, part_type=BitSetParts)

@testset "ClosureHistory" begin
  v_grph = @acset GrphT begin V=1 end
  v2_grph = @acset GrphT begin V=2 end
  dup_v_hist = ABMRule(:dup_hist,
    Rule(id(v_grph), homomorphism(v_grph, v2_grph; initial=(V=[1],))),
    ClosureHistory((m, t, traj) -> begin
      n = isnothing(traj) ? 0 : length(traj)
      Exponential(1.0 + n)
    end))
  abm = ABM([dup_v_hist])
  init = @acset GrphT begin V=2 end
  res = run!(abm, init; maxevent=5)
  @test length(res) == 5
  times = [e[1] for e in res.events]
  @test issorted(times)
  @test all(t -> t > 0, times)
end

@testset "ABMRule context/dependency" begin
  L = Graph(1)
  Ctx = @acset Graph begin V=2; E=1; src=[1]; tgt=[2] end
  ctx_morph = homomorphism(L, Ctx; initial=(V=[1],))
  R = Graph(2)
  I = Graph(1)
  dup_rule = Rule(id(I), homomorphism(I, R; initial=(V=[1],)))
  rule_with_ctx = ABMRule(dup_rule, ClosureState(m -> Exponential(1.0));
    context=ctx_morph, name=:dup_ctx)
  @test !isnothing(rule_with_ctx.context)
  @test isnothing(rule_with_ctx.dependency)

  Dep = Graph(1)
  dep_morph = homomorphism(Dep, Ctx; initial=(V=[1],))
  rule_with_dep = ABMRule(dup_rule, ClosureState(m -> Exponential(1.0));
    context=ctx_morph, dependency=dep_morph, name=:dup_dep)
  @test !isnothing(rule_with_dep.context)
  @test !isnothing(rule_with_dep.dependency)

  G = @acset Graph begin V=3; E=2; src=[1,2]; tgt=[2,3] end
  m = homomorphism(L, G; initial=(V=[1],))
  m_ctx = resolve_match(rule_with_ctx, m)
  @test nparts(dom(m_ctx), :V) == 2
  @test nparts(dom(m_ctx), :E) == 1
  m_dep = resolve_match(rule_with_dep, m)
  @test nparts(dom(m_dep), :V) == 1
  m_plain = resolve_match(ABMRule(dup_rule, ContinuousHazard(1.0); name=:dup_plain), m)
  @test m_plain == m

  G_noctx = Graph(1)
  m_noctx = homomorphism(L, G_noctx; initial=(V=[1],))
  @test resolve_match(rule_with_ctx, m_noctx) == m_noctx

  G_amb = @acset Graph begin V=3; E=2; src=[1,1]; tgt=[2,3] end
  m_amb = homomorphism(L, G_amb; initial=(V=[1],))
  @test_throws ErrorException resolve_match(rule_with_ctx, m_amb)
  m_amb_dep = resolve_match(rule_with_dep, m_amb)
  @test nparts(dom(m_amb_dep), :V) == 1
  @test collect(m_amb_dep[:V]) == [1]

  abm_ctx = ABM([rule_with_ctx])
  init_g = @acset Graph begin V=3; E=2; src=[1,2]; tgt=[2,3] end
  res = run!(abm_ctx, init_g; maxevent=3)
  @test length(res) == 3
end

@testset "match_equal with fix" begin
  L = @acset Graph begin V=1; E=1; src=[1]; tgt=[1] end
  L_fix = Graph(1)
  fix_morph = homomorphism(L_fix, L)
  rule_no_fix = ABMRule(Rule(id(L), id(L)), ContinuousHazard(1.0); name=:nf)
  rule_fix = ABMRule(Rule(id(L), id(L)), ContinuousHazard(1.0); name=:wf, fix=fix_morph)

  G = @acset Graph begin V=2; E=2; src=[1,2]; tgt=[1,2] end
  m1 = homomorphism(L, G; initial=(V=[1], E=[1]))
  m2 = homomorphism(L, G; initial=(V=[1], E=[1]))
  m3 = homomorphism(L, G; initial=(V=[2], E=[2]))

  @test match_equal(rule_no_fix, m1, m2) == true
  @test match_equal(rule_fix, m1, m2) == true
  @test match_equal(rule_no_fix, m1, m3) == false
  @test match_equal(rule_fix, m1, m3) == false
  @test isnothing(rule_no_fix.fix)
  @test !isnothing(rule_fix.fix)
end

if HAS_UNITFUL
@eval module UnitfulTests
  using Test, Unitful, Distributions
  using AlgebraicABMs
  using Catlab, AlgebraicRewriting

  @testset "Unitful extension" begin
    h1 = ContinuousHazard(0.1u"s^-1")
    @test h1.val isa Distributions.Exponential
    @test h1.val.θ ≈ 10.0

    h2 = DiscreteHazard(5.0u"s")
    @test h2.val isa Distributions.Dirac

    @test validate_units(0.1u"s^-1")
    @test validate_units(1.0u"d^-1")
    @test_throws Unitful.DimensionError validate_units(1.0u"m")
    @test_throws Unitful.DimensionError validate_units(1.0u"kg")

    @test strip_units(5.0u"s") == 5.0
    @test strip_units(3.14) == 3.14

    ext = Base.get_extension(AlgebraicABMs.ABMs, :UnitfulExt)
    mt, d = ext.check_time_units(100u"d", 0.1u"d")
    @test mt ≈ 100.0
    @test d ≈ 0.1

    create_rule = ABMRule(:Create,
      Rule(id(Graph()), create(ob(terminal(Graph)))),
      ContinuousHazard(1.0u"s^-1"))
    abm = ABM([create_rule])
    init = @acset Graph begin V=1; E=1; src=[1]; tgt=[1] end
    res = run!(abm, init; maxevent=3)
    @test length(res) == 3
  end
end
else
  @warn "Unitful not available, skipping Unitful extension tests"
end

@testset "Spatial utilities" begin
  state = SpatialSet{Float64}()
  add_parts!(state, :Agent, 4; px=[0.0, 3.0, 0.0, 10.0], py=[0.0, 4.0, 1.0, 10.0])

  pos = positions(state, :Agent, [:px, :py])
  @test size(pos) == (2, 4)
  @test pos[:, 1] == [0.0, 0.0]
  @test pos[:, 2] == [3.0, 4.0]

  near1 = within_radius(state, 1, 5.0, :Agent, [:px, :py])
  @test 3 ∈ near1
  @test 2 ∈ near1
  @test 4 ∉ near1
  @test 1 ∉ near1

  near1_inc = within_radius(state, 1, 5.0, :Agent, [:px, :py]; exclude_self=false)
  @test 1 ∈ near1_inc

  close = within_radius(state, 1, 2.0, :Agent, [:px, :py])
  @test close == [3]

  D = pairwise_distances(state, :Agent, [:px, :py])
  @test size(D) == (4, 4)
  @test D[1, 1] == 0.0
  @test D[1, 2] ≈ 5.0
  @test D[1, 3] ≈ 1.0
  @test all(D[i, j] ≈ D[j, i] for i in 1:4, j in 1:4)

  empty_state = SpatialSet{Float64}()
  pos_empty = positions(empty_state, :Agent, [:px, :py])
  @test size(pos_empty) == (2, 0)
end

@testset "Calibration utilities" begin
  v1 = Graph(1)
  v2 = Graph(2)
  dup = ABMRule(:dup,
    Rule(id(v1), homomorphism(v1, v2; initial=(V=[1],))),
    ContinuousHazard(1.0))
  abm = ABM([dup])
  init = @acset Graph begin V=2 end

  traj = run!(abm, init; maxevent=5)
  s0 = _state_at_time(traj, 0.0)
  @test nparts(s0, :V) == 2
  if !isempty(traj.events)
    t_last = traj.events[end][1]
    s_end = _state_at_time(traj, t_last + 1.0)
    @test nparts(s_end, :V) == 7
  end

  parts = [1, 2, 3, 4]
  weights = [0.7, 0.1, 0.1, 0.1]
  resampled = _systematic_resample(parts, weights)
  @test length(resampled) == 4
  @test count(==(1), resampled) >= 1

  obs_times = [traj.events[end][1]]
  obs_data = [nparts(codom(right(traj.hist[end])), :V)]
  ll = log_likelihood(traj, obs_data,
    (state, t) -> nparts(state, :V),
    (obs, sim) -> obs == sim ? 0.0 : -1000.0;
    times=obs_times)
  @test ll ≈ 0.0

  result = particle_filter(abm, init, [3],
    (s, t) -> Float64(nparts(s, :V)),
    (o, s) -> -0.5 * (o - s)^2,
    [0.5];
    nparticles=10, maxevent=50)
  @test haskey(result, :log_marginal_likelihood)
  @test haskey(result, :particles)
  @test length(result.particles) == 10

  accepted = abc_reject(abm, init, 5,
    traj -> length(traj),
    (s, o) -> abs(s - o);
    nsamples=20, threshold=3.0, maxevent=10,
    params_sampler=() -> (rate=rand(),))
  @test accepted isa Vector
  @test all(a -> a[2] <= 3.0, accepted)
end


end # module
