module TestABMs

ENV["JULIA_DEBUG"] = "AlgebraicABMs" # turn on @debug messages for this package

using Test
using AlgebraicABMs
using Catlab, AlgebraicRewriting

using AlgebraicABMs.ABMs: RegularP, EmptyP, RepresentableP, RuntimeABM
using AlgebraicRewriting.Incremental.IncrementalCC: match_vect
using Distributions: Exponential

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


# ODEs (IN PROGRESS)
#####################
using AlgebraicABMs, AlgebraicRewriting, Catlab

# State of world: a set of free-floating Float64s 
@present SchLSet(FreeSchema) begin X::Ob; D::AttrType; f::Attr(X, D) end 
@acset_type LSet(SchLSet){Float64} 

# Rule: copy a variable
v = @acset LSet begin X=1; D=1; f=[AttrVar(1)] end
v2 = @acset LSet begin X=2; D=1; f=[AttrVar(1), AttrVar(1)] end
dup_vertex = ABMRule(Rule(id(v), homomorphism(v, v2; initial=(X=[1],))), DiscreteHazard(1.))

# Dynamics: for an individual variable, it grows linearly w/ time
flow = ABMFlow(v, RawODE([_ -> 1.0]), :Grow, [], [(:D => 1)])
# Make ABM
abm = ABM([dup_vertex], [flow])

# Initial state
init = @acset LSet begin X=2; f=[1.1, 2.2] end

# res = run!(abm, init, maxevent=2)


# Parameters and scenarios
###########################

@testset "Parameters" begin
  # ABM with params accessible to ClosureParams timer
  dup_p = ABMRule(:dup_p, Rule(id(Graph(1)), homomorphism(Graph(1), Graph(2); initial=(V=[1],))),
                  ClosureParams((m, params) -> Exponential(params.rate)))
  
  abm_p = ABM([dup_p]; params=(rate=0.5,))
  @test abm_p.params == (rate=0.5,)
  
  traj = run!(abm_p, Graph(1); maxevent=3)
  @test length(traj) >= 3
end

@testset "FullClosureParams" begin
  # Timer depends on match, time, and params
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
  @test results[2].first == (rate=1.0,)
  @test results[3].first == (rate=10.0,)
  # All trajectories should have events
  @test all(r -> length(r.second) >= 5, results)
end

@testset "Params backward compatibility" begin
  # ABM without params still works (params=nothing by default)
  abm_no_p = ABM([create_loop, add_loop])
  @test isnothing(abm_no_p.params)
  traj = run!(abm_no_p, G; maxevent=5)
  @test length(traj) >= 5
end


end # module
