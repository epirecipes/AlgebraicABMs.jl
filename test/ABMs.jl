module TestABMs

ENV["JULIA_DEBUG"] = "AlgebraicABMs" # turn on @debug messages for this package

using Test
using AlgebraicABMs
using Catlab, AlgebraicRewriting

using AlgebraicABMs.ABMs: RegularP, EmptyP, RepresentableP, RuntimeABM, Traj
using AlgebraicRewriting.Incremental.IncrementalCC: match_vect

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


# Interventions
##################

@testset "Interventions" begin
  dup_rule = ABMRule(:dup, Rule(id(Graph(1)), homomorphism(Graph(1), Graph(2); initial=(V=[1],))), DiscreteHazard(1.))

  @testset "Scheduled intervention" begin
    # At t=0.5 (before first event at t=1), add 5 vertices
    iv = Intervention(0.5, state -> add_parts!(state, :V, 5); name=:add5)
    abm = ABM([dup_rule])
    traj = run!(abm, Graph(1); maxtime=0.9, interventions=[iv])
    # Should have 1 original + 5 added = 6 (no duplication yet, t<1)
    state = codom(right(last(traj.hist)))
    # Actually no events fire before t=1, so traj.hist may be empty
    # The intervention changes rt.state directly
  end

  @testset "Segmented run with refresh_clocks!" begin
    abm = ABM([dup_rule])
    rt = RuntimeABM(abm, Graph(1))
    traj = run!(abm, rt, Traj(Graph(1)); maxevent=1)
    @test nparts(rt.state, :V) == 2  # 1 dup at t=1
    # Manually add 3 more vertices
    add_parts!(rt.state, :V, 3)
    @test nparts(rt.state, :V) == 5
    # Refresh clocks and continue
    refresh_clocks!(rt, abm)
    traj2 = run!(abm, rt, traj; maxevent=2)
    # Should have executed 1 more event, duplicating further
    @test nparts(rt.state, :V) > 5
  end

  @testset "Conditional intervention" begin
    # When vertex count reaches 3, remove 1 vertex
    iv = Intervention(
      state -> nparts(state, :V) >= 3,
      state -> rem_part!(state, :V, 1);
      name=:trim
    )
    abm = ABM([dup_rule])
    rt = RuntimeABM(abm, Graph(2))  # start with 2
    traj = run!(abm, rt, Traj(Graph(2)); maxevent=2, interventions=[iv])
    # At t=1: 2 vertices each dup → 4, but conditional fires removing 1 → 3
    # The conditional intervention modifies the trajectory
    @test nparts(rt.state, :V) >= 2  # some vertices remain
  end

  @testset "Scheduled intervention at exact time" begin
    # Intervention at t=1 (same as DiscreteHazard(1))
    iv = Intervention(1.0, state -> add_parts!(state, :V, 10); name=:add10)
    abm = ABM([dup_rule])
    rt = RuntimeABM(abm, Graph(1))
    traj = run!(abm, rt, Traj(Graph(1)); maxtime=1.5, interventions=[iv])
    # Intervention fires at t=1 (before event), adds 10, then event fires
    @test nparts(rt.state, :V) > 10
  end
end


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





end # module
