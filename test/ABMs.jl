module TestABMs

ENV["JULIA_DEBUG"] = "AlgebraicABMs" # turn on @debug messages for this package

using Test
using AlgebraicABMs
using Catlab, AlgebraicRewriting

using AlgebraicABMs.ABMs: RegularP, EmptyP, RepresentableP, RuntimeABM
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

# Calibration / particle filtering test (#13)
##############################################

using Distributions: Exponential, logpdf, Poisson
using AlgebraicABMs.ABMs: _state_at_time, _systematic_resample

@testset "Calibration utilities" begin
  # Simple birth model: vertices duplicate at rate 1.0
  v1 = Graph(1)
  v2 = Graph(2)
  dup = ABMRule(:dup, 
    Rule(id(v1), homomorphism(v1, v2; initial=(V=[1],))),
    ContinuousHazard(1.0))
  abm = ABM([dup])
  init = @acset Graph begin V=2 end

  # Test _state_at_time
  traj = run!(abm, init; maxevent=5)
  s0 = _state_at_time(traj, 0.0)
  @test nparts(s0, :V) == 2
  if !isempty(traj.events)
    t_last = traj.events[end][1]
    s_end = _state_at_time(traj, t_last + 1.0)
    @test nparts(s_end, :V) == 7  # 2 initial + 5 duplications
  end

  # Test _systematic_resample
  parts = [1, 2, 3, 4]
  weights = [0.7, 0.1, 0.1, 0.1]
  resampled = _systematic_resample(parts, weights)
  @test length(resampled) == 4
  # High-weight particle should appear more often
  @test count(==(1), resampled) >= 1

  # Test log_likelihood
  obs_times = [traj.events[end][1]]
  obs_data = [nparts(codom(right(traj.hist[end])), :V)]
  ll = log_likelihood(traj, obs_data,
    (state, t) -> nparts(state, :V),
    (obs, sim) -> obs == sim ? 0.0 : -1000.0;
    times=obs_times)
  @test ll ≈ 0.0  # exact match

  # Test particle_filter runs
  obs = [3]
  times = [0.5]
  result = particle_filter(abm, init, obs,
    (s, t) -> Float64(nparts(s, :V)),
    (o, s) -> -0.5 * (o - s)^2,
    times; nparticles=10, maxevent=50)
  @test haskey(result, :log_marginal_likelihood)
  @test haskey(result, :particles)
  @test length(result.particles) == 10

  # Test abc_reject
  accepted = abc_reject(abm, init, 5,
    traj -> length(traj),
    (s, o) -> abs(s - o);
    nsamples=20, threshold=3.0, maxevent=10,
    params_sampler=() -> (rate=rand(),))
  @test accepted isa Vector
  @test all(a -> a[2] <= 3.0, accepted)
end


end # module
