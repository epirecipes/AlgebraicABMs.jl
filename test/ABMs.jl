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

# Dependency restriction test (#17)
####################################

@testset "ABMRule context/dependency" begin
  # Test 1: context extends the match to include neighborhood
  # Pattern: a single vertex (L = •)
  # Context: a vertex with an edge (Ctx = •→•), so hazard can see neighbors
  L = Graph(1)
  Ctx = @acset Graph begin V=2; E=1; src=[1]; tgt=[2] end
  ctx_morph = homomorphism(L, Ctx; initial=(V=[1],))

  # The rule duplicates a vertex: • ↦ ••
  R = Graph(2)
  I = Graph(1)
  dup_rule = Rule(id(I), homomorphism(I, R; initial=(V=[1],)))

  # With context, the hazard receives a match Ctx→X (can see the edge)
  rule_with_ctx = ABMRule(dup_rule, ClosureState(m -> begin
    # m is now Ctx→X, so we can count edges in the image
    Exponential(1.0)
  end); context=ctx_morph, name=:dup_ctx)

  @test !isnothing(rule_with_ctx.context)
  @test isnothing(rule_with_ctx.dependency)

  # Test 2: dependency restricts what the hazard sees
  Dep = Graph(1)  # dependency is just a vertex (subset of context)
  dep_morph = homomorphism(Dep, Ctx; initial=(V=[1],))

  rule_with_dep = ABMRule(dup_rule, ClosureState(m -> begin
    Exponential(1.0)
  end); context=ctx_morph, dependency=dep_morph, name=:dup_dep)

  @test !isnothing(rule_with_dep.context)
  @test !isnothing(rule_with_dep.dependency)

  # Test 3: resolve_match extends through context
  G = @acset Graph begin V=3; E=2; src=[1,2]; tgt=[2,3] end
  # Match L→G sending vertex 1 to vertex 1
  m = homomorphism(L, G; initial=(V=[1],))
  m_ctx = resolve_match(rule_with_ctx, m)
  # Extended match should map from Ctx (2 vertices, 1 edge) to G
  @test nparts(dom(m_ctx), :V) == 2
  @test nparts(dom(m_ctx), :E) == 1

  # Test 4: resolve_match with dependency restricts to Dep
  m_dep = resolve_match(rule_with_dep, m)
  # Dependency match maps from Dep (1 vertex) to G
  @test nparts(dom(m_dep), :V) == 1

  # Test 5: resolve_match without context returns original match
  rule_plain = ABMRule(dup_rule, ContinuousHazard(1.0); name=:dup_plain)
  m_plain = resolve_match(rule_plain, m)
  @test m_plain == m

  # Test 6: run with context-aware rule
  abm_ctx = ABM([rule_with_ctx])
  init_g = @acset Graph begin V=3; E=2; src=[1,2]; tgt=[2,3] end
  res = run!(abm_ctx, init_g; maxevent=3)
  @test length(res) == 3
end


end # module
