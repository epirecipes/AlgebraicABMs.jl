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

# Networkify
############

@testset "networkify Presentation" begin
  @present SchSIR(FreeSchema) begin S::Ob; I::Ob; R::Ob end
  S_net = networkify(SchSIR)
  gen_names = [first(g) for g in generators(S_net)]
  @test :V ∈ gen_names
  @test :E ∈ gen_names
  @test :src ∈ gen_names
  @test :tgt ∈ gen_names
  @test :loc_S ∈ gen_names
  @test :loc_I ∈ gen_names
  @test :loc_R ∈ gen_names
  @test length(generators(S_net, :Ob)) == 5  # S, I, R, V, E
  @test length(generators(S_net, :Hom)) == 5  # src, tgt, loc_S, loc_I, loc_R
end

@testset "networkify BasicSchema" begin
  bs = Catlab.BasicSchema{Symbol}(
    [:Person], Tuple{Symbol,Symbol,Symbol}[], [:Status], [(:status, :Person, :Status)],
    Tuple{Union{Nothing,Symbol},Symbol,Symbol,Tuple{Tuple{Vararg{Symbol}},Tuple{Vararg{Symbol}}}}[])
  bs_net = networkify(bs)
  @test :V ∈ objects(bs_net)
  @test :E ∈ objects(bs_net)
  @test (:src, :E, :V) ∈ homs(bs_net)
  @test (:tgt, :E, :V) ∈ homs(bs_net)
  @test (:loc_Person, :Person, :V) ∈ homs(bs_net)
  @test (:status, :Person, :Status) ∈ attrs(bs_net)
end

@testset "networkify custom prefix" begin
  @present SchA(FreeSchema) begin A::Ob end
  S_net = networkify(SchA; loc_prefix=:at_)
  gen_names = [first(g) for g in generators(S_net)]
  @test :at_A ∈ gen_names
  @test :loc_A ∉ gen_names
end

@testset "shortest_distance" begin
  @present SchG(FreeSchema) begin V::Ob; E::Ob; src::Hom(E,V); tgt::Hom(E,V) end
  @acset_type TG(SchG)
  g = @acset TG begin V=5; E=8; src=[1,2,3,4,2,3,4,5]; tgt=[2,3,4,5,1,2,3,4] end
  @test shortest_distance(g, 1, 1) == 0
  @test shortest_distance(g, 1, 2) == 1
  @test shortest_distance(g, 1, 3) == 2
  @test shortest_distance(g, 1, 5) == 4
  @test shortest_distance(g, 3, 5) == 2
  
  g2 = @acset TG begin V=4; E=2; src=[1,2]; tgt=[2,1] end
  @test shortest_distance(g2, 1, 2) == 1
  @test shortest_distance(g2, 1, 3) == typemax(Int)
end

@testset "networkify pattern matching" begin
  @present SchPopNet(FreeSchema) begin
    Agent::Ob; V::Ob; E::Ob
    src::Hom(E,V); tgt::Hom(E,V); loc::Hom(Agent,V)
  end
  @acset_type PopNet(SchPopNet)
  
  state = @acset PopNet begin
    Agent=3; V=3; E=6
    src=[1,2,3,2,3,1]; tgt=[2,3,1,1,2,3]
    loc=[1,2,3]
  end
  
  # Two adjacent agents
  pat = @acset PopNet begin Agent=2; V=2; E=1; loc=[1,2]; src=[1]; tgt=[2] end
  ms = homomorphisms(pat, state)
  @test length(ms) == 6  # 3 edges × 2 agent orderings
  
  # Two agents at same vertex (no matches)
  pat_same = @acset PopNet begin Agent=2; V=1; loc=[1,1] end
  ms_same = homomorphisms(pat_same, state; monic=true)
  @test length(ms_same) == 0
end


end # module
end # module
