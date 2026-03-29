# # SIR Model on a Network
#
# This example demonstrates how to use `networkify` to put agents on a contact
# network, where rewrite rules are constrained by network adjacency.
#
# ## Setup
#
# We model SIR using separate object types for each compartment (S, I, R),
# each with a location hom to the network's vertices.

using AlgebraicABMs, Catlab, AlgebraicRewriting

# ## 1. Define base schema and networkify
#
# Start with a simple SIR schema (three compartments, no network structure):

@present SchSIR(FreeSchema) begin
  S::Ob; I::Ob; R::Ob
end

# `networkify` extends this with graph structure (V, E, src, tgt) and
# location morphisms (loc_S, loc_I, loc_R):

SchSIRNet = networkify(SchSIR)

# The networkified schema has 5 objects and 5 homs:

println("Objects: ", [first(g) for g in generators(SchSIRNet, :Ob)])
println("Homs: ", [first(g) for g in generators(SchSIRNet, :Hom)])

# Create a concrete ACSet type from the networkified schema:

@acset_type SIRNet(SchSIRNet)

# ## 2. Define rewrite rules
#
# ### Infection rule
# An infected person (I) at vertex v₂ infects an adjacent susceptible (S) at
# vertex v₁. The edge in the pattern enforces the adjacency constraint.

L_inf = @acset SIRNet begin
  S=1; I=1; V=2; E=1; loc_S=[1]; loc_I=[2]; src=[1]; tgt=[2]
end
K_inf = @acset SIRNet begin
  I=1; V=2; E=1; loc_I=[2]; src=[1]; tgt=[2]
end
R_inf = @acset SIRNet begin
  I=2; V=2; E=1; loc_I=[1,2]; src=[1]; tgt=[2]
end

l_inf = ACSetTransformation(K_inf, L_inf; I=[1], V=[1,2], E=[1])
r_inf = ACSetTransformation(K_inf, R_inf; I=[2], V=[1,2], E=[1])
rule_inf = Rule(l_inf, r_inf)

# ### Recovery rule
# An infected person recovers (no network constraint needed).

L_rec = @acset SIRNet begin I=1; V=1; loc_I=[1] end
K_rec = @acset SIRNet begin V=1 end
R_rec = @acset SIRNet begin R=1; V=1; loc_R=[1] end

l_rec = ACSetTransformation(K_rec, L_rec; V=[1])
r_rec = ACSetTransformation(K_rec, R_rec; V=[1])
rule_rec = Rule(l_rec, r_rec)

# ## 3. Create initial state on a network
#
# A ring graph of 10 vertices with bidirectional edges.
# Person 1 is infected at vertex 1; persons 2-10 are susceptible.

nv = 10
init = SIRNet()
add_parts!(init, :V, nv)
for i in 1:nv
  j = i % nv + 1
  add_parts!(init, :E, 2; src=[i, j], tgt=[j, i])
end
add_parts!(init, :I, 1; loc_I=[1])
add_parts!(init, :S, nv - 1; loc_S=collect(2:nv))

println("Initial state: S=$(nparts(init,:S)) I=$(nparts(init,:I)) R=$(nparts(init,:R))")

# ## 4. Verify adjacency constraints
#
# The infection rule should only match adjacent S-I pairs:

matches = collect(get_matches(rule_inf, init))
println("Infection matches: $(length(matches))")

# Only the S person at vertex 2 and vertex 10 (adjacent to vertex 1 where I
# is) should match. With bidirectional edges, we expect matches for both
# directions.

# ## 5. Distance computation
#
# `shortest_distance` computes BFS shortest path between vertices:

for v in [1, 2, 5, 6]
  d = shortest_distance(init, 1, v)
  println("  Distance from V1 to V$v: $d")
end

# ## Notes
#
# The `networkify` function works on both `Presentation` and `BasicSchema`
# objects, making it easy to extend any existing schema with network structure.
# The adjacency constraint is enforced by including edges in the rule pattern —
# no special constraint syntax is needed.
#
# For models requiring attribute changes (e.g., integer status labels), use
# `AlgebraicRewriting`'s `expr` mechanism or the `ABMSchedule` type with
# rewrite schedules.
