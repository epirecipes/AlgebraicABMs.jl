# # Contact Rate Semantics in AlgebraicABMs
#
# In epidemiological modeling, the concept of a **contact rate** creates a 
# subtle challenge for agent-based models built on pattern matching. This 
# example explores the problem and demonstrates solutions available in 
# AlgebraicABMs.
#
# ## The Problem
#
# In a standard SIR model, the infection process involves a susceptible (S)
# meeting an infected (I) person. A naive implementation would create one timer 
# per (S, I) pair, giving a total rate proportional to |S| × |I|. But 
# epidemiologists often think of a **per-capita contact rate**: each infected 
# person attempts to infect *someone* at a fixed rate, regardless of how many 
# susceptible individuals exist. This distinction matters enormously when 
# population sizes are large.
#
# AlgebraicABMs provides two approaches:
# 1. **The `basis` parameter** — decouple timing from the full pattern
# 2. **Explicit contact formation** — model contact and infection as separate 
#    processes (following @wwaites's suggestion on 
#    [Issue #39](https://github.com/AlgebraicJulia/AlgebraicABMs.jl/issues/39))
#
# ## Setup

using AlgebraicABMs, Catlab, AlgebraicRewriting
ENV["JULIA_DEBUG"] = "";

# ## Approach 1: Using `basis` for Per-Capita Rates
#
# The `basis` parameter lets us separate *what triggers the timer* from *what 
# the rewrite rule needs*. We specify a morphism `B ↣ L` where `B` is the 
# "basis" (determines timer count) and `L` is the full pattern.
#
# ### Schema: Simple SI model (no recovery, for clarity of rate comparison)

@present SchSI(FreeSchema) begin
  (S, I)::Ob
end

@acset_type SI(SchSI)

# ### Naive approach: one timer per (S, I) pair
# 
# Without a basis, each match of the pattern (one S + one I) gets its own 
# timer. With `DiscreteHazard(1)`, every (S, I) pair fires at t=1. With 20 
# susceptibles and 1 infected, all 20 fire simultaneously — mass-action 
# kinetics where the rate scales with |S| × |I|.

i   = @acset SI begin I=1 end 
si  = @acset SI begin S=1; I=1 end 
ii  = @acset SI begin I=2 end 

# Infection: S + I → I + I
inf_L = homomorphism(i, si)   # I ↣ (S + I)
inf_R = homomorphism(i, ii; any=true)   # I ↣ (I + I)
inf_rule = Rule(inf_L, inf_R)

naive_rule = ABMRule(inf_rule, DiscreteHazard(1); name=:inf_naive)
init = @acset SI begin S=20; I=1 end

println("=== Naive (no basis): rate ∝ |S| × |I| ===")
traj_naive = run!(ABM([naive_rule]), init; maxevent=25,
                  save=x -> (S=nparts(x,:S), I=nparts(x,:I)))
for (t, _, name, counts) in traj_naive.events
  println("  t=$(round(t, digits=4)): $name → S=$(counts.S), I=$(counts.I)")
end

# With 20 susceptibles and 1 infected, all 20 (S,I) pair timers fire at t=1.
# Every susceptible is infected in a single timestep — classic mass-action.

# ### Basis approach: one timer per infected person
#
# With `basis = homomorphism(I, S+I)`, we get one timer per infected person. 
# When it fires, a susceptible is chosen uniformly at random to complete the 
# match.
#
# This gives the epidemiologically standard "per-capita contact rate": each 
# infected person contacts others at a fixed rate, independent of population 
# size.

basis_hom = homomorphism(i, si)  # basis: just the infected person
basis_rule = ABMRule(inf_rule, DiscreteHazard(1); basis=basis_hom, name=:inf_basis)

# Use a large population so simultaneous basis events always find valid matches
println("\n=== Basis (per-capita): rate ∝ |I| ===")
traj_basis = run!(ABM([basis_rule]), @acset(SI, begin S=10000; I=1 end); maxevent=3,
                  save=x -> (S=nparts(x,:S), I=nparts(x,:I)))
for (t, _, name, counts) in traj_basis.events
  println("  t=$(round(t, digits=4)): $name → S=$(counts.S), I=$(counts.I)")
end

# With the basis approach:
# - At t=0, there is 1 infected → 1 timer → first infection at t=1
# - At t=1, there are 2 infected → 2 timers → 2 infections at t=2
# - At t=2, there are 4 infected → 4 timers → 4 infections at t=3
# This gives clean exponential growth: 1, 2, 4, 8, ... independent of |S|.

# ## Approach 2: Explicit Contact Formation
#
# An alternative, advocated by @wwaites, is to model contact and transmission 
# as separate processes. This is more explicit and avoids the "contact rate" 
# abstraction entirely.
#
# The idea: individuals form and break contact edges dynamically. Infection 
# can only occur along an existing contact edge. This naturally captures:
# - Network structure (who meets whom)
# - Duration of contact (edge lifetime)
# - Location-based mixing (edges form within locations)
#
# ### Schema: Individuals with contact edges

@present SchSIContact(FreeSchema) begin
  (S, I)::Ob
  Contact::Ob
  s::Hom(Contact, S)
  i::Hom(Contact, I)
end

@acset_type SIContact(SchSIContact)

# ### Contact formation and breaking
# 
# Contacts form between S and I individuals at some mixing rate. 
# When a contact exists, infection can occur along that edge.

# Form an S-I contact: unlinked S and I gain a Contact edge
si_free = @acset SIContact begin S=1; I=1 end
si_linked = @acset SIContact begin S=1; I=1; Contact=1; s=1; i=1 end

i_contact = @acset SIContact begin I=1 end
form_rule = Rule(id(si_free), homomorphism(si_free, si_linked))
form_abm = ABMRule(form_rule, ContinuousHazard(5); name=:form_contact)

# Break a contact
break_rule = Rule(homomorphism(si_free, si_linked), id(si_free))
break_abm = ABMRule(break_rule, ContinuousHazard(48); name=:break_contact)

# Infect along a contact edge: S becomes I, Contact consumed.
# Interface is a single I (the original infected person, preserved).
# Left side: Contact(S, I). Right side: I + I (two unlinked infected).
inf_contact_L = homomorphism(i_contact, si_linked)
ii_free = @acset SIContact begin I=2 end
inf_contact_R = homomorphism(i_contact, ii_free; any=true)
inf_contact = Rule(inf_contact_L, inf_contact_R)
inf_contact_abm = ABMRule(inf_contact, ContinuousHazard(1); name=:infect_contact)

println("\n=== Explicit contact model ===")
println("Schema objects: S, I, Contact (with homs s: Contact→S, i: Contact→I)")
println("Contact formation rate: 5.0 per (S,I) pair")
println("Contact breaking rate: 48.0 per existing contact")
println("Infection rate: 1.0 per existing S-I contact")

# Run a small example
init_contact = @acset SIContact begin S=50; I=2 end
traj_contact = run!(ABM([form_abm, break_abm, inf_contact_abm]), init_contact;
                    maxevent=30,
                    save=x -> (S=nparts(x,:S), I=nparts(x,:I), C=nparts(x,:Contact)))
println("\nSimulation trace:")
for (t, _, name, counts) in traj_contact.events
  println("  t=$(round(t, digits=4)): $name → S=$(counts.S), I=$(counts.I), contacts=$(counts.C)")
end

# ### Advantages of explicit contacts
# 
# This approach has several benefits:
#
# 1. **Compositional**: contact structure and disease dynamics are modular — 
#    you can swap in different contact patterns (household, workplace, school)
#    without changing the disease rules.
#
# 2. **Mechanistic**: the model explicitly represents *who is in contact with 
#    whom*, enabling network-level analyses.
#
# 3. **Flexible**: different types of contacts (close vs. casual) can have 
#    different transmission probabilities.
#
# The main downside is computational cost: many contact events may occur 
# that don't lead to infection. Future optimizations (e.g., observable-preserving 
# schema migrations) could help reduce this overhead.

# ## Summary
#
# | Approach | Rate semantics | Complexity | Best for |
# |----------|---------------|------------|----------|
# | No basis | Rate ∝ \|S\| × \|I\| | Simple | Mass-action kinetics |
# | `basis` | Rate ∝ \|I\| (per-capita) | Simple | Standard epi contact rates |
# | Explicit contacts | Mechanistic | Higher | Network models, spatial mixing |
#
# The `basis` approach is recommended for most epidemiological models where 
# per-capita contact rates are desired. Explicit contact formation is 
# preferred when network structure matters or when contacts have meaningful 
# duration and location.
