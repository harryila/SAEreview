# Latent Escape target-domain labeling guide

Guide ID: `latent-escape-domain-labeling-v1`

This guide defines the frozen interpretation of the 19 target-domain labels used
by Latent Escape. It applies to the automated classifier, the blinded manual
audit, and all later condition-level domain labeling. It does not change the
base taxonomy in `protocol.json`.

## What to label

Assign exactly one label to the **generated analogy's target system**. Use the
target system, target roles, and target-side causal mechanism. Do not classify
the source mechanism, writing style, application area mentioned only in passing,
or the model's self-reported `target_domain` field.

Raters and classifiers must not receive the prompt split, experimental
condition, seed, feature ID, intervention metadata, or automated prediction.
The generated `target_domain` value is removed before labeling.

## Decision rule

1. Identify the system that receives the analogy mapping.
2. Identify the domain that best describes that system's central actors and
   causal mechanism.
3. Prefer the most specific applicable frozen category over a broader context.
4. If several domains are genuinely involved, choose the one carrying the
   majority of mapped roles and relations. Use the decisive causal mechanism as
   the tie-breaker.
5. Use `other` only when no named category fits, or when the output lacks enough
   target-side content to identify a domain. Uncertainty by itself is not a
   reason to choose `other`.

Do not repair, complete, or improve an analogy while labeling it. A malformed or
weak analogy can still receive a named domain when its intended target system is
clear. Structural quality is rated separately.

## Frozen taxonomy

- `biology/ecology`: organisms, populations, evolution, food webs, habitats,
  ecosystems, or ecological interactions.
- `medicine/public health`: patients, diagnosis, treatment, disease,
  epidemiology, hospitals, or population-health interventions.
- `physics`: matter, forces, energy, fields, particles, waves, thermodynamics,
  or other physical systems whose central mechanism is physical law.
- `chemistry/materials`: molecules, reactions, catalysis, polymers, alloys,
  material composition, or material properties driven by chemical structure.
- `engineering/control`: designed physical systems, feedback control, sensors,
  actuators, stability, manufacturing, or engineered infrastructure.
- `computer science/software`: algorithms, programs, databases, operating
  systems, distributed systems, cybersecurity, or software protocols.
- `AI/neural networks`: machine learning, model training, inference, neural
  representations, artificial agents, or AI-specific architectures.
- `economics/markets`: prices, incentives, trade, firms as market actors,
  supply and demand, allocation, or financial markets.
- `organizations/governance`: organizations, management, bureaucracies,
  corporate structure, collective decision procedures, or institutional
  governance not primarily framed as law or markets.
- `sociology/culture`: social groups, norms, identity, culture, communities,
  social networks, or society-level interaction.
- `psychology/cognition`: perception, memory, attention, belief, emotion,
  individual decision-making, or other mental processes.
- `education/learning`: students, teachers, curricula, classrooms, pedagogy,
  assessment, or educational institutions centered on learning.
- `law/policy`: courts, statutes, legal rights, regulation, legislation, public
  policy, or formal rule enforcement.
- `history`: historical periods, empires, wars, dynasties, or a historical
  process presented primarily as such rather than as a generic social system.
- `arts/literature`: music, visual art, fiction, poetry, theater, film, artistic
  production, or interpretation of creative works.
- `sports/games`: sports, teams as competitors, game mechanics, strategy games,
  tournaments, or rule-governed play.
- `geography/earth/environment`: climate, geology, rivers, landscapes, weather,
  Earth systems, environmental processes, or spatial geography.
- `everyday/household`: domestic routines, cooking, cleaning, shopping,
  household objects, family logistics, or ordinary personal activities not
  better captured above.
- `other`: a residual category for a discernible target outside every named
  category, or for an output with no classifiable target system.

## Boundary examples

- A neural network used to diagnose patients is `AI/neural networks` when the
  mapped mechanism is model training or inference; it is
  `medicine/public health` when the mapped mechanism is clinical triage or care.
- A company analogy is `economics/markets` when prices and exchange drive the
  mapping; it is `organizations/governance` when hierarchy and internal
  coordination drive it.
- Climate regulation enacted by a legislature is `law/policy` when the mapping
  centers on rules and enforcement; it is `geography/earth/environment` when the
  mapped mechanism is the climate system itself.
- A software simulation of an ecosystem is `computer science/software` when
  program components carry the mapping and `biology/ecology` when organisms and
  ecological relations carry it.

## Audit and analysis policy

The blinded manual audit independently labels the deterministic audit queue
using this guide. Adjudicated labels replace automated labels only for audited
records, as specified by the base protocol.

`other` remains part of taxonomy coverage, audit summaries, domain-output-rate
reporting, entropy, and distinct-domain calculations. Protocol Amendment 4
excludes it only from eligibility as the primary selected causal domain because
it is a heterogeneous residual rather than a single interpretable intervention
target.
