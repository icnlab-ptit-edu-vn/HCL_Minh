# Journal Direction 1: Overhead-Aware Adaptive Reconfiguration for Clustered WSNs

## 1. Core scientific question

The current repository already contains a useful phenomenon: frequent topology refresh can delay the mid-life and late-life collapse of the network, but it is not free. Re-clustering consumes control energy, changes leaders, and may accelerate early depletion if it is executed too aggressively. The journal-worthy question is therefore not *which optimizer chooses better leaders once*, but rather:

**How should a clustered WSN decide when to keep the current topology and when to rebuild it, once reconfiguration overhead is explicitly accounted for?**

This framing upgrades the work from an optimizer-comparison paper to a policy-design paper.

## 2. Why this direction is stronger than adding another optimizer

A new metaheuristic alone is weak because it usually only changes the search routine inside cluster-head selection. That type of contribution is often incremental and does not explain the system-level trade-off. In contrast, adaptive reconfiguration creates a stronger scientific story:

- static topology becomes stale as node energy becomes imbalanced,
- frequent re-clustering improves balance but consumes additional control energy,
- the best strategy depends on the network age, residual-energy distribution, and topology imbalance.

This lets the paper claim a concrete system finding: **the optimal reconfiguration policy is regime-dependent, and explicit control-overhead modeling changes the lifetime ranking of clustering strategies.**

## 3. Recommended paper framing

A suitable title family is:

- *Overhead-Aware Adaptive Reconfiguration for Energy-Balanced Wireless Sensor Networks*
- *Learning When to Recluster: An Overhead-Aware Topology Control Policy for Clustered WSNs*
- *Lifetime Shaping in Clustered WSNs via Overhead-Aware Adaptive Reconfiguration*

## 4. Research questions

### RQ1
When control overhead is modeled explicitly, does always re-clustering still remain competitive against adaptive or static policies?

### RQ2
Can a policy that jointly chooses topology refresh and cluster count improve the **lifetime trajectory** rather than only a single milestone?

### RQ3
Under which operating regimes does adaptive reconfiguration help most?

Examples:
- dense vs. sparse deployments,
- near vs. far base station,
- high vs. low initial energy,
- low vs. high control overhead budget.

## 5. Proposed method in this repo

The revised repo now implements the method as follows.

### Action design
The discrete action is now consistent across training and evaluation:

- `0`: keep current topology,
- `1..K`: rebuild topology with cluster count in `[MIN_CLUSTERS, MAX_CLUSTERS]`.


### State design
The RL state now exposes:

- residual-energy statistics of alive nodes,
- leader-energy statistics,
- recent per-round energy consumption,
- alive ratio,
- current cluster count,
- topology age,
- cluster-size imbalance,
- normalized last reconfiguration overhead,
- a flag indicating whether the previous round re-clustered.

This is important because the agent should not re-cluster blindly. It must react to both energy imbalance and the age of the current topology.

### Explicit reconfiguration overhead
Every topology rebuild now incurs a control-energy cost:

- each alive node sends a short status packet,
- each alive node receives a short topology/configuration packet.

That overhead is charged to node energy and added to the round energy budget. Therefore, the policy is trained and evaluated under the correct trade-off instead of receiving a hidden free operation.

## 6. What was wrong in the original repo

The main methodological inconsistency was this:

- during training, the discrete action was effectively interpreted as a binary flag (`delta == 1` means re-cluster, otherwise do nothing),
- during testing, the same action was interpreted as `num_clusters = delta + 3`.

This means the trained policy was not being evaluated under the same semantics it had seen during optimization. The revised repo fixes this by introducing a single action mapping used consistently in both training and evaluation.

## 7. What the revised repo now supports

### Environment changes
- topology age tracking,
- reconfiguration count tracking,
- explicit control-overhead accounting,
- consistent topology-control step interface,
- alive-only LEACH-C setup for fairer comparison,
- state vector expanded to match the adaptive reconfiguration problem.

### Training changes
- consistent action mapping,
- training on the new state representation,
- bootstrap initialization with an initial topology,
- logging of reconfiguration behavior.

### Evaluation changes
The new evaluation setup compares:

- `ADAPTIVE`: learned overhead-aware topology-control policy,
- `STATIC`: no re-clustering after initialization,
- `PERIODIC`: re-cluster every fixed interval,
- `ALWAYS_REFRESH`: re-cluster every round.

This is much closer to the actual journal question.

## 8. Minimum experiments for the journal paper

### A. Main comparison
Compare the four methods above on:
- FND,
- HND,
- LND,
- cumulative energy consumption,
- cumulative reconfiguration energy,
- Jain fairness over residual energy,
- alive-node trajectory over time.

### B. Ablation 1: remove overhead model
Train/evaluate the same adaptive policy without charging reconfiguration cost. This directly shows whether the overhead model changes the policy and the ranking.

### C. Ablation 2: fixed cluster count vs. adaptive cluster count
Test whether the gain comes mainly from *when to rebuild* or also from *how many clusters to use*.

### D. Ablation 3: topology age signal
Remove topology-age features from the state and observe whether performance degrades. This tests whether the policy is truly learning a refresh strategy.

### E. Sensitivity study
Vary:
- base-station distance,
- number of nodes,
- initial energy,
- control packet size.

This gives the paper deployment relevance.

## 9. Expected claim style for the paper

Avoid claiming that the policy is universally best. The stronger and more defensible claim is:

> Explicitly modeling reconfiguration overhead reveals that topology control in clustered WSNs is a trade-off problem. Static policies suffer from stale energy imbalance, whereas always-refresh policies waste control energy. The proposed adaptive policy learns a regime-dependent refresh behavior that reshapes the lifetime trajectory and reduces unnecessary reconfiguration.

That is a journal-level claim because it is a system insight, not just a leaderboard statement.

## 10. What to write in the Results section

Use the pattern:

1. **Observation**: the adaptive policy triggers fewer re-clustering operations than always-refresh baselines.
2. **Comparison**: despite fewer reconfigurations, it preserves or improves HND/LND.
3. **Explanation**: the policy spends control energy only when imbalance and topology staleness justify it.
4. **Implication**: clustered WSN lifetime should be designed as a dynamic control problem rather than a one-shot cluster-head optimization problem.

## 11. Immediate next coding steps after this repo patch

1. Train the new agent with `train.py`.
2. Evaluate with `test3.py`.
3. Add one more baseline with periodic refresh every `K` rounds.
4. Run sensitivity sweeps over control packet size and BS distance.
5. Export all milestone tables and trajectory plots for the paper.
