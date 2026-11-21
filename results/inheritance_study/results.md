# Inherited neural policy study

Original energy-only PPO with independent founder networks, inherited family policies and explicit survivor-based restocking. No relatedness or donation bonus enters PPO.

Clean feasibility passed: **False**. This screen is not a kin-selection or convergence test.

Evaluation worlds are averaged within independent training seeds. Intervals below are exploratory Student t intervals; bounded metrics can have out-of-range intervals. Null mimic shares mean no post-invasion donations, not successful defense.

| Condition / control / evaluation | Final residents | Donations / resident step | Mean fraction of worlds surviving | Mimic donation share |
| --- | --- | --- | --- | --- |
| clean/rule_dyadic_tit_for_tat/clean | 44.4000 [43.2894, 45.5106]; n=5 | 0.6477 [0.6267, 0.6686]; n=5 | 1.0000 [1.0000, 1.0000]; n=5 | undefined |
| clean/rule_fixed_threshold/clean | 18.0000 [-12.6039, 48.6039]; n=5 | 0.3949 [-0.2765, 1.0663]; n=5 | 0.4000 [-0.2801, 1.0801]; n=5 | undefined |
| clean/rule_reputation/clean | 45.0000 [45.0000, 45.0000]; n=5 | 0.9855 [0.9829, 0.9881]; n=5 | 1.0000 [1.0000, 1.0000]; n=5 | undefined |
| clean/rule_selfish/clean | 0.0000 [0.0000, 0.0000]; n=5 | 0.0000 [0.0000, 0.0000]; n=5 | 0.0000 [0.0000, 0.0000]; n=5 | undefined |
| clean/rule_tag_trust/clean | 18.0000 [-12.6039, 48.6039]; n=5 | 0.3719 [-0.2604, 1.0043]; n=5 | 0.4000 [-0.2801, 1.0801]; n=5 | undefined |
| clean/trained/clean | 0.0000 [0.0000, 0.0000]; n=5 | 0.0745 [-0.0105, 0.1595]; n=5 | 0.0000 [0.0000, 0.0000]; n=5 | undefined |
| clean/untrained_ancestors/clean | 45.0000 [45.0000, 45.0000]; n=5 | 0.5357 [0.5074, 0.5641]; n=5 | 1.0000 [1.0000, 1.0000]; n=5 | undefined |
| evolution_only/trained/clean | 45.0000 [45.0000, 45.0000]; n=5 | 0.4844 [0.4457, 0.5231]; n=5 | 1.0000 [1.0000, 1.0000]; n=5 | undefined |
| evolution_only/untrained_ancestors/clean | 45.0000 [45.0000, 45.0000]; n=5 | 0.4844 [0.4457, 0.5231]; n=5 | 1.0000 [1.0000, 1.0000]; n=5 | undefined |
| fallback/trained/clean | 24.0000 [24.0000, 24.0000]; n=5 | 0.0064 [-0.0059, 0.0187]; n=5 | 1.0000 [1.0000, 1.0000]; n=5 | undefined |
| fallback/untrained_ancestors/clean | 24.0000 [24.0000, 24.0000]; n=5 | 0.3752 [0.3329, 0.4175]; n=5 | 1.0000 [1.0000, 1.0000]; n=5 | undefined |
| no_selection/trained/clean | 0.0000 [0.0000, 0.0000]; n=5 | 0.0764 [0.0610, 0.0918]; n=5 | 0.0000 [0.0000, 0.0000]; n=5 | undefined |
| no_selection/untrained_ancestors/clean | 45.0000 [45.0000, 45.0000]; n=5 | 0.5362 [0.5240, 0.5484]; n=5 | 1.0000 [1.0000, 1.0000]; n=5 | undefined |

Not run: fixed_gate, mlp, noise025, noise050, noise_mimics, fixed_noise_mimics, bc150, bc450, mutation000, mutation010, mutation020, random_placement.

The untrained-ancestor control uses the same evolved founder chromosomes and ancestor composition with their initial neural weights. It is not an independent evolutionary training run. The evolution-only condition is the separate control for selection without PPO.

Fallback survival is forced when starvation is disabled. Adjacent Hamming similarity needs its matched random-placement comparison before it supports assortment; tag preference is descriptive, not Hamiltonian or pedigree evidence.

See [raw analysis](analysis.json) and [protocol](../../docs/inherited_policy_protocol.md).
