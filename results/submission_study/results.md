# Inherited neural policy study

Original energy-only PPO with independent founder networks, inherited family policies and explicit survivor-based restocking. No relatedness or donation bonus enters PPO.

Clean feasibility passed: **False**. This screen is not a kin-selection or convergence test.

Evaluation worlds are averaged within independent training seeds. Intervals below are exploratory Student t intervals; bounded metrics can have out-of-range intervals. Null mimic shares mean no post-invasion donations, not successful defense.

| Condition / control / evaluation | Final residents | Donations / resident step | Mean fraction of worlds surviving | Mimic donation share |
| --- | --- | --- | --- | --- |
| bc150/trained/clean | 0.0000 [0.0000, 0.0000]; n=5 | 0.2730 [0.2587, 0.2872]; n=5 | 0.0000 [0.0000, 0.0000]; n=5 | undefined |
| bc150/untrained_ancestors/clean | 0.0000 [0.0000, 0.0000]; n=5 | 0.3157 [0.2997, 0.3317]; n=5 | 0.0000 [0.0000, 0.0000]; n=5 | undefined |
| bc450/trained/clean | 18.0000 [-12.6039, 48.6039]; n=5 | 0.1316 [0.0508, 0.2124]; n=5 | 0.4000 [-0.2801, 1.0801]; n=5 | undefined |
| bc450/untrained_ancestors/clean | 45.0000 [45.0000, 45.0000]; n=5 | 0.5252 [0.4768, 0.5736]; n=5 | 1.0000 [1.0000, 1.0000]; n=5 | undefined |
| clean/rule_dyadic_tit_for_tat/clean | 44.4000 [43.2894, 45.5106]; n=5 | 0.6477 [0.6267, 0.6686]; n=5 | 1.0000 [1.0000, 1.0000]; n=5 | undefined |
| clean/rule_fixed_threshold/clean | 27.0000 [-3.6039, 57.6039]; n=5 | 0.5928 [-0.0791, 1.2647]; n=5 | 0.6000 [-0.0801, 1.2801]; n=5 | undefined |
| clean/rule_reputation/clean | 45.0000 [45.0000, 45.0000]; n=5 | 0.9855 [0.9829, 0.9881]; n=5 | 1.0000 [1.0000, 1.0000]; n=5 | undefined |
| clean/rule_selfish/clean | 0.0000 [0.0000, 0.0000]; n=5 | 0.0000 [0.0000, 0.0000]; n=5 | 0.0000 [0.0000, 0.0000]; n=5 | undefined |
| clean/rule_tag_trust/clean | 27.0000 [-3.6039, 57.6039]; n=5 | 0.5727 [-0.0766, 1.2220]; n=5 | 0.6000 [-0.0801, 1.2801]; n=5 | undefined |
| clean/trained/clean | 0.0000 [0.0000, 0.0000]; n=5 | 0.1864 [0.1499, 0.2228]; n=5 | 0.0000 [0.0000, 0.0000]; n=5 | undefined |
| clean/untrained_ancestors/clean | 45.0000 [45.0000, 45.0000]; n=5 | 0.5283 [0.5190, 0.5376]; n=5 | 1.0000 [1.0000, 1.0000]; n=5 | undefined |
| evolution_only/trained/clean | 45.0000 [45.0000, 45.0000]; n=5 | 0.5066 [0.4846, 0.5286]; n=5 | 1.0000 [1.0000, 1.0000]; n=5 | undefined |
| evolution_only/untrained_ancestors/clean | 45.0000 [45.0000, 45.0000]; n=5 | 0.5066 [0.4846, 0.5286]; n=5 | 1.0000 [1.0000, 1.0000]; n=5 | undefined |
| fallback/trained/clean | 24.0000 [24.0000, 24.0000]; n=5 | 0.3186 [0.2723, 0.3648]; n=5 | 1.0000 [1.0000, 1.0000]; n=5 | undefined |
| fallback/untrained_ancestors/clean | 24.0000 [24.0000, 24.0000]; n=5 | 0.3880 [0.3476, 0.4284]; n=5 | 1.0000 [1.0000, 1.0000]; n=5 | undefined |
| fixed_gate/trained/clean | 0.0000 [0.0000, 0.0000]; n=5 | 0.1586 [0.1335, 0.1837]; n=5 | 0.0000 [0.0000, 0.0000]; n=5 | undefined |
| fixed_gate/untrained_ancestors/clean | 45.0000 [45.0000, 45.0000]; n=5 | 0.5014 [0.4749, 0.5278]; n=5 | 1.0000 [1.0000, 1.0000]; n=5 | undefined |
| fixed_noise_mimics/trained/clean | 36.8000 [23.8116, 49.7884]; n=5 | 0.3954 [0.3148, 0.4759]; n=5 | 1.0000 [1.0000, 1.0000]; n=5 | undefined |
| fixed_noise_mimics/trained/attack | 0.0000 [0.0000, 0.0000]; n=5 | 0.3257 [0.2995, 0.3519]; n=5 | 0.0000 [0.0000, 0.0000]; n=5 | 0.2732 [0.2242, 0.3222]; n=5 |
| fixed_noise_mimics/untrained_ancestors/clean | 45.0000 [45.0000, 45.0000]; n=5 | 0.5241 [0.4916, 0.5566]; n=5 | 1.0000 [1.0000, 1.0000]; n=5 | undefined |
| high_entropy/trained/clean | 0.0000 [0.0000, 0.0000]; n=5 | 0.1924 [0.1645, 0.2203]; n=5 | 0.0000 [0.0000, 0.0000]; n=5 | undefined |
| high_entropy/untrained_ancestors/clean | 45.0000 [45.0000, 45.0000]; n=5 | 0.5360 [0.5040, 0.5681]; n=5 | 1.0000 [1.0000, 1.0000]; n=5 | undefined |
| long_training/trained/clean | 0.0000 [0.0000, 0.0000]; n=5 | 0.0772 [-0.0096, 0.1640]; n=5 | 0.0000 [0.0000, 0.0000]; n=5 | undefined |
| long_training/untrained_ancestors/clean | 45.0000 [45.0000, 45.0000]; n=5 | 0.5353 [0.5147, 0.5559]; n=5 | 1.0000 [1.0000, 1.0000]; n=5 | undefined |
| low_lr/trained/clean | 45.0000 [45.0000, 45.0000]; n=5 | 0.5067 [0.4614, 0.5520]; n=5 | 1.0000 [1.0000, 1.0000]; n=5 | undefined |
| low_lr/untrained_ancestors/clean | 45.0000 [45.0000, 45.0000]; n=5 | 0.5164 [0.4756, 0.5571]; n=5 | 1.0000 [1.0000, 1.0000]; n=5 | undefined |
| mlp/trained/clean | 9.0667 [-5.7248, 23.8582]; n=5 | 0.2444 [0.1673, 0.3216]; n=5 | 0.6000 [0.0604, 1.1396]; n=5 | undefined |
| mlp/untrained_ancestors/clean | 45.0000 [45.0000, 45.0000]; n=5 | 0.5488 [0.5263, 0.5714]; n=5 | 1.0000 [1.0000, 1.0000]; n=5 | undefined |
| mutation000/trained/clean | 0.4000 [-0.3404, 1.1404]; n=5 | 0.2127 [0.1796, 0.2458]; n=5 | 0.1333 [-0.0934, 0.3600]; n=5 | undefined |
| mutation000/untrained_ancestors/clean | 45.0000 [45.0000, 45.0000]; n=5 | 0.5274 [0.5105, 0.5442]; n=5 | 1.0000 [1.0000, 1.0000]; n=5 | undefined |
| mutation010/trained/clean | 0.0000 [0.0000, 0.0000]; n=5 | 0.1937 [0.1830, 0.2044]; n=5 | 0.0000 [0.0000, 0.0000]; n=5 | undefined |
| mutation010/untrained_ancestors/clean | 45.0000 [45.0000, 45.0000]; n=5 | 0.5388 [0.5038, 0.5738]; n=5 | 1.0000 [1.0000, 1.0000]; n=5 | undefined |
| mutation020/trained/clean | 0.0000 [0.0000, 0.0000]; n=5 | 0.1826 [0.1570, 0.2083]; n=5 | 0.0000 [0.0000, 0.0000]; n=5 | undefined |
| mutation020/untrained_ancestors/clean | 45.0000 [45.0000, 45.0000]; n=5 | 0.5045 [0.4596, 0.5494]; n=5 | 1.0000 [1.0000, 1.0000]; n=5 | undefined |
| no_selection/trained/clean | 12.4000 [2.3620, 22.4380]; n=5 | 0.3042 [0.2560, 0.3524]; n=5 | 0.8667 [0.4965, 1.2369]; n=5 | undefined |
| no_selection/untrained_ancestors/clean | 45.0000 [45.0000, 45.0000]; n=5 | 0.5381 [0.5282, 0.5481]; n=5 | 1.0000 [1.0000, 1.0000]; n=5 | undefined |
| noise025/trained/clean | 0.0000 [0.0000, 0.0000]; n=5 | 0.1862 [0.1802, 0.1922]; n=5 | 0.0000 [0.0000, 0.0000]; n=5 | undefined |
| noise025/untrained_ancestors/clean | 45.0000 [45.0000, 45.0000]; n=5 | 0.5192 [0.4810, 0.5574]; n=5 | 1.0000 [1.0000, 1.0000]; n=5 | undefined |
| noise050/trained/clean | 0.5333 [-0.9474, 2.0141]; n=5 | 0.1863 [0.1497, 0.2229]; n=5 | 0.1333 [-0.2369, 0.5035]; n=5 | undefined |
| noise050/untrained_ancestors/clean | 45.0000 [45.0000, 45.0000]; n=5 | 0.5240 [0.4910, 0.5569]; n=5 | 1.0000 [1.0000, 1.0000]; n=5 | undefined |
| noise_mimics/trained/clean | 34.8000 [10.4305, 59.1695]; n=5 | 0.3976 [0.2593, 0.5358]; n=5 | 0.8000 [0.2447, 1.3553]; n=5 | undefined |
| noise_mimics/trained/attack | 1.2000 [-1.6913, 4.0913]; n=5 | 0.3421 [0.2797, 0.4046]; n=5 | 0.2000 [-0.1702, 0.5702]; n=5 | 0.2758 [0.2278, 0.3239]; n=5 |
| noise_mimics/untrained_ancestors/clean | 45.0000 [45.0000, 45.0000]; n=5 | 0.5330 [0.4988, 0.5671]; n=5 | 1.0000 [1.0000, 1.0000]; n=5 | undefined |
| random_placement/trained/clean | 0.0000 [0.0000, 0.0000]; n=5 | 0.1940 [0.1801, 0.2079]; n=5 | 0.0000 [0.0000, 0.0000]; n=5 | undefined |
| random_placement/untrained_ancestors/clean | 45.0000 [45.0000, 45.0000]; n=5 | 0.5243 [0.5104, 0.5383]; n=5 | 1.0000 [1.0000, 1.0000]; n=5 | undefined |

Not run: none.

All-condition runs are exploratory, including when the clean feasibility screen fails. Long training uses four times the base budget; low learning rate and high entropy each change one optimizer setting. None changes the energy-only reward.

The untrained-ancestor control uses the same evolved founder chromosomes and ancestor composition with their initial neural weights. It is not an independent evolutionary training run. The evolution-only condition is the separate control for selection without PPO.

Fallback survival is forced when starvation is disabled. Adjacent Hamming similarity needs its matched random-placement comparison before it supports assortment; tag preference is descriptive, not Hamiltonian or pedigree evidence.

See [raw analysis](analysis.json) and [protocol](../../docs/final_report.md).
