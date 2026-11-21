# Verified frozen-policy evaluation

Worlds are averaged within training seed. Intervals use independent seed blocks; evaluation runtime can differ from training.

| Condition/control/evaluation | Residents (95% CI) | Donations/resident step (95% CI) |
| --- | --- | --- |
| bc150/trained/clean | 0.0000 [0.0000, 0.0000] | 0.2655 [0.2526, 0.2783] |
| bc150/untrained_ancestors/clean | 0.0000 [0.0000, 0.0000] | 0.3037 [0.2945, 0.3129] |
| bc450/trained/clean | 21.4000 [-5.1831, 47.9831] | 0.1382 [0.0648, 0.2117] |
| bc450/untrained_ancestors/clean | 45.0000 [45.0000, 45.0000] | 0.5218 [0.4735, 0.5702] |
| clean/rule_dyadic_tit_for_tat/clean | 44.4000 [43.2894, 45.5106] | 0.6477 [0.6267, 0.6686] |
| clean/rule_fixed_threshold/clean | 27.0000 [-3.6039, 57.6039] | 0.5928 [-0.0791, 1.2647] |
| clean/rule_reputation/clean | 45.0000 [45.0000, 45.0000] | 0.9855 [0.9829, 0.9881] |
| clean/rule_selfish/clean | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| clean/rule_tag_trust/clean | 27.0000 [-3.6039, 57.6039] | 0.5727 [-0.0766, 1.2220] |
| clean/trained/clean | 0.0000 [0.0000, 0.0000] | 0.1822 [0.1551, 0.2093] |
| clean/untrained_ancestors/clean | 45.0000 [45.0000, 45.0000] | 0.5236 [0.5120, 0.5352] |
| evolution_only/trained/clean | 45.0000 [45.0000, 45.0000] | 0.5056 [0.4827, 0.5285] |
| evolution_only/untrained_ancestors/clean | 45.0000 [45.0000, 45.0000] | 0.5056 [0.4827, 0.5285] |
| fallback/trained/clean | 24.0000 [24.0000, 24.0000] | 0.3015 [0.2352, 0.3678] |
| fallback/untrained_ancestors/clean | 24.0000 [24.0000, 24.0000] | 0.3761 [0.3352, 0.4170] |
| fixed_gate/trained/clean | 0.0000 [0.0000, 0.0000] | 0.1610 [0.1410, 0.1810] |
| fixed_gate/untrained_ancestors/clean | 45.0000 [45.0000, 45.0000] | 0.4982 [0.4729, 0.5235] |
| fixed_noise_mimics/trained/clean | 38.1333 [26.4502, 49.8164] | 0.3994 [0.3233, 0.4754] |
| fixed_noise_mimics/trained/attack | 0.4667 [-0.8290, 1.7623] | 0.3203 [0.2916, 0.3490] |
| fixed_noise_mimics/untrained_ancestors/clean | 45.0000 [45.0000, 45.0000] | 0.5258 [0.4991, 0.5524] |
| high_entropy/trained/clean | 0.0000 [0.0000, 0.0000] | 0.1851 [0.1618, 0.2085] |
| high_entropy/untrained_ancestors/clean | 45.0000 [45.0000, 45.0000] | 0.5302 [0.4964, 0.5639] |
| long_training/trained/clean | 0.0000 [0.0000, 0.0000] | 0.0745 [-0.0105, 0.1595] |
| long_training/untrained_ancestors/clean | 45.0000 [45.0000, 45.0000] | 0.5357 [0.5074, 0.5641] |
| low_lr/trained/clean | 45.0000 [45.0000, 45.0000] | 0.5000 [0.4578, 0.5422] |
| low_lr/untrained_ancestors/clean | 45.0000 [45.0000, 45.0000] | 0.5127 [0.4736, 0.5518] |
| mlp/trained/clean | 9.6000 [-8.5908, 27.7908] | 0.2432 [0.1643, 0.3220] |
| mlp/untrained_ancestors/clean | 45.0000 [45.0000, 45.0000] | 0.5441 [0.5166, 0.5716] |
| mutation000/trained/clean | 1.3333 [-0.7153, 3.3820] | 0.2138 [0.1714, 0.2562] |
| mutation000/untrained_ancestors/clean | 45.0000 [45.0000, 45.0000] | 0.5345 [0.5133, 0.5557] |
| mutation010/trained/clean | 0.0000 [0.0000, 0.0000] | 0.2075 [0.1986, 0.2164] |
| mutation010/untrained_ancestors/clean | 45.0000 [45.0000, 45.0000] | 0.5341 [0.5045, 0.5636] |
| mutation020/trained/clean | 0.3333 [-0.5921, 1.2588] | 0.1900 [0.1537, 0.2262] |
| mutation020/untrained_ancestors/clean | 44.5333 [43.2377, 45.8290] | 0.5007 [0.4589, 0.5424] |
| no_selection/trained/clean | 12.2667 [3.4897, 21.0436] | 0.3013 [0.2534, 0.3492] |
| no_selection/untrained_ancestors/clean | 45.0000 [45.0000, 45.0000] | 0.5362 [0.5240, 0.5484] |
| noise025/trained/clean | 0.0000 [0.0000, 0.0000] | 0.1833 [0.1664, 0.2001] |
| noise025/untrained_ancestors/clean | 45.0000 [45.0000, 45.0000] | 0.5228 [0.4926, 0.5530] |
| noise050/trained/clean | 1.2667 [-2.2502, 4.7835] | 0.1867 [0.1426, 0.2309] |
| noise050/untrained_ancestors/clean | 45.0000 [45.0000, 45.0000] | 0.5223 [0.4888, 0.5558] |
| noise_mimics/trained/clean | 32.5333 [9.4193, 55.6474] | 0.3926 [0.2593, 0.5259] |
| noise_mimics/trained/attack | 2.8000 [-2.1718, 7.7718] | 0.3350 [0.2527, 0.4173] |
| noise_mimics/untrained_ancestors/clean | 45.0000 [45.0000, 45.0000] | 0.5280 [0.4953, 0.5608] |
| random_placement/trained/clean | 0.0000 [0.0000, 0.0000] | 0.2011 [0.1799, 0.2223] |
| random_placement/untrained_ancestors/clean | 45.0000 [45.0000, 45.0000] | 0.5226 [0.5090, 0.5363] |
