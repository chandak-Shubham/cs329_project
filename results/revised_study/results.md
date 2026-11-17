# Revised study: executed results

All intervals and comparisons are exploratory. See protocol and limitations in docs/review_response_2026-11-17.md.

Values are means [95% intervals]. Survival uses Wilson intervals; other metrics use Student t intervals across training seeds.

| Condition | Resident survival | Final residents | Donations / resident step | Post-invasion mimic donation share |
| --- | --- | --- | --- | --- |
| demographic_clean/clean | 0.0% [0.0%, 43.4%]; n=5 | 0.0000 [0.0000, 0.0000]; n=5 | 0.0000 [0.0000, 0.0000]; n=5 | not evaluated |
| demographic_clean/attack | 0.0% [0.0%, 43.4%]; n=5 | 0.0000 [0.0000, 0.0000]; n=5 | 0.0000 [0.0000, 0.0000]; n=5 | not evaluated |
| demographic_clean_untrained/clean | 100.0% [56.6%, 100.0%]; n=5 | 45.0000 [45.0000, 45.0000]; n=5 | 0.5524 [0.4953, 0.6096]; n=5 | not evaluated |
| demographic_noise025/clean | 0.0% [0.0%, 43.4%]; n=5 | 0.0000 [0.0000, 0.0000]; n=5 | 0.0000 [0.0000, 0.0000]; n=5 | not evaluated |
| demographic_noise025/attack | 0.0% [0.0%, 43.4%]; n=5 | 0.0000 [0.0000, 0.0000]; n=5 | 0.0002 [-0.0004, 0.0009]; n=5 | 0.0% [CI unavailable]; n=1 |
| demographic_noise025_untrained/clean | 100.0% [56.6%, 100.0%]; n=5 | 45.0000 [45.0000, 45.0000]; n=5 | 0.5526 [0.4908, 0.6145]; n=5 | not evaluated |
| demographic_noise050/clean | 0.0% [0.0%, 43.4%]; n=5 | 0.0000 [0.0000, 0.0000]; n=5 | 0.0000 [0.0000, 0.0000]; n=5 | not evaluated |
| demographic_noise050/attack | 0.0% [0.0%, 43.4%]; n=5 | 0.0000 [0.0000, 0.0000]; n=5 | 0.0000 [0.0000, 0.0000]; n=5 | not evaluated |
| demographic_noise050_untrained/clean | 100.0% [56.6%, 100.0%]; n=5 | 45.0000 [45.0000, 45.0000]; n=5 | 0.5540 [0.4953, 0.6126]; n=5 | not evaluated |
| demographic_noise_mimics/clean | 0.0% [0.0%, 43.4%]; n=5 | 0.0000 [0.0000, 0.0000]; n=5 | 0.0005 [-0.0008, 0.0017]; n=5 | not evaluated |
| demographic_noise_mimics/attack | 0.0% [0.0%, 43.4%]; n=5 | 0.0000 [0.0000, 0.0000]; n=5 | 0.0002 [-0.0004, 0.0009]; n=5 | not evaluated |
| demographic_noise_mimics_untrained/clean | 100.0% [56.6%, 100.0%]; n=5 | 45.0000 [45.0000, 45.0000]; n=5 | 0.5540 [0.4953, 0.6126]; n=5 | not evaluated |
| energy_clean/clean | 0.0% [0.0%, 43.4%]; n=5 | 0.0000 [0.0000, 0.0000]; n=5 | 0.0002 [-0.0004, 0.0008]; n=5 | not evaluated |
| energy_clean/attack | 0.0% [0.0%, 43.4%]; n=5 | 0.0000 [0.0000, 0.0000]; n=5 | 0.0002 [-0.0004, 0.0009]; n=5 | not evaluated |
| energy_clean_untrained/clean | 100.0% [56.6%, 100.0%]; n=5 | 45.0000 [45.0000, 45.0000]; n=5 | 0.5524 [0.4953, 0.6096]; n=5 | not evaluated |
| fallback/clean | 100.0% [56.6%, 100.0%]; n=5 | 24.0000 [24.0000, 24.0000]; n=5 | 0.0001 [-0.0000, 0.0002]; n=5 | not evaluated |
| fallback/attack | 100.0% [56.6%, 100.0%]; n=5 | 24.0000 [24.0000, 24.0000]; n=5 | 0.0001 [-0.0000, 0.0002]; n=5 | 50.0% [-585.3%, 685.3%]; n=2 |
| fallback_untrained/clean | 100.0% [56.6%, 100.0%]; n=5 | 24.0000 [24.0000, 24.0000]; n=5 | 0.4024 [0.3427, 0.4621]; n=5 | not evaluated |
| fixed_noise_mimics/clean | 0.0% [0.0%, 43.4%]; n=5 | 0.0000 [0.0000, 0.0000]; n=5 | 0.0000 [0.0000, 0.0000]; n=5 | not evaluated |
| fixed_noise_mimics/attack | 0.0% [0.0%, 43.4%]; n=5 | 0.0000 [0.0000, 0.0000]; n=5 | 0.0000 [0.0000, 0.0000]; n=5 | not evaluated |
| fixed_noise_mimics_untrained/clean | 100.0% [56.6%, 100.0%]; n=5 | 45.0000 [45.0000, 45.0000]; n=5 | 0.5538 [0.4950, 0.6127]; n=5 | not evaluated |

Fallback survival is forced by disabling starvation and cannot support ecological conclusions. Small-sample t intervals for bounded quantities may extend outside their physical range; raw values and denominators are retained.

## Cue sensitivity on identical inputs

| Training condition | Genetic cue TV [95% CI] | Behaviour cue TV [95% CI] |
| --- | --- | --- |
| energy_clean | 0.0451 [0.0324, 0.0577]; n=5 | 0.0602 [0.0435, 0.0770]; n=5 |
| demographic_clean | 0.0501 [-0.0022, 0.1024]; n=5 | 0.0604 [-0.0223, 0.1431]; n=5 |
| demographic_noise025 | 0.0292 [-0.0002, 0.0586]; n=5 | 0.0516 [0.0222, 0.0811]; n=5 |
| demographic_noise050 | 0.0280 [-0.0035, 0.0596]; n=5 | 0.0275 [0.0028, 0.0522]; n=5 |
| demographic_noise_mimics | 0.0217 [-0.0029, 0.0462]; n=5 | 0.0365 [-0.0051, 0.0781]; n=5 |
| fixed_noise_mimics | 0.0173 [-0.0022, 0.0367]; n=5 | 0.0491 [-0.0118, 0.1099]; n=5 |
| fallback | 0.0105 [0.0029, 0.0182]; n=5 | 0.0075 [0.0044, 0.0107]; n=5 |

Raw seed trajectories, exposure-conditioned mimic shares, cue interventions, mutation sweeps and paired contrasts: [analysis.json](analysis.json).

Figures: [training curves](figures/learning_curves.png), [common-input cue sensitivity](figures/cue_sensitivity.png).
