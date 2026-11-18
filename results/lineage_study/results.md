# Lineage-fitness follow-up

Clean feasibility failed. Noise, mimic and fixed-gate follow-ups were not run. This remains a negative result.


All intervals and comparisons are exploratory. See protocol and limitations in ../../docs/final_review_response_2026-11-18.md.

Values are means [95% intervals]. Survival uses Wilson intervals; other metrics use Student t intervals across training seeds.

| Condition | Resident survival | Final residents | Donations / resident step | Post-invasion mimic donation share |
| --- | --- | --- | --- | --- |
| lineage_clean/clean | 0.0% [0.0%, 43.4%]; n=5 | 0.0000 [0.0000, 0.0000]; n=5 | 0.0000 [0.0000, 0.0000]; n=5 | not evaluated |
| lineage_clean_untrained/clean | 100.0% [56.6%, 100.0%]; n=5 | 45.0000 [45.0000, 45.0000]; n=5 | 0.5497 [0.4966, 0.6028]; n=5 | not evaluated |

Small-sample t intervals for bounded quantities may extend outside their physical range; raw values and denominators are retained.

## Cue sensitivity on identical inputs

| Training condition | Genetic cue TV [95% CI] | Behaviour cue TV [95% CI] |
| --- | --- | --- |
| lineage_clean | 0.0690 [-0.0262, 0.1642]; n=5 | 0.0607 [0.0152, 0.1062]; n=5 |

Raw clean trajectories, cue interventions, trained-minus-untrained contrasts and explicitly unperformed follow-ups: [analysis.json](analysis.json).

Figures: [training curves](figures/learning_curves.png), [common-input cue sensitivity](figures/cue_sensitivity.png).
