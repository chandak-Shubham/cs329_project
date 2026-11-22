# Revised project proposal
CS329 Multi-Agent Systems, Group 21. This retrospective revision clarifies the original idea review after implementation. The final report records the experiments and outcomes.

## Decision problem

A resident on a spatial grid can spend energy to help one nearby recipient. Receiving help aids survival, and helping costs the donor. Two locally visible signals may disagree: a chromosome marker that can be corrupted or copied, and a neighbour's recent giving. Can learners sustain giving, and when does information about the recipient change whom they help?

## Model and ownership of learning

Each of 24 founders owns a separate PPO actor, critic and optimizer. A descendant shares its founder family's policy. Each family updates from its own members' transitions. A resident chooses among four moves, staying, and giving to one of up to eight observed neighbours. It observes its own state and local positions, energies, marker cues and giving histories. The policy must infer the value of an action without a model of transitions, true cue reliability or mimic identity.

The baseline reward is the resident's pre-division energy change. A separate intervention credits the donor for the recipient's marker-weighted gain. An eight-bit marker proxies similarity. Biological kin selection would require independent evidence about heritable fitness and ancestry.

## Experiments and criteria

| Question | Comparison | Observable outcome |
| --- | --- | --- |
| Does private learning sustain cooperation? | Trained versus untrained policies with matched founders and worlds, plus selection without PPO | Donations per resident step and residents alive at step 300. Record extinction even if early giving was high. |
| Does recipient identity matter? | Marker-weighted credit versus equal credit for every gift and recipient-blind credit matched in expectation | Among available neighbours, measure excess recipient similarity and the share of gifts satisfying rB > C. Survival alone cannot establish targeting. |
| Does preference change with the benefit/cost ratio? | Sweep five ratios and compare learned choices with a scripted rule | Fit a recipient-choice model with marker, behavioural history and a marker-by-ratio interaction. A positive interaction is a directional prediction that the data can contradict. |
| Do agents switch cues or form spatial clusters? | Corrupt markers, introduce non-giving mimics, and randomize offspring placement | Test behavioural reliance and post-invasion gifts alongside survival. Compare adjacent resident similarity under local and random placement. |

Average fresh evaluation worlds within each training seed. Pair treatments on seed and world. For survival across several ratios, aggregate within seed before inference. Extinction can remove opportunities to observe recipient choice, so report the relevant denominators. A negative result is informative when the control preserves opportunities for the behaviour.

Hamilton (1964) motivates the biological inequality. Leibo et al. (2017) motivate the sequential social dilemma, and Schulman et al. (2017) describe PPO. The final report gives the references and the implemented protocol.
