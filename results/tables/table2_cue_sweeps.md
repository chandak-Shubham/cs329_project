# Table 2: Parameter Sweep Matrix Across Genetic Cue Noise (5 Training Seeds)
| Noise Level ($p$) | Mean $w^g$ (Genetic) | Mean $w^b$ (Behavioral) | Final Population | Total Donations |
| :---: | :---: | :---: | :---: | :---: |
| $0.0$ | $0.287 \pm 0.077$ | $0.713 \pm 0.077$ | $0.0$ | $87$ |
| $0.1$ | $0.287 \pm 0.078$ | $0.713 \pm 0.078$ | $0.0$ | $85$ |
| $0.2$ | $0.287 \pm 0.079$ | $0.713 \pm 0.079$ | $0.0$ | $89$ |
| $0.3$ | $0.286 \pm 0.078$ | $0.714 \pm 0.078$ | $0.0$ | $88$ |
| $0.4$ | $0.287 \pm 0.078$ | $0.713 \pm 0.078$ | $0.0$ | $83$ |
| $0.5$ | $0.287 \pm 0.078$ | $0.713 \pm 0.078$ | $0.0$ | $90$ |

*Mean weights across noise conditions: genetic 0.287, behavioral 0.713. Frozen policies; these sweeps do not measure online learning.*

95% Student t intervals for genetic and behavioral weights:
- Noise 0.0: genetic [0.17981365181953107, 0.39474385687187935]; behavioral [0.6052561451780908, 0.8201863460089708]
- Noise 0.1: genetic [0.17898927240778256, 0.39543302615655357]; behavioral [0.6045669815468567, 0.8210107232712379]
- Noise 0.2: genetic [0.17778840281656022, 0.3960490916788662]; behavioral [0.603950907502483, 0.8222115954966408]
- Noise 0.3: genetic [0.17822305805837724, 0.3946670778640249]; behavioral [0.6053329175947247, 0.8217769376525251]
- Noise 0.4: genetic [0.17803116316814457, 0.39580175812907714]; behavioral [0.6041982427936219, 0.8219688363565536]
- Noise 0.5: genetic [0.17878695939520212, 0.39578565021040074]; behavioral [0.6042143510665471, 0.8212130371684453]