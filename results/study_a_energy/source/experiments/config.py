"""Shared, explicit environment for the revised study (historical runs are separate)."""

SEEDS = (42, 101, 202, 303, 404)
BASE_ENV = dict(grid_size=18, max_capacity=45, initial_population=24,
                initial_energy=30.0, e_mitosis=60.0, mutation_rate=0.05,
                c_base=0.6, c_density=0.08, donation_cost=2.0,
                donation_benefit=5.0, max_steps=300)


def environment_config(**overrides):
    return {**BASE_ENV, **overrides}
