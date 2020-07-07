from gym.envs.registration import register

register(
    id='bushberry-v0',
    entry_point='gym_foo.envs:BushBerryEnv',
)
register(
    id='bushberry-extrahard-v0',
    entry_point='gym_foo.envs:BushBerryExtraHardEnv',
)