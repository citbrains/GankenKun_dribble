from gym.envs.registration import register

register(
        id='GankenKun-v0',
        entry_point='envs.GankenKun_pybullet.GankenKun_env:GankenKunEnv',
)
register(
        id='GankenKun-v1',
        entry_point='envs.GankenKun_pybullet.GankenKunObstacle_env:GankenKunObstacleEnv',
)
register(
        id='GankenKun-v2',
        entry_point='envs.GankenKun_pybullet.GankenKunObstacleRandom_env:GankenKunObstacleEnv',
)
register(
        id='GankenKunWalk-v1',
        entry_point='envs.GankenKun_pybullet.GankenKun_walk:GankenKunWalkEnv',
)
