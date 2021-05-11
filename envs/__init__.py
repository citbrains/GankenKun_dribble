from gym.envs.registration import register

register(
        id='GankenKun-v0',
        entry_point='envs.GankenKun_pybullet.GankenKun_env:GankenKunEnv',
        )
register(
        id='GankenKun_obs-v0',
        entry_point='envs.GankenKun_pybullet.GankenKunObstacleRandom_env:GankenKunObstacleEnv',
        )
register(
        id='GankenKun_map_obs-v0',
        entry_point='envs.GankenKun_pybullet.MapGankenKunObstacle_env:GankenKunObstacleEnv',
        )
