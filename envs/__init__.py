from gym.envs.registration import register

register(
        id='GankenKun-v0',
        entry_point='envs.GankenKun_pybullet.GankenKun_env:GankenKunEnv',
        )
