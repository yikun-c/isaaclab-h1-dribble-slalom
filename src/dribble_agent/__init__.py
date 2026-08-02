import gymnasium as gym

gym.register(
    id="Isaac-H1-Penalty-Kick-Direct-v0",
    entry_point="penalty_agent.penalty_env:PenaltyKickEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "penalty_agent.penalty_env:PenaltyKickEnvCfg",
        "rsl_rl_cfg_entry_point": "penalty_agent.ppo_cfg:PenaltyKickPPORunnerCfg",
    },
)
