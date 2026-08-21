# Validation caveat

The original `enhanced_joint_prediction.py` was used during exploration and its early local RMSE report was later found to have a validation leakage in the evaluation runner. The real private score remains genuine, but that old RMSE claim is withdrawn. Use `optuna_goal_tuning.py` for the corrected causal walk-forward protocol.
