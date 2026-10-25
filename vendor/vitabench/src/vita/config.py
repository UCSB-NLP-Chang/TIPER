import os
import yaml
from pathlib import Path

_models_yaml_path = Path(__file__).parent / "models.yaml"
if os.environ.get("VITA_MODEL_CONFIG_PATH", None):
    _models_yaml_path = Path(os.environ.get("VITA_MODEL_CONFIG_PATH"))

if not os.path.exists(str(_models_yaml_path)):
    raise FileNotFoundError(
        f"Model configuration file ({_models_yaml_path}) dose not exists, you should create it first.")


def _deep_merge_dict(base_dict: dict, override_dict: dict) -> dict:
    result = base_dict.copy()

    for key, value in override_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge_dict(result[key], value)
        else:
            result[key] = value

    return result


def _expand_environment(value):
    """Expand ${VAR} placeholders without storing credentials in YAML."""
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, str):
        return os.path.expandvars(value)
    return value


try:
    with open(_models_yaml_path, 'r') as f:
        models_config_yaml = _expand_environment(
            yaml.load(f, Loader=yaml.FullLoader)
        )

    default_model_config = models_config_yaml.get('default', {})

    models = {"default": default_model_config}
    for model in models_config_yaml.get('models', []):
        model_name = model['name']
        merged_config = _deep_merge_dict(default_model_config, model)
        models[model_name] = merged_config

    print(f"Available models: {list(models.keys())}")

except FileNotFoundError:
    print(f"Warning: models.yaml not found at {_models_yaml_path}")
    models = {}
except Exception as e:
    print(f"Error loading models.yaml: {e}")
    models = {}

# SIMULATION
DEFAULT_MAX_STEPS = 300
DEFAULT_MAX_RETRIES = 3
DEFAULT_MAX_ERRORS = 10
# Per-task wall-clock budget in seconds. 0 disables the cap (paper-faithful:
# the paper does not impose a wall-clock timeout). When > 0 the orchestrator
# terminates a run that exceeds it (cooperative check in the step loop), which
# stops runaway non-converging tasks from clogging a concurrency slot.
DEFAULT_MAX_DURATION = 0
DEFAULT_SEED = 300
DEFAULT_MAX_CONCURRENCY = 1
DEFAULT_NUM_TRIALS = 1
DEFAULT_SAVE_TO = None
DEFAULT_LOG_LEVEL = "DEBUG"
DEFAULT_LANGUAGE = "chinese"
DEFAULT_EVALUATION_TYPE = "trajectory"

# LLM
DEFAULT_AGENT_IMPLEMENTATION = "llm_agent"
DEFAULT_USER_IMPLEMENTATION = "user_simulator"
DEFAULT_LLM_AGENT = "gpt-4.1"
DEFAULT_LLM_USER = "gpt-4.1"
DEFAULT_LLM_EVALUATOR = "anthropic.claude-3.7-sonnet"
