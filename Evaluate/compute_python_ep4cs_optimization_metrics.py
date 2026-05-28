from pathlib import Path

from compute_summary_metrics import run_metric_script


ROOT_DIR = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT_DIR / "result" / "EP4CS" / "Python"
DEFAULT_BASELINE_PATH = RESULT_DIR / "test_new_pred_EP4CS_optimized.jsonl"
DEFAULT_ABLATION_PATH = RESULT_DIR / "test_new_pred_EP4CS_optimized.jsonl"
DEFAULT_OUTPUT_DIR = RESULT_DIR / "optimization_effect_metrics"

SYSTEMS = [
    {
        "name": "EP4CS",
        "summary_field": "pred_EP4CS",
        "score_prefix": "",
    },
    {
        "name": "No_Components_Ablation",
        "summary_field": "pred_EP4CS_optimized",
        "score_prefix": "_ablation_no_critic",
    },
]


if __name__ == "__main__":
    run_metric_script(DEFAULT_BASELINE_PATH, DEFAULT_ABLATION_PATH, DEFAULT_OUTPUT_DIR, SYSTEMS)
