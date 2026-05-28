# MOCSR

This repository contains the code and curated test sets for MOCSR, a code-summary refinement workflow based on code evidence, retrieved dimension examples, iterative optimization, CodeRPE four-dimensional evaluation, and final metric aggregation.

## Repository Layout

- `dataset/java/test_new.jsonl` and `dataset/python/test_new.jsonl`: curated test sets released with this repository.
- `result/<model>/<language>/extract_*_evidence.py`: build structured code evidence.
- `result/<model>/<language>/retrieve_dimension_examples.py`: retrieve similar examples for each target sample.
- `RQ1/`: main optimization scripts for EP4CS, CodeT5, and HGA outputs.
- `RQ2/`: ablation scripts.
- `RQ3/`: retrieval and model-variant analysis scripts.
- `CodeRPE/`: four-dimensional evaluation scripts.
- `Evaluate/`: final metric scripts. These scripts report only `corpus_BLEU`, `mean_METEOR`, `mean_ROUGE_L`, and the four-dimensional scores.

## Data

Only the two newly extracted test sets are included in this repository:

- `dataset/java/test_new.jsonl`
- `dataset/python/test_new.jsonl`

For the remaining Java and Python training/validation/test data, prepare CodeSearchNet from the CodeXGLUE code-to-text task:

https://github.com/microsoft/CodeXGLUE/tree/main/Code-Text/code-to-text

After downloading and cleaning CodeSearchNet, place the processed files under:

- `dataset/java/train.jsonl`
- `dataset/java/valid.jsonl`
- `dataset/java/test.jsonl`
- `dataset/python/train.jsonl`
- `dataset/python/valid.jsonl`
- `dataset/python/test.jsonl`

Generated prediction and optimization files under `result/**/*.jsonl` are intentionally ignored in git.

## Environment

Create or activate the project environment, then install dependencies:

```powershell
conda activate CodeRPE
pip install -r requirements.txt
```

Set API configuration with environment variables instead of editing scripts:

```powershell
$env:OPENAI_API_KEY="your_api_key"
$env:OPENAI_BASE_URL="https://api.deepseek.com"
```

## Workflow

1. Prepare CodeSearchNet Java/Python data and keep the released curated test sets in `dataset/java/test_new.jsonl` and `dataset/python/test_new.jsonl`.
2. Generate baseline summaries for EP4CS, CodeT5, or HGA and place the prediction JSONL files under the corresponding `result/<model>/<language>/` directory.
3. Build code evidence, for example:

```powershell
python result/EP4CS/Java/extract_java_evidence.py
python result/EP4CS/Python/extract_python_evidence.py
```

4. Retrieve similar examples, for example:

```powershell
python result/EP4CS/Java/retrieve_dimension_examples.py
python result/EP4CS/Python/retrieve_dimension_examples.py
```

5. Run the corresponding optimization script in `RQ1/`, `RQ2/`, or `RQ3/`, for example:

```powershell
python RQ1/optimize_ep4cs_summaries_.py
python RQ1/optimize_python_ep4cs_summaries.py
```

6. Run CodeRPE four-dimensional evaluation:

```powershell
python CodeRPE/setting_performance_ep4cs_full_optimized.py --method ablation --num-samples 1000 --force
python CodeRPE/setting_performance_python_ep4cs_full_optimized.py --method ablation --num-samples 1000 --force
```

7. Compute final metrics:

```powershell
python Evaluate/compute_ep4cs_optimization_metrics.py
python Evaluate/compute_python_ep4cs_optimization_metrics.py
```

The final metric scripts write `auto_metrics_summary.csv` and `auto_metrics_summary.json` to the corresponding `result/<model>/<language>/optimization_effect_metrics/` directory.

## Notes

- API keys are not stored in the repository. Use `OPENAI_API_KEY` and `OPENAI_BASE_URL`.
- Result JSONL files are excluded from git because they may be renamed or regenerated.
- The same evidence extraction and retrieval workflow applies to EP4CS, CodeT5, HGA, Java, and Python variants.
