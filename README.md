

ELLMER
=======

# Installation

simply run :
```shell
pip install .
```

# Usage

To replicate experiments, first download the DeepMatcher datasets somewhere on your local disk, then use the python `eval` script.

You can choose the LLM `model_type` by choosing: 
 * OpenAI models deployed on Azure with `--model_type azure_openai`
 * local Llama2-13B model `--model_type llama2`
 * local Falcon model `--model_type falcon`
 * HF models `--model_type hf --model_name meta-llama/Llama-3.1-8B-Instruct`

You can choose how many samples the evaluation should account for (`--samples` param), the explanation granularity (`--granularity` param, accepted values are `token` and `attribute`).

You can choose one or more `datasets` for the evaluation as the name of one or more directories in the `base_dir`.

```python
python scripts/eval.py --base_dir path/to/deepmatcher_datasets --model_type azure_openai --datasets beers --samples 5 --granularity token
```

Other optional parameters can be specified in the [script](scripts/eval.py#l160).

**Timing:** Results include `total_local_time` and `avg_latency_local`, which measure run time without the remote LLM execution step (LLMChain / HuggingFace / OpenAI API calls), for more stable timing across runs. When an explainer provides `llm_time`, only that remote execution is subtracted; otherwise the full `predict_and_explain` duration is treated as LLM time. With `--workers > 1`, the split between local and LLM time is approximate because wall time can be less than the sum of per-call latencies.

# Notebooks

* [ChatGPT self-explanations prompt sensitivity](notebooks/self_expl_prompt_variance.ipynb).
* [ChatGPT self-explanations vs post_hoc attribute consistency](notebooks/example_attribute.ipynb).
* [ChatGPT self-explanations vs post_hoc token consistency](notebooks/example_token.ipynb).

