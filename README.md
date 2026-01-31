

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

# Notebooks

* [ChatGPT self-explanations prompt sensitivity](notebooks/self_expl_prompt_variance.ipynb).
* [ChatGPT self-explanations vs post_hoc attribute consistency](notebooks/example_attribute.ipynb).
* [ChatGPT self-explanations vs post_hoc token consistency](notebooks/example_token.ipynb).

