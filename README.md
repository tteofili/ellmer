

ELLMER
=======
<img src="https://github.com/tteofili/ellmer/assets/512815/71b950f3-2b36-4a58-8f48-955380b30a09" width=20%>

Code for _Can we trust LLM Self-Explanations for Entity Resolution?_.

# Installation

To install _ELLMER_ locally run :
```shell
pip install .
```

# Usage

To replicate experiments, first download the DeepMatcher datasets somewhere on your local disk, then use the python `eval` script.

You can choose the LLM `model_type` by choosing: 
 * OpenAI models deployed on Azure with `--model_type azure_openai`
 * local Llama2-13B model `--model_type llama2`
 * local Falcon model `--model_type falcon`

You can choose how many samples the evaluation should account for (`--samples` param), the explanation granularity (`--granularity` param, accepted values are `token` and `attribute`).

You can choose one or more `datasets` for the evaluation as the name of one or more directories in the `base_dir`.

```python
python scripts/eval.py --base_dir path/to/deepmatcher_datasets --model_type azure_openai --datasets beers --samples 5 --granularity token
```

Other optional parameters can be specified in the [script](scripts/eval.py#l160).

# Notebooks

* [ChatGPT self-explanations prompt sensitivity](notebooks/examples.ipynb).

# Citing ELLMER

If you extend or use this work, please cite:

```
@article{teofili2023ellmer,
  title={Can we trust LLM Self-Explanations for Entity Resolution?},
  author={Teofili, Tommaso and Firmani, Donatella and Koudas, Nick and Merialdo, Paolo and Srivastava, Divesh},
  year={2024}
}
```
