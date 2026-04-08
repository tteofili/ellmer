import openai
import os
import torch
from langchain.llms import HuggingFacePipeline, LlamaCpp
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, pipeline

import ellmer.utils
from ellmer.base_explainer import BaseLLMExplainer

openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_key = os.getenv("OPENAI_API_KEY")


def falcon_pipeline(model_id="vilsonrodrigues/falcon-7b-instruct-sharded", quantized: bool = False):
    if quantized:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model_4bit = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            quantization_config=quantization_config,
            trust_remote_code=True
        )
        model = model_4bit
    else:
        model = model_id

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    fpip = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        use_cache=True,
        device_map="auto",
        max_length=296,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    return HuggingFacePipeline(pipeline=fpip)


def llama2_llm(verbose=True, quantized_model_path='ggml-model-q4_0.gguf', temperature=0.0, top_p=1, n_ctx=6000):
    llama2_llm = LlamaCpp(
        model_path=quantized_model_path,
        temperature=temperature,
        top_p=top_p,
        n_ctx=n_ctx,
        verbose=verbose,
    )
    return llama2_llm
