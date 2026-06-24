#!/usr/bin/env python3
"""
LLM Introspection Experiment
=============================
For each LLM (Claude 4.6 Sonnet, ChatGPT-5, LLaMA), sample 10 record pairs from the
`datasets/mistral/books` and `datasets/mistral/carparts` datasets, run the self-explainer
to obtain saliency + counterfactual explanations, then ask follow-up questions:
  - "What method did you use for computing the saliency scores?"
  - "What method did you use for computing the counterfactuals?"

Aggregate the responses and report which algorithm/heuristic each LLM uses.

Usage
-----
Set environment variables before running (only needed for the LLMs you want to probe):

  # Claude 4.6 Sonnet via AWS Bedrock
  export AWS_ACCESS_KEY_ID=...
  export AWS_SECRET_ACCESS_KEY=...
  export AWS_REGION=us-east-1          # or us-west-2

  # ChatGPT-5 via Azure OpenAI
  export AZURE_OPENAI_API_KEY=...
  export AZURE_OPENAI_ENDPOINT=...     # e.g. https://<resource>.openai.azure.com
  export OPENAI_API_BASE=$AZURE_OPENAI_ENDPOINT
  export OPENAI_API_KEY=$AZURE_OPENAI_API_KEY

  # LLaMA 3.1-8B via Hugging Face Inference Endpoint
  export HUGGINGFACEHUB_API_TOKEN=...

Then run:
  cd /path/to/dev--ellmer
  python llm_introspection_experiment.py [--llms claude gpt5 llama] [--n 10]
"""

import argparse
import json
import os
import sys
import textwrap
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Union

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from tqdm import tqdm

# ── repo root on sys.path ─────────────────────────────────────────────────────
# Resolve the ellmer repo root relative to this script's location.
# If this script lives in <workspace>/Projects--ellmer/, the repo is at
# <workspace>/dev--ellmer/.  Fall back to ELLMER_REPO_ROOT env variable or
# current working directory so the script works from other locations too.
def _find_repo_root() -> Path:
    env_override = os.environ.get("ELLMER_REPO_ROOT")
    if env_override:
        return Path(env_override).resolve()
    # Sibling directory pattern (script in Projects--ellmer/, repo in dev--ellmer/)
    sibling = Path(__file__).resolve().parent.parent / "dev--ellmer"
    if sibling.is_dir():
        return sibling
    # Caller ran script from inside the repo
    cwd = Path.cwd()
    if (cwd / "ellmer").is_dir():
        return cwd
    raise FileNotFoundError(
        "Cannot locate the ellmer repo root. "
        "Set ELLMER_REPO_ROOT=/path/to/dev--ellmer or run from inside the repo."
    )

REPO_ROOT = _find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ellmer.selfexplainer import SelfExplainer
from ellmer.post_hoc.utils import merge_sources

# ── dataset paths ──────────────────────────────────────────────────────────────
DATASETS_ROOT = REPO_ROOT / "datasets" / "mistral"
DATASETS = ["books", "carparts"]

# ── prompt files (ZS/ptse) – same as the main experiment ──────────────────────
PROMPTS_DIR = REPO_ROOT / "ellmer" / "prompts"
ZS_PROMPTS = {
    "ptse_staged": {
                    "er": "ellmer/prompts/cot_staged_er.txt",
                    "saliency": "ellmer/prompts/cot_staged_saliency.txt",
                    "cf": "ellmer/prompts/cot_staged_cf.txt",
                }
}

# ── LLM configurations ────────────────────────────────────────────────────────
LLM_CONFIGS = {
    "claude": {
        "label": "Claude 4.6 Sonnet",
        "model_type": "bedrock",
        "model_name": "anthropic.claude-sonnet-4-6",
        # Use inference profile if available; falls back automatically.
        "deployment_name": "global.anthropic.claude-sonnet-4-6",
    },
    "gpt5": {
        "label": "ChatGPT-5 (gpt-5-nano)",
        "model_type": "azure_openai",
        "model_name": "gpt-5-nano",
        "deployment_name": "gpt-5-nano",
        "model_version": "2024-02-01",
        "temperature" : 1
    },
    "llama": {
        "label": "LLaMA 3.1-8B",
        "model_type": "hf",
        # Set HF_LLAMA_ENDPOINT env var to your endpoint URL, or use the repo ID for hosted inference.
        "model_name": os.environ.get(
            "HF_LLAMA_ENDPOINT",
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
        ),
        "deployment_name": "",
    },
}

# ── follow-up question templates ──────────────────────────────────────────────
FOLLOWUP_SALIENCY = (
    "What method or algorithm did you use for computing the saliency scores "
    "in your previous explanation? Please be specific about the technique, "
    "formula, or heuristic you applied."
)
FOLLOWUP_CF = (
    "What method or algorithm did you use for computing the counterfactual "
    "explanation in your previous answer? Please describe the strategy or "
    "heuristic you followed to generate the altered record pair."
)


# ── helpers ───────────────────────────────────────────────────────────────────

def load_dataset_samples(dataset_name: str, n: int) -> list[dict]:
    """Load n test pairs (with balanced match/non-match where possible)."""
    ds_dir = DATASETS_ROOT / dataset_name
    tableA = pd.read_csv(ds_dir / "tableA.csv")
    tableB = pd.read_csv(ds_dir / "tableB.csv")
    test = pd.read_csv(ds_dir / "test.csv")

    merged = merge_sources(test, "ltable_", "rtable_", tableA, tableB, ["label"], [])
    # Balanced sample: n/2 matches + n/2 non-matches (or all if < n/2 available)
    half = n // 2
    matches = merged[merged["label"] == 1].head(half)
    non_matches = merged[merged["label"] == 0].head(n - len(matches))
    sample = pd.concat([matches, non_matches]).head(n).reset_index(drop=True)

    rows = []
    for _, row in sample.iterrows():
        ltuple = {
            k[len("ltable_"):]: v
            for k, v in row.items()
            if k.startswith("ltable_") and k != "ltable_id"
        }
        rtuple = {
            k[len("rtable_"):]: v
            for k, v in row.items()
            if k.startswith("rtable_") and k != "rtable_id"
        }
        rows.append({"ltuple": ltuple, "rtuple": rtuple, "label": int(row["label"])})
    return rows


def build_explainer(llm_key: str) -> SelfExplainer:
    """Construct a SelfExplainer instance for the given LLM key."""
    cfg = LLM_CONFIGS[llm_key]
    if 'temperature' in cfg:
        temperature = cfg["temperature"]
    else:
        temperature = 0.0
    kwargs: dict[str, Any] = dict(
        model_type=cfg["model_type"],
        model_name=cfg["model_name"],
        temperature=temperature,
        explanation_granularity="attribute",
        prompts=ZS_PROMPTS,
    )
    if cfg.get("deployment_name"):
        kwargs["deployment_name"] = cfg["deployment_name"]
    if cfg.get("model_version"):
        kwargs["model_version"] = cfg["model_version"]
    return SelfExplainer(**kwargs)


def old_ask_followup(explainer: SelfExplainer, conversation: list, question: str) -> str:
    """
    Extend the existing conversation with a follow-up user question and
    return the LLM's reply.
    """
    extended = list(conversation) + [("user", question)]
    template = ChatPromptTemplate.from_messages(extended)
    try:
        answer = explainer._invoke(template)
        return answer or "(empty response)"
    except Exception as exc:
        return f"ERROR: {exc}"


class _SafeDict(dict):
    """dict subclass that returns '{key}' for missing keys instead of raising KeyError."""
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _conversation_to_messages(
        conversation: list[tuple[str, str]], template_kwargs: dict
) -> list:
    """
    Convert a (role, content) conversation list into LangChain message objects,
    substituting any remaining {template_variables} from the original prompts.

    The stored `conversation` in SelfExplainer results contains the raw prompt
    templates (with {ltuple}, {rtuple}, {feature}, {prediction} placeholders).
    We substitute known kwargs and leave any unrecognised placeholders untouched
    (they might be literal curly-brace text, e.g. JSON examples in prompt files).
    """
    role_map = {
        "system": SystemMessage,
        "human": HumanMessage,
        "user": HumanMessage,
        "ai": AIMessage,
        "assistant": AIMessage,
    }
    messages = []
    for role, content in conversation:
        # Attempt substitution; fall back to the raw string on any error
        # so that JSON examples with {key: value} don't break things.
        try:
            content = content.format_map(_SafeDict(template_kwargs))
        except Exception:
            pass
        msg_cls = role_map.get(role, HumanMessage)
        messages.append(msg_cls(content=content))
    return messages

def ask_followup(
    explainer: SelfExplainer,
    conversation: list[tuple[str, str]],
    question: str,
    template_kwargs: Union[dict, None] = None,
) -> str:
    """
    Append *question* to the conversation and call the LLM directly (bypassing
    LangChain template formatting to avoid KeyErrors from JSON-like content in
    prompt templates or assistant responses).
    """
    kwargs = template_kwargs or {}
    messages = _conversation_to_messages(conversation, kwargs)
    messages.append(HumanMessage(content=question))
    try:
        raw = explainer.llm.invoke(messages)
        answer = getattr(raw, "content", raw) or "(empty response)"
        return SelfExplainer._strip_response(answer)
    except Exception as exc:
        return f"ERROR: {exc}"

# ── main experiment ───────────────────────────────────────────────────────────

def run_experiment(llm_keys: list[str], n_samples: int, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, dict] = {}  # llm_key → aggregated data

    for llm_key in llm_keys:
        cfg = LLM_CONFIGS[llm_key]
        print(f"\n{'='*70}")
        print(f"  LLM: {cfg['label']}")
        print(f"{'='*70}")

        try:
            explainer = build_explainer(llm_key)
        except Exception as exc:
            print(f"  [SKIP] Could not build explainer: {exc}")
            continue

        llm_results: list[dict] = []

        for dataset_name in DATASETS:
            print(f"\n  Dataset: {dataset_name}")
            try:
                samples = load_dataset_samples(dataset_name, n_samples)
            except Exception as exc:
                print(f"  [SKIP] Could not load dataset {dataset_name}: {exc}")
                continue

            for idx, sample in enumerate(tqdm(samples, desc=f"  {dataset_name}")):
                ltuple = sample["ltuple"]
                rtuple = sample["rtuple"]
                label = sample["label"]

                record: dict[str, Any] = {
                    "llm": llm_key,
                    "dataset": dataset_name,
                    "sample_idx": idx,
                    "label": label,
                    "ltuple": ltuple,
                    "rtuple": rtuple,
                }

                try:
                    result = explainer.predict_and_explain(ltuple, rtuple)
                    record["prediction"] = result.get("prediction")
                    record["saliency"] = result.get("saliency", {})
                    record["cf"] = result.get("cf", {})
                    conversation = result.get("conversation", [])

                    result = explainer.predict_and_explain(ltuple, rtuple)
                    prediction = result.get("prediction", 0)
                    record["prediction"] = prediction
                    record["saliency"] = result.get("saliency", {})
                    record["cf"] = result.get("cf", {})
                    conversation = result.get("conversation", [])

                    pred_int = int(prediction) if prediction in (0, 1) else int(bool(prediction))
                    tmpl_kwargs = dict(
                        ltuple=ltuple,
                        rtuple=rtuple,
                        feature="attribute",
                        prediction=pred_int,
                        prediction_int=pred_int,
                        prediction_label="MATCH" if pred_int == 1 else "NON-MATCH",
                    )

                    '''# ── follow-up 1: saliency method ──────────────────────
                    sal_method = ask_followup(explainer, conversation, FOLLOWUP_SALIENCY)
                    record["saliency_method_response"] = sal_method
                    # Extend conversation with the follow-up exchange
                    conversation = list(conversation) + [
                        ("user", FOLLOWUP_SALIENCY),
                        ("assistant", sal_method),
                    ]

                    # ── follow-up 2: CF method ────────────────────────────
                    cf_method = ask_followup(explainer, conversation, FOLLOWUP_CF)
                    record["cf_method_response"] = cf_method'''
                    sal_method = ask_followup(
                        explainer, conversation, FOLLOWUP_SALIENCY, tmpl_kwargs
                    )
                    record["saliency_method_response"] = sal_method
                    # Extend conversation with the follow-up exchange
                    conversation = list(conversation) + [
                        ("user", FOLLOWUP_SALIENCY),
                        ("assistant", sal_method),
                    ]

                    # ── follow-up 2: CF method ────────────────────────────
                    cf_method = ask_followup(
                        explainer, conversation, FOLLOWUP_CF, tmpl_kwargs
                    )
                    record["cf_method_response"] = cf_method

                except Exception as exc:
                    traceback.print_exc()
                    record["error"] = str(exc)
                    record["saliency_method_response"] = ""
                    record["cf_method_response"] = ""

                llm_results.append(record)

        # ── save raw results per LLM ──────────────────────────────────────
        raw_path = output_dir / f"introspection_{llm_key}_raw.json"
        with open(raw_path, "w") as f:
            json.dump(llm_results, f, indent=2, default=str)
        print(f"\n  Raw results saved → {raw_path}")

        all_results[llm_key] = llm_results

    # ── aggregate and report ──────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  AGGREGATED FINDINGS")
    print(f"{'='*70}")

    summary: dict[str, dict] = {}
    for llm_key, records in all_results.items():
        cfg = LLM_CONFIGS[llm_key]
        sal_responses = [r["saliency_method_response"] for r in records if r.get("saliency_method_response")]
        cf_responses  = [r["cf_method_response"]       for r in records if r.get("cf_method_response")]

        summary[llm_key] = {
            "label": cfg["label"],
            "n_records": len(records),
            "saliency_method_responses": sal_responses,
            "cf_method_responses": cf_responses,
        }

        print(f"\n── {cfg['label']} ──")
        print(f"  Samples processed: {len(records)}")

        if sal_responses:
            print("\n  Saliency method self-reports (first 2 samples):")
            for i, resp in enumerate(sal_responses[:2]):
                print(f"    [{i}] {textwrap.fill(resp[:500], width=72, subsequent_indent='       ')}")

        if cf_responses:
            print("\n  Counterfactual method self-reports (first 2 samples):")
            for i, resp in enumerate(cf_responses[:2]):
                print(f"    [{i}] {textwrap.fill(resp[:500], width=72, subsequent_indent='       ')}")

    # ── save summary ──────────────────────────────────────────────────────────
    summary_path = output_dir / "introspection_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved → {summary_path}")

    # ── keyword frequency analysis ────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  KEYWORD FREQUENCY ANALYSIS")
    print(f"{'='*70}")

    # Broad set of keywords to look for in self-reports
    SAL_KEYWORDS = [
        "attention", "weight", "score", "feature importance", "frequency",
        "heuristic", "uniform", "equal", "gradient", "shap", "lime",
        "semantic similarity", "overlap", "cosine", "tf-idf", "count",
        "position", "relevance", "manual", "rule", "arbitrary", "proportional",
        "softmax", "probability", "intuition", "reasoning",
    ]
    CF_KEYWORDS = [
        "substitute", "replace", "swap", "change", "alter", "perturb",
        "heuristic", "greedy", "brute force", "exhaustive", "random",
        "minimal", "nearest", "similar", "semantic", "cosine", "distance",
        "rule", "manual", "intuition", "reasoning", "counterfactual",
        "flip", "opposite", "negate", "contrast", "ablation",
    ]

    for llm_key, data in summary.items():
        label = data["label"]
        print(f"\n  {label}")

        sal_text = " ".join(data["saliency_method_responses"]).lower()
        found_sal = [kw for kw in SAL_KEYWORDS if kw in sal_text]
        print(f"    Saliency keywords mentioned : {', '.join(found_sal) if found_sal else '(none matched)'}")

        cf_text = " ".join(data["cf_method_responses"]).lower()
        found_cf = [kw for kw in CF_KEYWORDS if kw in cf_text]
        print(f"    CF keywords mentioned       : {', '.join(found_cf) if found_cf else '(none matched)'}")

    print(f"\n{'='*70}")
    print("  Done. Review the raw JSON files for full responses.")
    print(f"{'='*70}\n")


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LLM introspection experiment: ask LLMs what method they used for saliency/CF."
    )
    parser.add_argument(
        "--llms",
        nargs="+",
        choices=list(LLM_CONFIGS.keys()),
        default=list(LLM_CONFIGS.keys()),
        help="Which LLMs to probe (default: all)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Number of sample pairs per dataset (default: 10)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).parent / "introspection_results"),
        help="Directory for output files",
    )
    args = parser.parse_args()

    run_experiment(
        llm_keys=args.llms,
        n_samples=args.n,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()