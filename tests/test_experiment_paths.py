from ellmer.experiment_paths import parse_results_json_path


def test_parse_multi_run_session_path():
    p = "experiments/bedrock/anthropic.claude-3-haiku/attribute/20260408_17_15/run_1/abt_buy/zs_sample_results.json"
    info = parse_results_json_path(p)
    assert info is not None
    assert info.model_type == "bedrock"
    assert info.model_name == "anthropic.claude-3-haiku"
    assert info.granularity == "attribute"
    assert info.session == "20260408_17_15"
    assert info.run_id == 1
    assert info.dataset == "abt_buy"
    assert info.explainer_key == "zs_sample"


def test_parse_legacy_path():
    p = "experiments/azure_openai/gpt-35-turbo/attribute/Walmart-Amazon/20240101/12_00/cot_sample_results.json"
    info = parse_results_json_path(p)
    assert info is not None
    assert info.run_id is None
    assert info.session == "20240101_12_00"
    assert info.dataset == "Walmart-Amazon"
    assert info.explainer_key == "cot_sample"
