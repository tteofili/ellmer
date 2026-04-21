"""Tests for Amazon Bedrock LLM factory (optional langchain-aws)."""

import os
from unittest.mock import MagicMock, patch

import pytest


def test_resolve_bedrock_region_explicit():
    from ellmer.bedrock_llm import resolve_bedrock_region

    assert resolve_bedrock_region("ap-southeast-2") == "ap-southeast-2"


def test_build_chat_bedrock():
    pytest.importorskip("langchain_aws")
    from ellmer.bedrock_llm import build_chat_bedrock

    llm = build_chat_bedrock(
        "anthropic.claude-3-haiku-20240307-v1:0",
        temperature=0.0,
        region_name="us-east-1",
    )
    assert getattr(llm, "model_id", None) == "anthropic.claude-3-haiku-20240307-v1:0"


def test_build_chat_bedrock_inference_profile_over_model_name():
    pytest.importorskip("langchain_aws")
    from ellmer.bedrock_llm import build_chat_bedrock

    llm = build_chat_bedrock(
        "anthropic.claude-3-haiku-20240307-v1:0",
        temperature=0.0,
        inference_profile_id="my.profile.id",
        region_name="us-east-1",
    )
    assert llm.model_id == "my.profile.id"


def test_env_access_key_id_omits_credentials_profile_name():
    """When AWS_ACCESS_KEY_ID is set, do not pass credentials_profile_name (LangChain would ignore env keys)."""
    pytest.importorskip("langchain_aws")
    from ellmer.bedrock_llm import build_chat_bedrock

    with patch.dict(
        os.environ,
        {
            "AWS_PROFILE": "named-profile",
            "AWS_ACCESS_KEY_ID": "AKIAIOSFODNN7EXAMPLE",
            "AWS_SECRET_ACCESS_KEY": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            "AWS_REGION": "us-east-1",
        },
        clear=False,
    ):
        with patch("langchain_aws.ChatBedrockConverse") as mock_converse:
            mock_converse.return_value = MagicMock()
            build_chat_bedrock(
                "anthropic.claude-3-haiku-20240307-v1:0",
                temperature=0.0,
                region_name="us-east-1",
            )
            kwargs = mock_converse.call_args.kwargs
            assert "credentials_profile_name" not in kwargs
            assert kwargs["aws_access_key_id"].get_secret_value() == "AKIAIOSFODNN7EXAMPLE"
            assert "wJalr" in kwargs["aws_secret_access_key"].get_secret_value()


def test_explicit_env_includes_session_token():
    pytest.importorskip("langchain_aws")
    from ellmer.bedrock_llm import build_chat_bedrock

    with patch.dict(
        os.environ,
        {
            "AWS_ACCESS_KEY_ID": "AKIAIOSFODNN7EXAMPLE",
            "AWS_SECRET_ACCESS_KEY": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            "AWS_SESSION_TOKEN": "tokentestvalue",
            "AWS_REGION": "us-east-1",
        },
        clear=False,
    ):
        with patch("langchain_aws.ChatBedrockConverse") as mock_converse:
            mock_converse.return_value = MagicMock()
            build_chat_bedrock(
                "anthropic.claude-3-haiku-20240307-v1:0",
                temperature=0.0,
                region_name="us-east-1",
            )
            kwargs = mock_converse.call_args.kwargs
            assert kwargs["aws_session_token"].get_secret_value() == "tokentestvalue"
