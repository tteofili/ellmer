"""LangChain chat model factory for Amazon Bedrock (Anthropic Claude and other Bedrock chat models)."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from pydantic import SecretStr


def _strip_env(name: str) -> Optional[str]:
    v = os.environ.get(name)
    if v is None:
        return None
    s = v.strip()
    return s if s else None


def _explicit_env_credentials() -> Dict[str, SecretStr]:
    """
    When both access key and secret are non-empty in the environment, return kwargs for
    ChatBedrockConverse so credentials are bound at construction time (avoids profile/chain
    ambiguity). Includes AWS_SESSION_TOKEN when set (required for assumed-role / STS creds).
    """
    ak = _strip_env("AWS_ACCESS_KEY_ID")
    sk = _strip_env("AWS_SECRET_ACCESS_KEY")
    if not ak or not sk:
        return {}
    out: Dict[str, SecretStr] = {
        "aws_access_key_id": SecretStr(ak),
        "aws_secret_access_key": SecretStr(sk),
    }
    st = _strip_env("AWS_SESSION_TOKEN")
    if st:
        out["aws_session_token"] = SecretStr(st)
    return out


def resolve_bedrock_region(region_name: Optional[str] = None) -> str:
    """Prefer explicit region, then ELLMER_BEDROCK_REGION / AWS env, boto3 session, then us-east-1."""
    if region_name:
        return region_name
    r = os.environ.get("ELLMER_BEDROCK_REGION") or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    if r:
        return r
    try:
        import boto3

        session = boto3.Session()
        if session.region_name:
            return session.region_name
    except Exception:
        pass
    return "us-east-1"


def build_chat_bedrock(
    model_name: str,
    temperature: float = 0.0,
    inference_profile_id: Optional[str] = None,
    region_name: Optional[str] = None,
) -> Any:
    """
    Return a LangChain ``ChatBedrockConverse`` instance.

    ``model_name`` is the Bedrock model ID (e.g. ``anthropic.claude-3-5-sonnet-20240620-v1:0``).
    If ``inference_profile_id`` is set, it is used as the model identifier instead (application
    inference profile or cross-region profile ID).
    Region: ``region_name``, else ``ELLMER_BEDROCK_REGION`` / ``AWS_REGION`` / ``AWS_DEFAULT_REGION``, else boto3 default, else ``us-east-1``.

    When **both** ``AWS_ACCESS_KEY_ID`` and ``AWS_SECRET_ACCESS_KEY`` are set (non-empty), they are
    passed explicitly to LangChain (with ``AWS_SESSION_TOKEN`` if set). ``AWS_PROFILE`` is not used
    in that case. For temporary credentials (assume role, SSO-exported keys), you must set
    ``AWS_SESSION_TOKEN``; omitting it often yields "invalid security token" / ``UnrecognizedClientException``.
    """
    try:
        from langchain_aws import ChatBedrockConverse
    except ImportError as e:
        raise ImportError(
            "Bedrock requires langchain-aws and boto3. Install with: pip install 'ellmer[bedrock]' "
            "or pip install langchain-aws boto3"
        ) from e

    override = (inference_profile_id or "").strip()
    base = (model_name or "").strip()
    model_id = override or base
    if not model_id:
        raise ValueError("bedrock requires a non-empty model_name (Bedrock model ID) or inference_profile_id")

    region = resolve_bedrock_region(region_name)
    kwargs: dict = {
        "model_id": model_id,
        "region_name": region,
        "temperature": temperature,
    }
    explicit = _explicit_env_credentials()
    kwargs.update(explicit)

    profile = _strip_env("AWS_PROFILE")
    # Named profile forces a boto3 Session that ignores env access keys (langchain_aws create_aws_client).
    if profile and not explicit:
        kwargs["credentials_profile_name"] = profile

    return ChatBedrockConverse(**kwargs)
