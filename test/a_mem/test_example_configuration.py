"""One local env file is enough to configure real service examples."""
import pytest


def test_environment_configuration_keeps_chat_and_embedding_models_separate(monkeypatch):
    from mematlas.methods.a_mem.example import _config_from_environment
    for key in ("AMEM_EMBEDDING_BASE_URL", "AMEM_EMBEDDING_DIMS"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("AMEM_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("AMEM_LLM_MODEL", "my-chat-model")
    monkeypatch.setenv("AMEM_EMBEDDING_MODEL", "my-vector-model")
    config = _config_from_environment()
    assert config.method.method_id == "a-mem"
    assert config.providers["llm"]["model"] == "my-chat-model"
    assert config.providers["embedding"]["model"] == "my-vector-model"
    assert config.providers["embedding"]["base_url"] == "https://example.invalid/v1"
    assert config.providers["embedding"]["dimensions"] is None
    assert all(provider["api_key_env"] == "AMEM_API_KEY" for provider in config.providers.values())
    monkeypatch.delenv("AMEM_EMBEDDING_MODEL")
    with pytest.raises(ValueError, match="AMEM_EMBEDDING_MODEL"):
        _config_from_environment()
