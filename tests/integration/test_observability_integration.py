"""集成测试：覆盖 Langfuse 可观测性辅助的初始化、trace 和 flush 分支。"""

import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest


pytestmark = [pytest.mark.integration, pytest.mark.full]


def test_observability_disabled_and_enabled_trace_paths(monkeypatch):
    """验证未配置和已配置 Langfuse 时，可观测性辅助都能稳定工作。"""
    import observability

    class FakeObservation:
        def __init__(self):
            self.updated = {}

        def update(self, **kwargs):
            self.updated.update(kwargs)

    class FakeClient:
        def __init__(self):
            self.flushed = False
            self.observation = FakeObservation()

        @contextmanager
        def start_as_current_observation(self, **kwargs):
            self.kwargs = kwargs
            yield self.observation

        def flush(self):
            self.flushed = True

    fake_client = FakeClient()

    class FakeLangfuse:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    @contextmanager
    def fake_propagate_attributes(**kwargs):
        yield

    fake_langfuse_module = SimpleNamespace(
        Langfuse=FakeLangfuse,
        get_client=lambda: fake_client,
        propagate_attributes=fake_propagate_attributes,
    )
    monkeypatch.setitem(sys.modules, "langfuse", fake_langfuse_module)
    monkeypatch.setattr(observability, "_initialized", False)
    monkeypatch.setattr(observability, "_configured", False)
    monkeypatch.setattr(observability, "_warned_missing_config", False)
    monkeypatch.setattr(observability, "LANGFUSE_PUBLIC_KEY", "")
    monkeypatch.setattr(observability, "LANGFUSE_SECRET_KEY", "")
    monkeypatch.setattr(observability, "LANGFUSE_HOST", "http://host")
    monkeypatch.setattr(observability, "LANGFUSE_BASE_URL", "")

    assert observability.init_langfuse() is fake_client
    assert not observability.is_langfuse_enabled()
    assert observability.get_langfuse_metadata(
        session_id="s",
        user_id="u",
        request_id="r",
        component="c",
        operation="o",
        extra={"x": "y"},
    )["x"] == "y"

    monkeypatch.setattr(observability, "_initialized", True)
    monkeypatch.setattr(observability, "_configured", True)
    monkeypatch.setattr(observability, "LANGFUSE_TRACING_ENABLED", "true")

    with observability.langfuse_request_trace(
        session_id="s",
        user_id="u",
        request_id="r",
        input_data={"q": "hi"},
        metadata={"mode": "test"},
    ) as observation:
        observability.update_langfuse_observation(observation, output="ok")

    observability.flush_langfuse()

    assert fake_client.kwargs["name"] == observability.TRACE_NAME
    assert fake_client.observation.updated["output"] == "ok"
    assert fake_client.flushed is True


def test_observability_import_failure_and_observation_update_errors(monkeypatch):
    """验证 Langfuse SDK 不可用和 observation.update 异常时不会影响主流程。"""
    import observability

    monkeypatch.setattr(observability, "_initialized", False)
    monkeypatch.setattr(observability, "_warned_missing_config", False)
    monkeypatch.setattr(observability, "LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setattr(observability, "LANGFUSE_SECRET_KEY", "sk")
    monkeypatch.delitem(sys.modules, "langfuse", raising=False)

    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "langfuse":
            raise ImportError("missing langfuse")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    assert observability.init_langfuse() is None
    assert observability.get_langfuse_client() is None

    class BadObservation:
        def update(self, **kwargs):
            raise RuntimeError("bad update")

    observability.update_langfuse_observation(BadObservation(), output="ignored")
