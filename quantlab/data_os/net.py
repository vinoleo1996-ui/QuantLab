"""代理访问失败时改走本机网络。研究读取不使用这里。"""

from __future__ import annotations

from contextlib import contextmanager
import os


_PROXY_KEYS = ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY")
_BYPASS_KEYS = ("NO_PROXY", "no_proxy")
_RETRY_MARKERS = (
    "ProxyError",
    "proxy",
    "RemoteDisconnected",
    "Max retries exceeded",
    "ConnectTimeout",
    "ConnectionError",
    "ReadTimeout",
    "Tunnel connection failed",
    "502",
    "503",
    "504",
    "Bad Gateway",
)


def proxy_configured() -> bool:
    return any(os.environ.get(key) for key in _PROXY_KEYS)


def should_retry_direct(error: BaseException | str | None) -> bool:
    if not error or not proxy_configured():
        return False
    text = str(error)
    return any(marker.lower() in text.lower() for marker in _RETRY_MARKERS)


@contextmanager
def direct_network():
    saved = {key: os.environ.pop(key) for key in _PROXY_KEYS if key in os.environ}
    previous_bypass = {key: os.environ.get(key) for key in _BYPASS_KEYS}
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"
    try:
        yield
    finally:
        for key in _BYPASS_KEYS:
            if previous_bypass[key] is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous_bypass[key]
        os.environ.update(saved)


def fetch_bytes(url: str, headers: dict | None = None, timeout: int = 30) -> bytes:
    import requests
    try:
        response = requests.get(url, headers=headers or {}, timeout=timeout)
        response.raise_for_status()
        return response.content
    except Exception as exc:
        if not should_retry_direct(exc):
            raise
        with direct_network():
            session = requests.Session()
            session.trust_env = False
            response = session.get(url, headers=headers or {}, timeout=timeout)
            response.raise_for_status()
            return response.content
