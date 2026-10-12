"""One OpenRouter chat helper shared by the factor lab and the event agent.

The key comes from the macOS keychain at call time and is never written. Models rotate on any error;
every attempt (model, status, prompt hash, token usage) goes to one append-only usage log, and the
daily call cap is counted from that log across all callers.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
USAGE = ROOT / 'output/llm_usage.jsonl'
ENDPOINT = 'https://openrouter.ai/api/v1/chat/completions'
MODELS = ('nvidia/nemotron-3-super-120b-a12b:free', 'google/gemma-4-31b-it:free',
          'google/gemma-4-26b-a4b-it:free', 'dots-studio/dots-3-note-preview:free')
DAILY_CAP = 40


class LLMError(RuntimeError):
    pass


def api_key():
    out = subprocess.run(['security', 'find-generic-password', '-a', 'quantlab', '-s', 'openrouter_api_key', '-w'],
                         capture_output=True, text=True, check=True)
    return out.stdout.strip()


def http_post(body, key, timeout=240):
    request = urllib.request.Request(ENDPOINT, data=json.dumps(body).encode(), method='POST',
                                     headers={'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def calls_today(usage=USAGE):
    path = Path(usage)
    if not path.exists():
        return 0
    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    return sum(1 for line in path.read_text().splitlines() if json.loads(line)['at'].startswith(today))


def chat(prompt, *, caller, usage=USAGE, models=MODELS, max_tokens=3000, post=http_post, key=None, cap=DAILY_CAP):
    """Return ``(model, text, prompt_sha)``; raise ``LLMError`` when every model fails or the cap is hit."""
    if calls_today(usage) >= cap:
        raise LLMError('daily_llm_call_cap_reached')
    digest = hashlib.sha256(prompt.encode()).hexdigest()[:16]
    key = key or api_key()
    log = Path(usage)
    log.parent.mkdir(parents=True, exist_ok=True)
    for model in models:
        record = dict(at=datetime.now(timezone.utc).isoformat(), caller=caller, model=model, prompt_sha=digest)
        try:
            reply = post(dict(model=model, messages=[{'role': 'user', 'content': prompt}], max_tokens=max_tokens), key)
            if reply.get('error'):
                raise LLMError(str(reply['error'].get('code') or reply['error'])[:80])
            text = reply['choices'][0]['message']['content'] or ''
            record.update(status='ok', usage=reply.get('usage', {}))
            with log.open('a') as handle:
                handle.write(json.dumps(record) + '\n')
            return model, text, digest
        except Exception as exc:  # rotate on rate limits and provider errors, recording every failure
            record.update(status='error', error=f'{type(exc).__name__}:{str(exc)[:120]}')
            with log.open('a') as handle:
                handle.write(json.dumps(record) + '\n')
    raise LLMError('all_models_failed')
