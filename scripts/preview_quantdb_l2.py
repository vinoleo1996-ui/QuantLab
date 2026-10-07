"""Preview QuantDB L2 column names.  Do not download the training table."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quantlab.data.quantdb_l2_preview import assess_l2_preview, preview_column_names  # noqa: E402


def main() -> None:
    key = os.environ.get("QUANTDB_API_KEY", "").strip()
    if not key:
        raise SystemExit("QUANTDB_API_KEY_required")
    url = "https://www.quantdb.cn/api/v1/data/preview?" + urllib.parse.urlencode(
        {"category_id": "6", "sub_category": "l2_factors", "limit": "1"}
    )
    request = urllib.request.Request(url, headers={"X-API-Key": key, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        result = assess_l2_preview(columns=())
        result["reasons"] = tuple(result["reasons"]) + (f"preview_http_{exc.code}",)
        raw = b""
    else:
        payload = json.loads(raw)
        result = assess_l2_preview(
            columns=preview_column_names(payload),
            preview_response_sha256=hashlib.sha256(raw).hexdigest(),
        )
    destination = ROOT / "output" / "quantdb_l2_preview_20260930"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: result[key] for key in ("status", "present_fields", "missing_fields", "reasons", "training_allowed", "sample_count")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
