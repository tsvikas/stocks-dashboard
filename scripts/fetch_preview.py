"""Download the OG preview image for the deployed Streamlit app.

Usage:
    uv run python scripts/fetch_preview.py [APP_URL] [OUT_PATH]

Defaults: https://tsvikas-stocks-dashboard.streamlit.app/ -> docs/preview.png
"""

from __future__ import annotations

import http.cookiejar
import re
import sys
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

DEFAULT_APP_URL = "https://tsvikas-stocks-dashboard.streamlit.app/"
DEFAULT_OUT = Path("docs/preview.png")
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

_opener = urllib.request.build_opener(
    urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())
)
_opener.addheaders = [
    ("User-Agent", USER_AGENT),
    ("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"),
    ("Accept-Language", "en-US,en;q=0.9"),
]


class OgImageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.url: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "meta":
            return
        d = dict(attrs)
        prop = (d.get("property") or d.get("name") or "").lower()
        if prop in {"og:image", "twitter:image"} and d.get("content"):
            self.url = self.url or d["content"]


def _get(url: str) -> bytes:
    with _opener.open(url, timeout=30) as resp:
        return resp.read()


def find_og_image(app_url: str) -> str:
    html = _get(app_url).decode("utf-8", errors="replace")
    parser = OgImageParser()
    parser.feed(html)
    if parser.url:
        return parser.url
    m = re.search(
        r'<meta[^>]+(?:property|name)=["\'](?:og:image|twitter:image)["\'][^>]+content=["\']([^"\']+)',
        html,
        re.IGNORECASE,
    )
    if m:
        return m.group(1)
    raise SystemExit("No og:image / twitter:image meta tag found.")


def main(argv: list[str]) -> int:
    app_url = argv[1] if len(argv) > 1 else DEFAULT_APP_URL
    out = Path(argv[2]) if len(argv) > 2 else DEFAULT_OUT

    img_url = find_og_image(app_url)
    print(f"og:image -> {img_url}")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(_get(img_url))
    print(f"saved    -> {out} ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
