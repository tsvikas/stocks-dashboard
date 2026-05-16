"""Download the OG preview image for the deployed Streamlit app.

Usage:
    uv run python scripts/fetch_preview.py [APP_URL] [OUT_PATH]
    uv run python scripts/fetch_preview.py --debug [APP_URL]

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


class HeadDumpParser(HTMLParser):
    """Collects <title>, every <meta>, and every <link> tag."""

    def __init__(self) -> None:
        super().__init__()
        self.title: str = ""
        self._in_title = False
        self.metas: list[dict[str, str]] = []
        self.links: list[dict[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        d = {k: (v or "") for k, v in attrs}
        if tag == "meta":
            self.metas.append(d)
        elif tag == "link":
            self.links.append(d)
        elif tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data


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


def debug_dump(app_url: str) -> None:
    html = _get(app_url).decode("utf-8", errors="replace")
    print(f"# fetched {len(html):,} chars from {app_url}\n")
    parser = HeadDumpParser()
    parser.feed(html)
    print(f"<title>: {parser.title.strip()!r}\n")
    print(f"## <meta> tags ({len(parser.metas)}):")
    for m in parser.metas:
        print(f"  {m}")
    print(f"\n## <link> tags ({len(parser.links)}):")
    for ln in parser.links:
        print(f"  {ln}")
    print("\n## image-ish strings in body:")
    for m in re.finditer(r'https?://[^\s"\'<>]+\.(?:png|jpe?g|webp|gif|svg)', html, re.I):
        print(f"  {m.group(0)}")

    bundle_paths = sorted(
        {ln["href"] for ln in parser.links if ln.get("href", "").endswith(".js")}
    )
    print(f"\n## scanning {len(bundle_paths)} JS bundles for preview/screenshot strings")
    keywords = ("preview", "screenshot", "thumbnail", "snapshot", "ogImage", "og_image")
    pattern = re.compile(
        r'(?:"|\')([^"\']{0,200}(?:'
        + "|".join(keywords)
        + r')[^"\']{0,200})(?:"|\')',
        re.IGNORECASE,
    )
    base = re.match(r"https?://[^/]+", app_url).group(0)
    for path in bundle_paths:
        full = path if path.startswith("http") else base + path
        try:
            body = _get(full).decode("utf-8", errors="replace")
        except Exception as exc:
            print(f"  [skip] {path}: {exc}")
            continue
        hits = {m.group(1) for m in pattern.finditer(body)}
        if hits:
            print(f"\n  >>> {path}")
            for h in sorted(hits):
                print(f"      {h}")


def main(argv: list[str]) -> int:
    args = argv[1:]
    if args and args[0] == "--debug":
        debug_dump(args[1] if len(args) > 1 else DEFAULT_APP_URL)
        return 0

    app_url = args[0] if args else DEFAULT_APP_URL
    out = Path(args[1]) if len(args) > 1 else DEFAULT_OUT

    img_url = find_og_image(app_url)
    print(f"og:image -> {img_url}")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(_get(img_url))
    print(f"saved    -> {out} ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

