"""Crawl bcit.ca into data_new/ in the same txt format as the existing corpus.

Discovery:
- Course outlines: /wp-json/bcit/outlines/v1/load_subjects_term/{term} —
  one JSON per term lists every (subject, course, CRN). We keep terms
  >= MIN_TERM and, per course, only the latest term (first CRN section).
- Courses / programs / site pages: WordPress sitemaps.

Politeness: 3 concurrent requests, ~0.35s stagger, retries with backoff.
Resumable: existing non-empty output files are skipped on re-run.

Usage:
    .venv/bin/python crawl_bcit.py [--phase all|worklist|outlines|courses|programs|pages]
"""

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import aiohttp
from bs4 import BeautifulSoup

BASE = "https://www.bcit.ca"
OUT_DIR = Path(os.environ.get("CRAWL_OUT", "./data_new"))
STATE_DIR = OUT_DIR / "_state"

MIN_TERM = "202430"  # September 2024 intake onwards
TERMS = ["202430", "202510", "202520", "202530", "202610", "202620", "202630"]

CONCURRENCY = 3
DELAY_RANGE = (0.25, 0.45)
RETRIES = 3
TIMEOUT = aiohttp.ClientTimeout(total=45)
UA = "bcit-rag-chatbot/2.0 (personal academic RAG project; contact: juhyun5328@gmail.com)"

# bcit.ca path prefix -> output category (mirrors the old corpus layout)
PAGE_CATEGORIES = {
    "/about/": "about",
    "/accessibility/": "accessibility",
    "/admission/": "admission",
    "/advising/": "advising",
    "/global-partnerships/": "global_partnerships",
    "/international-applicants/": "international_applicants",
    "/international-students/": "international_students",
    "/learning-commons/": "learning_commons",
    "/student-services/": "student_services",
    "/test-centre/": "test_centre",
}

# BCIT Student Association (separate WordPress site)
BCITSA_BASE = "https://www.bcitsa.ca"
BCITSA_SITEMAP = f"{BCITSA_BASE}/page-sitemap.xml"
BCITSA_SKIP = ("terms-and-conditions", "privacy-policy", "-minutes")

PROGRAM_SITEMAPS = ["wp-sitemap-posts-program-1.xml", "wp-sitemap-posts-program_umbrella-1.xml"]
COURSE_SITEMAPS = [f"wp-sitemap-posts-pts_course-{i}.xml" for i in range(1, 5)]
PAGE_SITEMAPS = ["wp-sitemap-posts-page-1.xml", "wp-sitemap-posts-page-2.xml"]

STRIP_TAGS = ["script", "style", "noscript", "svg", "iframe", "form", "nav", "header", "footer", "aside", "button"]
STRIP_SELECTORS = [
    ".site-header", ".site-footer", ".page-hero__media", ".breadcrumbs", ".a11y-visual-hide",
    ".skip-link", "#site-header", "#site-footer", ".social-share", ".back-to-top",
    ".sidebar", ".site-alert", ".cookie", ".modal",
]


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^\w\s,&-]", "", text)
    text = re.sub(r"\s+", "_", text)
    return text[:120] or "untitled"


class Fetcher:
    def __init__(self, session: aiohttp.ClientSession):
        self.session = session
        self.sem = asyncio.Semaphore(CONCURRENCY)
        self.done = 0
        self.errors = []

    async def get(self, url: str, as_json=False):
        async with self.sem:
            await asyncio.sleep(random.uniform(*DELAY_RANGE))
            for attempt in range(RETRIES):
                try:
                    async with self.session.get(url) as resp:
                        if resp.status == 404:
                            return None
                        if resp.status >= 500:
                            raise aiohttp.ClientError(f"HTTP {resp.status}")
                        resp.raise_for_status()
                        return await (resp.json() if as_json else resp.text())
                except Exception as e:
                    if attempt == RETRIES - 1:
                        self.errors.append(f"{url}\t{e}")
                        return None
                    await asyncio.sleep(2 ** attempt * 2)

    def tick(self, total: int, label: str):
        self.done += 1
        if self.done % 100 == 0 or self.done == total:
            log(f"{label}: {self.done}/{total} (errors: {len(self.errors)})")


# ---------------------------------------------------------------- HTML -> txt

def node_to_text(node) -> str:
    """Convert a content container to the corpus text format."""
    lines = []

    def emit(text):
        text = re.sub(r"[ \t ]+", " ", text).strip()
        if text:
            lines.append(text)

    def walk(el):
        name = getattr(el, "name", None)
        if name is None:
            return
        if name in ("h1", "h2"):
            emit("\n## " + el.get_text(" ", strip=True))
        elif name in ("h3", "h4", "h5"):
            emit("\n### " + el.get_text(" ", strip=True))
        elif name == "p":
            emit(el.get_text(" ", strip=True))
        elif name == "li":
            txt = el.get_text(" ", strip=True)
            if txt:
                emit("- " + txt)
        elif name == "dt":
            txt = el.get_text(" ", strip=True)
            dd = el.find_next_sibling("dd")
            emit(f"{txt}: {dd.get_text(' ', strip=True) if dd else ''}")
        elif name == "tr":
            cells = [c.get_text(" ", strip=True) for c in el.find_all(["th", "td"])]
            cells = [c for c in cells if c]
            if cells:
                emit(" | ".join(cells))
        else:
            for child in el.children:
                walk(child)

    walk(node)
    # drop breadcrumb fragments emitted before the first heading
    first_heading = next((i for i, l in enumerate(lines) if l.lstrip().startswith("##")), 0)
    text = "\n".join(lines[first_heading:])
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_main(html: str) -> tuple[str, BeautifulSoup]:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(STRIP_TAGS):
        tag.decompose()
    for sel in STRIP_SELECTORS:
        for tag in soup.select(sel):
            tag.decompose()
    main = soup.find("main") or soup.find(id="main-content") or soup.find("article") or soup.body
    return (node_to_text(main) if main else ""), soup


def page_title(soup: BeautifulSoup) -> str:
    h1 = soup.find("h1")
    if h1:
        return h1.get_text(" ", strip=True)
    if soup.title:
        return soup.title.get_text().split("|")[0].strip()
    return "Untitled"


def write_doc(path: Path, header: str, body: str) -> bool:
    if len(body) < 80:  # empty shells / soft 404s
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{header}\n\n{'=' * 80}\n\n{body}\n", encoding="utf-8")
    return True


# ------------------------------------------------------------------ sitemaps

async def sitemap_urls(fetcher: Fetcher, sitemap: str) -> list[str]:
    xml = await fetcher.get(f"{BASE}/{sitemap}")
    return re.findall(r"<loc>(.*?)</loc>", xml or "")


# ------------------------------------------------------------------ worklist

async def build_worklist(fetcher: Fetcher) -> dict:
    """One API call per term -> latest outline per course (term >= MIN_TERM)."""
    best = {}  # "SUBJ NUM" -> {term, crn, name, subject_name}
    for term in TERMS:
        if term < MIN_TERM:
            continue
        data = await fetcher.get(f"{BASE}/wp-json/bcit/outlines/v1/load_subjects_term/{term}", as_json=True)
        if not data or not data.get("success"):
            log(f"worklist: term {term} unavailable, skipping")
            continue
        n = 0
        for subj in data["data"]["courses"].values():
            code = subj["code"].strip().upper()
            for num, sections in subj["courses"].items():
                if not sections:
                    continue
                key = f"{code} {num}"
                # terms are zero-padded 6-digit strings: lexical order == chronological
                if key not in best or term > best[key]["term"]:
                    best[key] = {
                        "term": term,
                        "crn": str(sections[0]["crn"]).strip(),
                        "name": sections[0].get("name", ""),
                        "subject_name": subj.get("subject", ""),
                    }
                    n += 1
        log(f"worklist: term {term} -> {n} course updates (cumulative {len(best)})")
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    (STATE_DIR / "outline_worklist.json").write_text(json.dumps(best, indent=1))
    log(f"worklist: {len(best)} unique courses, latest term each, saved")
    return best


# ------------------------------------------------------------------- outlines

async def crawl_one_outline(fetcher: Fetcher, code: str, info: dict, total: int):
    dept, num = code.split()
    out = OUT_DIR / "course_outline" / f"{dept}_{num}_{info['term']}.txt"
    if out.exists() and out.stat().st_size > 300:
        fetcher.tick(total, "outlines")
        return
    url = f"{BASE}/outlines/{info['term']}{info['crn']}/"
    html = await fetcher.get(url)
    if html:
        body, soup = extract_main(html)
        header = (
            f"URL: {url}\nCourse: {code}\nField: {info['name']}\n"
            f"Subject: {info['subject_name']}\nTerm: {info['term']}\nCRN: {info['crn']}"
        )
        write_doc(out, header, body)
    fetcher.tick(total, "outlines")


async def crawl_outlines(fetcher: Fetcher, worklist: dict):
    fetcher.done = 0
    items = sorted(worklist.items())
    log(f"outlines: fetching {len(items)} outline pages (latest term per course)")
    await asyncio.gather(*[crawl_one_outline(fetcher, c, i, len(items)) for c, i in items])


# -------------------------------------------------------------------- courses

async def crawl_one_course(fetcher: Fetcher, url: str, total: int):
    slug = url.rstrip("/").rsplit("/", 1)[-1]                  # e.g. 3d-gis-gist-8100
    m = re.search(r"-([a-z]{2,4})-(\d{4}[a-z]?)$", slug)
    code = f"{m.group(1).upper()} {m.group(2).upper()}" if m else ""
    out = OUT_DIR / "course" / f"{slugify(slug.replace('-', ' '))}.txt"
    if out.exists() and out.stat().st_size > 300:
        fetcher.tick(total, "courses")
        return
    html = await fetcher.get(url)
    if html:
        body, soup = extract_main(html)
        title = page_title(soup)
        header = f"URL: {url}\nTitle: {title}\nType: Course"
        if code:
            header += f"\nCourse Code: {code}"
        write_doc(out, header, body)
    fetcher.tick(total, "courses")


async def crawl_courses(fetcher: Fetcher):
    urls = []
    for sm in COURSE_SITEMAPS:
        urls += await sitemap_urls(fetcher, sm)
    urls = sorted(set(u for u in urls if "/courses/" in u))
    fetcher.done = 0
    log(f"courses: {len(urls)} catalog pages")
    await asyncio.gather(*[crawl_one_course(fetcher, u, len(urls)) for u in urls])


# ------------------------------------------------------------------- programs

async def crawl_one_program(fetcher: Fetcher, url: str, total: int):
    slug = url.rstrip("/").rsplit("/", 1)[-1]
    out = OUT_DIR / "program" / f"{slugify(slug.replace('-', ' '))}.txt"
    if out.exists() and out.stat().st_size > 300:
        fetcher.tick(total, "programs")
        return
    html = await fetcher.get(url)
    if html:
        body, soup = extract_main(html)
        title = page_title(soup)
        write_doc(out, f"URL: {url}\nTitle: {title}\nType: Program", body)
    fetcher.tick(total, "programs")


async def crawl_programs(fetcher: Fetcher):
    urls = []
    for sm in PROGRAM_SITEMAPS:
        urls += await sitemap_urls(fetcher, sm)
    urls = sorted(set(urls))
    fetcher.done = 0
    log(f"programs: {len(urls)} pages")
    await asyncio.gather(*[crawl_one_program(fetcher, u, len(urls)) for u in urls])


# ---------------------------------------------------------------------- pages

def page_category(url: str) -> str | None:
    path = url.replace(BASE, "")
    for prefix, cat in PAGE_CATEGORIES.items():
        if path.startswith(prefix):
            return cat
    return None


async def crawl_one_page(fetcher: Fetcher, url: str, cat: str, total: int):
    path = url.replace(BASE, "").strip("/")
    out = OUT_DIR / cat / f"{slugify(path.replace('/', ' '))}.txt"
    if out.exists() and out.stat().st_size > 300:
        fetcher.tick(total, "pages")
        return
    html = await fetcher.get(url)
    if html:
        body, soup = extract_main(html)
        write_doc(out, f"URL: {url}\nTitle: {page_title(soup)}", body)
    fetcher.tick(total, "pages")


async def crawl_pages(fetcher: Fetcher):
    urls = []
    for sm in PAGE_SITEMAPS:
        urls += await sitemap_urls(fetcher, sm)
    targets = sorted({u: c for u in set(urls) if (c := page_category(u))}.items())
    fetcher.done = 0
    log(f"pages: {len(targets)} pages across {len(PAGE_CATEGORIES)} categories")
    await asyncio.gather(*[crawl_one_page(fetcher, u, c, len(targets)) for u, c in targets])


# --------------------------------------------------------------------- bcitsa

def bcitsa_category(url: str) -> str:
    path = url.replace(BCITSA_BASE, "")
    if path.startswith("/campus-life/") or path.startswith("/wellbeing/"):
        return "campus_life"
    if path.startswith("/about-us/"):
        return "student_association"
    return "bcitsa"


async def crawl_one_bcitsa(fetcher: Fetcher, url: str, total: int):
    path = url.replace(BCITSA_BASE, "").strip("/") or "home"
    out = OUT_DIR / bcitsa_category(url) / f"bcitsa_{slugify(path.replace('/', ' '))}.txt"
    if out.exists() and out.stat().st_size > 300:
        fetcher.tick(total, "bcitsa")
        return
    html = await fetcher.get(url)
    if html:
        body, soup = extract_main(html)
        write_doc(out, f"URL: {url}\nTitle: {page_title(soup)} - BCITSA", body)
    fetcher.tick(total, "bcitsa")


async def crawl_bcitsa(fetcher: Fetcher):
    xml = await fetcher.get(BCITSA_SITEMAP)
    urls = sorted({
        u for u in re.findall(r"<loc>(.*?)</loc>", xml or "")
        if not any(s in u for s in BCITSA_SKIP)
    })
    fetcher.done = 0
    log(f"bcitsa: {len(urls)} pages")
    await asyncio.gather(*[crawl_one_bcitsa(fetcher, u, len(urls)) for u in urls])


# ----------------------------------------------------------------------- main

async def main(phase: str):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": UA, "Accept-Language": "en"}
    async with aiohttp.ClientSession(headers=headers, timeout=TIMEOUT) as session:
        fetcher = Fetcher(session)

        worklist = None
        if phase in ("all", "worklist", "outlines"):
            cache = STATE_DIR / "outline_worklist.json"
            if phase == "outlines" and cache.exists():
                worklist = json.loads(cache.read_text())
                log(f"worklist: loaded {len(worklist)} cached entries")
            else:
                worklist = await build_worklist(fetcher)

        if phase in ("all", "outlines"):
            await crawl_outlines(fetcher, worklist)
        if phase in ("all", "courses"):
            await crawl_courses(fetcher)
        if phase in ("all", "programs"):
            await crawl_programs(fetcher)
        if phase in ("all", "pages"):
            await crawl_pages(fetcher)
        if phase in ("all", "bcitsa"):
            await crawl_bcitsa(fetcher)

        if fetcher.errors:
            STATE_DIR.mkdir(parents=True, exist_ok=True)
            (STATE_DIR / "errors.log").write_text("\n".join(fetcher.errors))
            log(f"done with {len(fetcher.errors)} errors -> {STATE_DIR / 'errors.log'}")

    counts = {d.name: len(list(d.glob("*.txt"))) for d in sorted(OUT_DIR.iterdir()) if d.is_dir() and d.name != "_state"}
    log(f"final counts: {counts}  total={sum(counts.values())}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all", choices=["all", "worklist", "outlines", "courses", "programs", "pages", "bcitsa"])
    args = ap.parse_args()
    asyncio.run(main(args.phase))
