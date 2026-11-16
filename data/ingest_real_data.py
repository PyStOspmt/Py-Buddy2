"""Ingest real-world product data into the project's SQLite catalog schema.

Why this file exists
--------------------
Every number in docs/02 so far comes from a catalog this repository wrote for
itself (data/generate_marketplace_scale.py). A synthetic catalog can only prove
that the code agrees with the assumptions of its own generator. This script
replaces the input, not the code: public, openly licensed datasets are mapped
onto the *existing* schema (merchants / products / product_eav_meta) so that the
existing harnesses -- attribute_algebra, benchmark_cases.load_snapshot, metrics
-- run on real listings unchanged.

Sources (all public, no API key, no login)
------------------------------------------
off     Open Food Facts product database, HF mirror
        `openfoodfacts/product-database` (default/food, 4 762 888 products),
        ODbL 1.0. Real EAV in the wild: multilingual names, unit-carrying
        nutrients, additive/allergen/label tag sets.
prices  `openfoodfacts/open-prices` (default/prices, 318 187 observations),
        ODbL 1.0. Real price, real currency, real retail store per product
        code -- the multi-seller dimension taken from shops that exist.
wdc     `wdc/products-2017` (Peeters & Bizer; CC-BY 4.0, Web Data Commons).
        Cross-shop offers already grouped by `cluster_id` (same real product,
        different shops, different languages) with real price strings and
        free-text specification tables.

Nothing here is generated, rounded to look nice, or filled in from a template.
Where a source has no field that the schema demands, the gap is written into
`ingest_meta` and the column is left at a documented neutral constant.

Usage
-----
    python data/ingest_real_data.py --source off    --limit 3000
    python data/ingest_real_data.py --source prices --limit 20000
    python data/ingest_real_data.py --source wdc    --limit 4000
    python data/ingest_real_data.py --source all
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
SERVER = "https://datasets-server.huggingface.co"
# A descriptive User-Agent is Open Food Facts' API policy and is polite
# anywhere else. It carries no personal data.
USER_AGENT = ("PyBuddyResearch/0.1 "
              "(academic study; contact: ospmt@users.noreply.github.com)")
PAGE = 100          # rows per request -- the datasets-server maximum
#: Seconds between requests. The server rate-limits bursts, and a benchmark
#: that hammers a public mirror is a worse benchmark for it.
PAGE_DELAY = 0.4

DDL = """
CREATE TABLE merchants (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    rating REAL NOT NULL,
    delivery_days INTEGER NOT NULL,
    is_official INTEGER NOT NULL
);
CREATE TABLE products (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    brand TEXT NOT NULL,
    price REAL,
    merchant_id TEXT NOT NULL,
    stock_quantity INTEGER NOT NULL,
    performance_score REAL NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE product_eav_meta (
    meta_id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id TEXT NOT NULL,
    meta_key TEXT NOT NULL,
    meta_value TEXT NOT NULL
);
CREATE TABLE commercial (
    product_id TEXT PRIMARY KEY,
    sponsored INTEGER NOT NULL DEFAULT 0,
    commission_pct REAL NOT NULL DEFAULT 0.0
);
CREATE TABLE ingest_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE INDEX idx_eav_product ON product_eav_meta(product_id);
CREATE INDEX idx_eav_key ON product_eav_meta(meta_key);
CREATE INDEX idx_products_category ON products(category);
CREATE INDEX idx_commercial_sponsored ON commercial(sponsored);
"""


# --------------------------------------------------------------------------
# Transport
# --------------------------------------------------------------------------

def _get_json(path: str, params: Dict[str, object],
              tries: int = 10) -> Dict[str, object]:
    """GET with a rate-limit-aware backoff.

    The public datasets-server answers 429 once a burst exceeds its window; the
    fix is patience, not a shorter timeout. Retry-After is honoured when the
    server sends it, and the wait grows otherwise.
    """
    url = f"{SERVER}{path}?{urllib.parse.urlencode(params)}"
    last: Optional[Exception] = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=180) as resp:
                payload = json.loads(resp.read())
            time.sleep(PAGE_DELAY)
            return payload
        except urllib.error.HTTPError as exc:                 # noqa: PERF203
            last = exc
            if exc.code not in (429, 500, 502, 503, 504):
                break
            header = (exc.headers.get("Retry-After") if exc.headers else "") or ""
            wait = float(header) if header.strip().isdigit() else \
                min(90.0, 5.0 * (attempt + 1) ** 1.4)
            if attempt >= tries - 1:
                break
            print(f"    [rate-limit] HTTP {exc.code}; waiting {wait:.0f}s "
                  f"(attempt {attempt + 1}/{tries})", flush=True)
            time.sleep(wait)
        except Exception as exc:                              # noqa: BLE001
            last = exc
            if attempt >= tries - 1:
                break
            time.sleep(min(60.0, 2.0 * (attempt + 1)))
    raise RuntimeError(f"datasets-server request failed after {tries} tries: {url}\n"
                       f"  {type(last).__name__}: {last}")



def page_rows(dataset: str, config: str, split: str, offset: int,
              length: int = PAGE,
              where: Optional[str] = None) -> Tuple[List[dict], int]:
    """One page of rows plus the total row count of the split."""
    params: Dict[str, object] = {"dataset": dataset, "config": config,
                                 "split": split, "offset": offset, "length": length}
    if where:
        params["where"] = where
    data = _get_json("/rows", params)
    rows = [r.get("row", r) for r in data.get("rows", [])]
    return rows, int(data.get("num_rows_total") or 0)


def iter_rows(dataset: str, config: str, split: str, limit: int,
              where: Optional[str] = None, label: str = "") -> Iterable[dict]:
    """Page through a split until `limit` rows have been yielded.

    The first N rows of the split are taken in storage order, deliberately:
    picking "products that look interesting" would be a selection choice, and a
    selection choice is how a benchmark ends up flattering the system built on
    top of it.
    """
    offset, yielded, pages = 0, 0, 0
    while yielded < limit:
        want = min(PAGE, limit - yielded)
        rows, total = page_rows(dataset, config, split, offset, want, where)
        if not rows:
            break
        for r in rows:
            yield r
        yielded += len(rows)
        offset += len(rows)
        pages += 1
        if pages % 20 == 0 or yielded >= limit:
            cap = min(limit, total) if total else limit
            print(f"    [{label}] {yielded}/{cap} rows", flush=True)
        if len(rows) < want:
            break


# --------------------------------------------------------------------------
# SQLite helpers
# --------------------------------------------------------------------------

def open_db(path: str) -> sqlite3.Connection:
    if os.path.exists(path):
        os.remove(path)
    conn = sqlite3.connect(path)
    conn.executescript(DDL)
    return conn


def prepare_build_target(out_path: str, overwrite: bool) -> str:
    """Fail fast before any download if the target may not be replaced.

    F36: the old workflow deleted the existing database up front -- one
    failed run destroyed the previous evidence. The new database is built
    into a scratch file and published atomically at the end; replacing an
    existing target requires the explicit `overwrite` decision.
    """
    if os.path.exists(out_path) and not overwrite:
        raise FileExistsError(
            f"target exists: {out_path} -- pass --rebuild to replace it")
    return out_path + ".build"


def publish_build_target(build_path: str, out_path: str) -> None:
    """Atomic publication: previous target kept as .bak (rollback path),
    the new file swapped in with os.replace -- never a torn database."""
    if os.path.exists(out_path):
        bak = out_path + ".bak"
        if os.path.exists(bak):
            os.remove(bak)
        os.replace(out_path, bak)
    os.replace(build_path, out_path)


def record(conn: sqlite3.Connection, key: str, value: object) -> None:
    conn.execute("INSERT OR REPLACE INTO ingest_meta(key, value) VALUES (?, ?)",
                 (key, str(value)))


def neutral_commercial(conn: sqlite3.Connection, dataset: str) -> None:
    """Fill `commercial` with documented neutral constants.

    None of the three sources carries a sponsor or commission field, so
    every product gets sponsored=0 / commission_pct=0.0. The rows are
    written (rather than leaving the table empty) so the FIC dynamic
    channel is evaluated on a fully-populated commercial state instead of
    being vacuous; the provenance goes into ingest_meta.
    """
    conn.execute(
        "INSERT INTO commercial (product_id, sponsored, commission_pct) "
        "SELECT id, 0, 0.0 FROM products")
    record(conn, f"commercial_source:{dataset}",
           "synthetic-neutral: the source has no sponsor/commission field; "
           "all rows sponsored=0, commission_pct=0.0")


class EavWriter:
    """Buffers (product_id, key, value) triples so inserts stay batched."""

    def __init__(self, conn: sqlite3.Connection, batch: int = 5000):
        self.conn = conn
        self.batch = batch
        self.rows: List[Tuple[str, str, str]] = []
        self.n = 0
        #: F36: per-reason loss counters -- a dropped value must be
        #: accounted, not silently filtered.
        self.dropped_empty = 0
        self.dropped_long = 0

    def add(self, product_id: str, key: str, value: object) -> None:
        if value is None:
            return
        if isinstance(value, bool):
            value = "1" if value else "0"
        text = str(value).strip()
        if not text:
            self.dropped_empty += 1
            return
        if len(text) > 240:
            self.dropped_long += 1
            return
        self.rows.append((product_id, key, text))
        if len(self.rows) >= self.batch:
            self.flush()

    def flush(self) -> int:
        if not self.rows:
            return 0
        self.conn.executemany(
            "INSERT INTO product_eav_meta(product_id, meta_key, meta_value) "
            "VALUES (?,?,?)", self.rows)
        self.n += len(self.rows)
        self.rows.clear()
        return self.n


# --------------------------------------------------------------------------
# Source: Open Food Facts product database
# --------------------------------------------------------------------------

# Fields written verbatim under their own source name. No synonym list and no
# translation: if the algebra is to survive a real catalog, it has to survive
# the catalog's own field names and value spellings.
OFF_SCALARS = (
    "quantity", "product_quantity", "product_quantity_unit", "serving_size",
    "serving_quantity", "nova_group", "nutriscore_grade", "nutriscore_score",
    "environmental_score_grade", "environmental_score_score", "additives_n",
    "ingredients_n", "no_nutrition_data", "completeness", "stores", "origins",
    "manufacturing_places", "last_editor", "creator",
)
OFF_LISTS = (
    ("labels_tags", "labels"),
    ("allergens_tags", "allergens"),
    ("traces_tags", "traces"),
    ("ingredients_analysis_tags", "ingredients_analysis"),
    ("additives_tags", "additives"),
    ("vitamins_tags", "vitamins"),
    ("minerals_tags", "minerals"),
    ("origins_tags", "origins"),
    ("countries_tags", "countries"),
    ("categories_tags", "category_tag"),
    ("packaging_tags", "packaging"),
    ("food_groups_tags", "food_group"),
)


def unwrap_name(value: object) -> str:
    """OFF names arrive as [{"lang": "main", "text": ...}, ...].

    Preference order main -> en -> first available. The language actually used
    is also stored as an EAV row, so a later measurement can see which spelling
    it was handed instead of assuming one.
    """
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        first = ""
        for item in value:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            if str(item.get("lang") or "") in ("main", "en"):
                return text
            if not first:
                first = text
        return first
    return ""


def nutriments_rows(row: dict) -> List[Tuple[str, str]]:
    """(key, value) pairs for every nutrient this product declares.

    The unit the source states travels with the number, which is the point:
    `g`, `mg`, `ug`, `kcal` and `kJ` are what a real catalog writes, and whether
    the algebra can canonicalise them is a property of the algebra, not of this
    file.
    """
    out: List[Tuple[str, str]] = []
    for n in row.get("nutriments") or []:
        if not isinstance(n, dict):
            continue
        name = str(n.get("name") or "").strip()
        if not name:
            continue
        unit = str(n.get("unit") or "").strip()
        for field, suffix in (("100g", "per100g"), ("value", "value"),
                              ("serving", "perServing")):
            val = n.get(field)
            if val is None or val == "":
                continue
            out.append((f"nutrition_{name}_{suffix}", f"{val} {unit}".strip()))
    return out


def ingest_off(limit: int, out_path: str, overwrite: bool = False) -> Dict[str, object]:
    build_path = prepare_build_target(out_path, overwrite)
    conn = open_db(build_path)
    eav = EavWriter(conn)
    conn.execute("INSERT INTO merchants VALUES (?,?,?,?,?)",
                 ("m_openfoodfacts", "Open Food Facts (crowdsourced, ODbL 1.0)",
                  0.0, 0, 0))
    seen, named = set(), 0
    for row in iter_rows("openfoodfacts/product-database", "default", "food",
                         limit, label="off"):
        code = str(row.get("code") or "").strip()
        if not code or code in seen:
            continue
        seen.add(code)
        name = (unwrap_name(row.get("product_name"))
                or unwrap_name(row.get("generic_name"))
                or f"code {code}")
        cats = [str(c) for c in (row.get("categories_tags") or []) if c]
        brand_raw = str(row.get("brands") or "").strip()
        brand = brand_raw.split(",")[0].strip() if brand_raw else "unknown"
        pid = f"off:{code}"
        try:
            created = time.strftime("%Y-%m-%d",
                                    time.gmtime(int(row.get("created_t") or 0)))
        except (TypeError, ValueError):
            created = ""
        conn.execute(
            "INSERT OR IGNORE INTO products(id,name,category,brand,price,"
            "merchant_id,stock_quantity,performance_score,created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            # OFF has no price field at all: NULL, not 0.0. A 0.0 would
            # masquerade as "free" downstream -- an absent price must stay
            # absent (F10). `price_coverage` in ingest_meta documents this.
            (pid, name, cats[-1] if cats else "unknown", brand or "unknown", None,
             "m_openfoodfacts", 1,
             float(row.get("completeness") or 0.0) * 100.0, created))
        named += 1 if name != f"code {code}" else 0
        for field in OFF_SCALARS:
            eav.add(pid, field, row.get(field))
        for field, key in OFF_LISTS:
            for value in (row.get(field) or []):
                eav.add(pid, key, value)
        for key, value in nutriments_rows(row):
            eav.add(pid, key, value)
        lang = str(row.get("lang") or "")
        if lang:
            eav.add(pid, "language", lang)
    eav.flush()
    conn.commit()
    record(conn, "source", "openfoodfacts/product-database default/food")
    record(conn, "license", "ODbL 1.0")
    record(conn, "price_coverage",
           "none -- the product database carries no price; price-dependent "
           "measurements must not run on this database")
    record(conn, "stock_semantics",
           "1 means 'present in the export'; OFF has no stock field")
    record(conn, "performance_semantics",
           "completeness x 100 (data completeness, NOT product quality)")
    neutral_commercial(conn, "off")
    conn.commit()
    record(conn, "eav_dropped_empty", eav.dropped_empty)
    record(conn, "eav_dropped_long", eav.dropped_long)
    conn.commit()
    stats = _stats(conn)
    stats["named_products"] = named
    stats["distinct_codes_seen"] = len(seen)
    stats["eav_dropped_empty"] = eav.dropped_empty
    stats["eav_dropped_long"] = eav.dropped_long
    conn.close()
    publish_build_target(build_path, out_path)
    return stats


# --------------------------------------------------------------------------
# Source: Open Prices (real offers, real shops)
# --------------------------------------------------------------------------

PRICE_FIELDS = ("price_per", "price_is_discounted", "price_without_discount",
                "discount_type", "category_tag", "source", "owner",
                "receipt_quantity")
PRICE_LISTS = (("labels_tags", "labels"), ("origins_tags", "origins"))
PRICE_PLACES = ("location_osm_address_city", "location_osm_address_country_code",
                "location_type", "proof_type")


def _store_key(row: dict) -> Tuple[str, str]:
    """Identity and display name of the real shop this observation came from."""
    website = str(row.get("location_website_url") or "").strip()
    display = str(row.get("location_osm_display_name") or "").strip()
    loc_id = str(row.get("location_id") or "").strip()
    if website:
        host = re.sub(r"^https?://", "", website).split("/")[0].lower()
        return f"shop:{host}", host
    if display:
        name = display.split(",")[0].strip()
        return f"shop:{loc_id or name.lower()}", name
    return f"shop:{loc_id or 'unknown'}", loc_id or "unknown"


def ingest_prices(limit: int, out_path: str, overwrite: bool = False) -> Dict[str, object]:
    """One product row per (code, shop) real offer; newest observation wins."""
    # Fail fast before the download, not after it.
    build_path = prepare_build_target(out_path, overwrite)
    offers: Dict[Tuple[str, str], dict] = {}
    currencies: Dict[str, int] = {}
    skipped = {"no_price": 0, "per_weight": 0}
    for row in iter_rows("openfoodfacts/open-prices", "default", "prices",
                         limit, label="prices"):
        code = str(row.get("product_code") or "").strip()
        try:
            price = float(row.get("price") or 0.0)
        except (TypeError, ValueError):
            price = 0.0
        if not code or price <= 0.0:
            skipped["no_price"] += 1
            continue
        per = str(row.get("price_per") or "").strip().lower()
        if per and per != "unit":
            # EUR/kg next to EUR/unit inside one price index would make
            # "cheapest" meaningless, so mixed-unit rows are excluded and the
            # exclusion is reported rather than hidden.
            skipped["per_weight"] += 1
            continue
        currency = str(row.get("currency") or "").strip().upper() or "NONE"
        currencies[currency] = currencies.get(currency, 0) + 1
        shop_id, shop_name = _store_key(row)
        date = str(row.get("date") or "")
        key = (code, shop_id)
        prev = offers.get(key)
        if prev is None or date >= str(prev["row"].get("date") or ""):
            offers[key] = {"shop_name": shop_name, "row": row, "price": price,
                           "currency": currency}

    # One currency per database: a price index is only comparable inside a
    # currency. The choice is the modal currency of the fetched slice and goes
    # into ingest_meta, so a reader can see what was dropped and why.
    keep_currency = (max(currencies, key=lambda k: currencies[k])
                     if currencies else "NONE")
    conn = open_db(build_path)
    eav = EavWriter(conn)
    merchants: Dict[str, str] = {}
    for (code, shop_id), off in offers.items():
        if off["currency"] == keep_currency:
            merchants.setdefault(shop_id, off["shop_name"])
    conn.executemany("INSERT OR IGNORE INTO merchants VALUES (?,?,?,?,?)",
                     [(mid, name, 0.0, 0, 0) for mid, name in sorted(merchants.items())])
    kept = 0
    for (code, shop_id), off in offers.items():
        if off["currency"] != keep_currency:
            continue
        row, pid = off["row"], f"prc:{code}:{shop_id}"
        name = str(row.get("product_name") or "").strip() or f"code {code}"
        conn.execute(
            "INSERT OR IGNORE INTO products(id,name,category,brand,price,"
            "merchant_id,stock_quantity,performance_score,created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (pid, name, str(row.get("category_tag") or "").strip() or "unknown",
             "unknown", off["price"], shop_id, 1, 0.0, str(row.get("date") or "")))
        kept += 1
        eav.add(pid, "gtin", code)
        eav.add(pid, "currency", off["currency"])
        for field in PRICE_FIELDS:
            eav.add(pid, field, row.get(field))
        for field, key in PRICE_LISTS:
            for value in (row.get(field) or []):
                eav.add(pid, key, value)
        for field in PRICE_PLACES:
            eav.add(pid, field, row.get(field))
    eav.flush()
    conn.commit()
    record(conn, "source", "openfoodfacts/open-prices default/prices")
    record(conn, "license", "ODbL 1.0")
    record(conn, "currency_kept", keep_currency)
    record(conn, "currencies_seen", json.dumps(currencies, sort_keys=True))
    record(conn, "rows_skipped", json.dumps(skipped, sort_keys=True))
    record(conn, "brand_semantics", "'unknown' -- Open Prices carries no brand field")
    record(conn, "performance_semantics", "0.0 -- no quality signal in this source")
    neutral_commercial(conn, "prices")
    record(conn, "eav_dropped_empty", eav.dropped_empty)
    record(conn, "eav_dropped_long", eav.dropped_long)
    conn.commit()
    stats = _stats(conn)
    stats["offers_seen"] = len(offers)
    stats["offers_kept"] = kept
    stats["eav_dropped_empty"] = eav.dropped_empty
    stats["eav_dropped_long"] = eav.dropped_long
    conn.close()
    publish_build_target(build_path, out_path)
    return stats


def _stats(conn: sqlite3.Connection) -> Dict[str, object]:
    def q(sql: str) -> object:
        return conn.execute(sql).fetchone()[0]

    return {
        "products": q("SELECT COUNT(*) FROM products"),
        "distinct_names": q("SELECT COUNT(DISTINCT name) FROM products"),
        "merchants": q("SELECT COUNT(*) FROM merchants"),
        "categories": q("SELECT COUNT(DISTINCT category) FROM products"),
        "eav_rows": q("SELECT COUNT(*) FROM product_eav_meta"),
        "eav_keys": q("SELECT COUNT(DISTINCT meta_key) FROM product_eav_meta"),
        "priced_products": q("SELECT COUNT(*) FROM products WHERE price > 0"),
        "top_keys": conn.execute(
            "SELECT meta_key, COUNT(*) FROM product_eav_meta "
            "GROUP BY meta_key ORDER BY 2 DESC LIMIT 12").fetchall(),
    }


# --------------------------------------------------------------------------
# Source: WDC Products 2017 (cross-shop offers, real price strings)
# --------------------------------------------------------------------------

_LANG_MARK = re.compile(r"@[a-z]{2}(-[A-Z]{2})?")
_NULLISH = {"", "none", "null", "n/a", "na", "-", "--"}
# `price` in this release is a JSON-ish fragment, e.g.  '"CAD", "$757.74 CAD"'.
_CURRENCY_RE = re.compile(r'["\s]([A-Z]{3})["\s,]')
_AMOUNT_RE = re.compile(r"(\d[\d .\u00a0]*(?:[.,]\d+)?)")


def clean_text(value: object) -> str:
    """Drop the corpus's own decoration: quotes, `@en` marks, literal Nulls."""
    text = str(value or "").strip()
    text = _LANG_MARK.sub(" ", text)
    text = text.replace('"', " ").replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip(" -|,;")
    return "" if text.strip().lower() in _NULLISH else text


def parse_price(value: object) -> Tuple[float, str]:
    """(amount, currency) out of a real, inconsistently formatted price string."""
    text = str(value or "").strip()
    if not text or text.lower() in _NULLISH:
        return 0.0, ""
    cur_m = _CURRENCY_RE.search(text)
    currency = cur_m.group(1) if cur_m else ""
    body = text[cur_m.end():] if currency else text
    amounts = _AMOUNT_RE.findall(body) or _AMOUNT_RE.findall(text)
    best_val = 0.0
    for raw in amounts:
        num = raw.strip().rstrip(",.- ")
        if not num:
            continue
        # European "1.234,56" vs Anglo "1,234.56": whichever separator comes
        # last is the decimal separator, the other one a thousands separator.
        if "," in num and "." in num:
            num = num.replace("." if num.rfind(",") > num.rfind(".") else ",", "")
            num = num.replace(",", ".")
        elif "," in num:
            num = num.replace(",", "." if num.count(",") == 1 else "")
        num = num.replace(" ", "")
        try:
            val = float(num)
        except ValueError:
            continue
        if val > best_val:
            best_val = val
    return best_val, currency


SPEC_KEY_RE = re.compile(
    r"([A-Za-z\u00c0-\u024f][A-Za-z\u00c0-\u024f /()\-]{1,38}?):\s*([^:;\n]{1,120})")


def ingest_wdc(limit: int, out_path: str,
               config: str = "computers_xlarge",
               overwrite: bool = False) -> Dict[str, object]:
    """Dedupe the pairwise release into individual offers, one row per offer."""
    build_path = prepare_build_target(out_path, overwrite)
    offers: Dict[str, dict] = {}
    pairs = 0
    for row in iter_rows("wdc/products-2017", config, "train", limit, label="wdc"):
        pairs += 1
        for side in ("left", "right"):
            oid = row.get(f"id_{side}")
            if oid is None:
                continue
            key = f"{side}:{oid}"
            if key in offers:
                continue
            offers[key] = {
                "offer_id": str(oid),
                "cluster": str(row.get(f"cluster_id_{side}") or ""),
                "category": clean_text(row.get(f"category_{side}")),
                "brand_raw": str(row.get(f"brand_{side}") or ""),
                "title_raw": str(row.get(f"title_{side}") or ""),
                "price_raw": str(row.get(f"price_{side}") or ""),
                "spec_raw": str(row.get(f"specTableContent_{side}") or ""),
            }

    parsed = {oid: parse_price(o["price_raw"]) for oid, o in offers.items()}
    currencies: Dict[str, int] = {}
    for _, cur in parsed.values():
        if cur:
            currencies[cur] = currencies.get(cur, 0) + 1
    keep_currency = (max(currencies, key=lambda k: currencies[k])
                     if currencies else "")

    conn = open_db(build_path)
    eav = EavWriter(conn)
    conn.execute("INSERT INTO merchants VALUES (?,?,?,?,?)",
                 ("wdc_cross_shop",
                  "WDC Products 2017 (aggregated cross-shop crawl, CC-BY 4.0)",
                  0.0, 0, 0))
    unparsed, kept, with_spec, no_price = 0, 0, 0, 0
    for oid, off in offers.items():
        amount, currency = parsed[oid]
        if currency and currency != keep_currency:
            continue
        name = clean_text(off["title_raw"])
        if not name:
            continue
        pid = f"wdc:{oid}"
        conn.execute(
            "INSERT OR IGNORE INTO products(id,name,category,brand,price,"
            "merchant_id,stock_quantity,performance_score,created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (pid, name, off["category"] or "unknown",
             clean_text(off["brand_raw"]) or "unknown",
             amount if amount > 0.0 else None, "wdc_cross_shop",
             1, 0.0, ""))
        kept += 1
        if amount <= 0.0:
            # Distinguish "the source states no price" from "a price is stated
            # and this parser could not read it": only the second one is a
            # defect of this file.
            if clean_text(off["price_raw"]):
                unparsed += 1
            else:
                no_price += 1
        eav.add(pid, "cluster_id", off["cluster"] or None)
        eav.add(pid, "offer_id", off["offer_id"])
        eav.add(pid, "price_raw", clean_text(off["price_raw"]) or None)
        eav.add(pid, "price_currency", currency or None)
        eav.add(pid, "title_raw", clean_text(off["title_raw"]) or None)
        eav.add(pid, "brand_raw", clean_text(off["brand_raw"]) or None)
        spec = clean_text(off["spec_raw"])
        if spec:
            with_spec += 1
            eav.add(pid, "spec_text", spec[:240])
            for skey, sval in SPEC_KEY_RE.findall(spec):
                eav.add(pid, f"spec:{clean_text(skey).lower()}", clean_text(sval))
    eav.flush()
    conn.commit()
    record(conn, "source", f"wdc/products-2017 {config}/train")
    record(conn, "license", "CC-BY 4.0 (Web Data Commons)")
    record(conn, "pairs_read", pairs)
    record(conn, "currency_kept", keep_currency or "-")
    record(conn, "currencies_seen", json.dumps(currencies, sort_keys=True))
    record(conn, "price_parse_failures", unparsed)
    record(conn, "offers_without_a_stated_price", no_price)
    record(conn, "offers_with_spec_text", with_spec)
    record(conn, "merchant_semantics",
           "one aggregated merchant: this redistribution of WDC keeps the "
           "cluster grouping (same product across shops) but not shop identity")
    record(conn, "performance_semantics", "0.0 -- no quality signal in this source")
    neutral_commercial(conn, "wdc")
    record(conn, "eav_dropped_empty", eav.dropped_empty)
    record(conn, "eav_dropped_long", eav.dropped_long)
    conn.commit()
    stats = _stats(conn)
    stats["offers_seen"] = len(offers)
    stats["offers_kept"] = kept
    stats["price_parse_failures"] = unparsed
    stats["offers_without_a_stated_price"] = no_price
    stats["eav_dropped_empty"] = eav.dropped_empty
    stats["eav_dropped_long"] = eav.dropped_long
    stats["clusters"] = conn.execute(
        "SELECT COUNT(DISTINCT meta_value) FROM product_eav_meta "
        "WHERE meta_key='cluster_id'").fetchone()[0]
    conn.close()
    publish_build_target(build_path, out_path)
    return stats


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

DEFAULT_LIMITS = {"off": 3000, "prices": 20000, "wdc": 4000}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=("off", "prices", "wdc", "all"), default="all")
    ap.add_argument("--limit", type=int, default=0,
                    help="source rows to read (0 = the default for that source)")
    ap.add_argument("--wdc-config", default="computers_xlarge")
    ap.add_argument("--out-dir", default=HERE)
    ap.add_argument("--rebuild", action="store_true",
                    help="replace existing target databases (kept as .bak)")
    args = ap.parse_args(argv)

    wanted = ("off", "prices", "wdc") if args.source == "all" else (args.source,)
    # --out-dir implies you may point it anywhere; creating the directory is
    # this script's job, not the caller's. Without it a fresh --out-dir died
    # with "unable to open database file" after the whole download had run.
    os.makedirs(args.out_dir, exist_ok=True)
    summary: Dict[str, object] = {}
    for source in wanted:
        limit = args.limit or DEFAULT_LIMITS[source]
        out_path = os.path.join(args.out_dir, f"real_{source}.db")
        print(f"[ingest] {source}: up to {limit} source rows -> {out_path}", flush=True)
        if source == "off":
            stats = ingest_off(limit, out_path, overwrite=args.rebuild)
        elif source == "prices":
            stats = ingest_prices(limit, out_path, overwrite=args.rebuild)
        else:
            stats = ingest_wdc(limit, out_path, args.wdc_config,
                               overwrite=args.rebuild)
        summary[source] = stats
        print(f"[ingest] {source}: {stats['products']} products, "
              f"{stats['merchants']} merchants, {stats['eav_rows']} EAV rows, "
              f"{stats['eav_keys']} distinct keys, "
              f"{stats['priced_products']} priced", flush=True)

    print("\n=== ingest summary ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())








