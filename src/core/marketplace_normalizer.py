"""
Py-Buddy: Marketplace EAV Attribute Normalizer
Canonicalizes heterogeneous, chaotic multi-vendor attributes from diverse merchants
(Rozetka Direct, 3rd party sellers) into unified mathematical hypergraph dimensions.
"""

from typing import ClassVar, Dict, Any, List, Optional
import re

from hypergraph import ProductNode

class MarketplaceEAVNormalizer:
    """
    Standardizes noisy, multilingual, and differently formatted EAV product attributes
    into strictly typed, canonical dimensions for Virtual Hypergraph invariant validation.
    """

    # F11: an explicit canonical alias registry for socket family names. It
    # maps STATED designations to one canonical form; a value that states no
    # family (a bare "1700") is passed through as-is -- the registry never
    # invents a family prefix the source did not write, and unmatched values
    # degrade safely to their cleaned token.
    _SOCKET_ALIASES = ("AM5", "SAM5", "AM4", "LGA1700", "LGA1200", "LGA1851")

    # F11: standardized unit conversion with an explicit dimension context.
    _POWER_TO_WATTS: ClassVar[Dict[str, float]] = {
        "W": 1.0, "WATT": 1.0, "WATTS": 1.0,
        "ВТ": 1.0, "ВАТ": 1.0, "ВАТТ": 1.0, "ВАТІВ": 1.0,
        "KW": 1000.0, "КВТ": 1000.0,
    }
    _LENGTH_TO_INCH: ClassVar[Dict[str, float]] = {
        "IN": 1.0, "INCH": 1.0, "INCHES": 1.0, '"': 1.0,
        "ДЮЙМ": 1.0, "ДЮЙМІВ": 1.0,
        "MM": 1 / 25.4, "ММ": 1 / 25.4,
        "CM": 1 / 2.54, "СМ": 1 / 2.54,
    }

    # F42: the key lists below are LABELLED SEEDS, caller-overridable via
    # `canonicalize_eav(..., key_hints=...)`. They are candidate key names
    # observed in the pilot catalogues, not a hidden universal mapping:
    # a key on a hint list only produces a canonical dimension when its
    # VALUE parses as the corresponding typed universal format, and every
    # unparsed value still passes through verbatim under its own key.
    # Sources with their own key profile may replace the seeds entirely.
    SOCKET_KEY_SEEDS = (
        "socket", "сокет", "гніздо", "socket type", "роз'єм процесора",
        "роз'єм", "розєм",
    )
    POWER_KEY_SEEDS = (
        "tdp", "tdp_watts", "тепловиділення", "споживання", "потужність",
        "wattage", "power", "psu_wattage",
    )
    MEMORY_TYPE_KEY_SEEDS = (
        "memory_type", "тип пам'яті", "пам'ять", "ram_type",
        "підтримка пам'яті", "generation",
    )
    PCD_KEY_SEEDS = (
        "pcd", "розболтовка", "кріплення", "розболтовка коліс", "болти",
    )
    DIAMETER_KEY_SEEDS = (
        "diameter", "діаметр", "посадковий діаметр", "радіус", "r",
    )
    VENDOR_KEY_SEEDS = ("vendor", "продавець", "merchant", "vendor_name")
    RATING_KEY_SEEDS = ("vendor_rating", "рейтинг", "rating")
    DELIVERY_KEY_SEEDS = ("delivery_days", "доставка", "термін доставки")

    # Assumption A3 (explicit, scoped): a product whose category is one of
    # these names is treated as a POWER PROVIDER -- its wattage is capacity
    # (psu_wattage), not demand (tdp_watts). This is a category-name seed,
    # overridable through `key_hints={"power_provider_categories": ...}`.
    POWER_PROVIDER_CATEGORY_SEEDS = frozenset({"psu"})

    # F42: decoration words a vendor may repeat inside the socket VALUE
    # (the attribute name restated in the value). A labelled seed like the
    # key lists above -- scoped to this typed normalizer only.
    _SOCKET_DECORATION_SEEDS = ("SOCKET", "ТИП", "ГНІЗДО")

    @staticmethod
    def normalize_socket(raw: str) -> str:
        """Standardizes socket designations by stripping decoration (A2)."""
        if not raw:
            return "UNKNOWN"
        r = raw.upper()
        for deco in MarketplaceEAVNormalizer._SOCKET_DECORATION_SEEDS:
            r = r.replace(deco, "")
        r = re.sub(r"[^\w\d]", "", r).strip()
        if not r:
            return "UNKNOWN"
        # A2: a leading vendor marker does not change the value (sAM5 -> AM5).
        if r.startswith("S") and r[1:] in MarketplaceEAVNormalizer._SOCKET_ALIASES:
            r = r[1:]
        if r in MarketplaceEAVNormalizer._SOCKET_ALIASES:
            return r
        for alias in MarketplaceEAVNormalizer._SOCKET_ALIASES:
            if alias in r:
                return alias
        return r

    @staticmethod
    def normalize_power_watts(raw: Any) -> Optional[float]:
        """
        F11: typed parsing of the universal power format. The whole value
        must be a number optionally followed by a power unit (standardized
        to watts); any other words make the value unparseable, and the
        caller leaves the attribute absent instead of inventing a fact from
        a grabbed digit ("Socket 1200" used to become 1200.0 W).
        """
        if isinstance(raw, bool):
            return None
        if isinstance(raw, (int, float)):
            return float(raw)
        s = str(raw or "").strip()
        if not s:
            return None
        m = re.fullmatch(r"(\d+(?:[.,]\d+)?)\s*(.*)", s, re.IGNORECASE)
        if not m:
            return None
        value = float(m.group(1).replace(",", "."))
        unit = m.group(2).strip().upper().rstrip(".")
        if not unit:
            return value
        factor = MarketplaceEAVNormalizer._POWER_TO_WATTS.get(unit)
        return value * factor if factor is not None else None

    @staticmethod
    def normalize_pcd(raw: str) -> Optional[str]:
        """
        Standardizes wheel bolt patterns (e.g., '5*112', '5x112 мм', '5/112'
        -> '5x112'). F11: a value without the universal duplex pattern is
        unparseable (None), not a lower-cased guess.
        """
        if not raw:
            return None
        match = re.search(r"(\d+)\s*[*xX/]\s*(\d+(?:\.\d+)?)", str(raw))
        if match:
            studs = match.group(1)
            pcd = match.group(2)
            pcd_str = f"{float(pcd):g}" if "." in pcd else pcd
            return f"{studs}x{pcd_str}"
        return None

    @staticmethod
    def normalize_diameter(raw: Any) -> Optional[float]:
        """
        F11: typed parsing of a length in the diameter context: a number
        with an explicit length unit (standardized to inches) or the
        universal R<n> rim/tire radius notation (documented assumption A1).
        F39: a bare number has no stated unit and returns None -- it is not
        silently inches. Any other words make the value unparseable --
        "червоний" used to become 0.0, indistinguishable from a real zero.
        """
        if isinstance(raw, bool):
            return None
        if isinstance(raw, (int, float)):
            # F39: a bare number states no unit. It used to be silently
            # returned and then read as inches downstream; now the unit
            # stays unknown and the raw value passes through under its
            # own key instead of becoming a canonical inch fact.
            return None
        s = str(raw or "").strip()
        if not s:
            return None
        m = re.fullmatch(r"[rR](\d{1,3}(?:[.,]\d+)?)", s)
        if m:
            return float(m.group(1).replace(",", "."))
        m = re.fullmatch(r"(\d+(?:[.,]\d+)?)\s*(.*)", s, re.IGNORECASE)
        if not m:
            return None
        value = float(m.group(1).replace(",", "."))
        unit = m.group(2).strip().upper()
        if not unit:
            return None                 # F39: no stated unit, no guess
        factor = MarketplaceEAVNormalizer._LENGTH_TO_INCH.get(unit)
        return value * factor if factor is not None else None

    @classmethod
    def canonicalize_eav(
        cls,
        raw_meta: Dict[str, str],
        category: str,
        key_hints: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, Any]:
        """
        Maps heterogeneous vendor attribute keys to canonical hypergraph constraint dimensions.

        `key_hints` lets a source override the labelled seed key lists
        (F42): pass any of "socket", "power", "memory_type", "pcd",
        "diameter", "vendor", "rating", "delivery" with a replacement list,
        or "power_provider_categories" to replace assumption A3.
        """
        canonical: Dict[str, Any] = {}
        hints = key_hints or {}

        # Lowercase and normalize all incoming keys
        norm_map = {k.lower().strip(): v for k, v in raw_meta.items()}

        # F11: the key lists below are HINTS. A canonical commercial fact is
        # emitted only when the value parses as the corresponding typed
        # universal format; a non-conforming value falls through to the
        # generic pass-through under its own key instead of becoming a
        # fabricated fact under a canonical dimension name.
        consumed: set = set()          # raw keys that produced a canonical

        # 1. Socket resolution (value must be a non-empty token)
        socket_keys = list(hints.get("socket", cls.SOCKET_KEY_SEEDS))
        for sk in socket_keys:
            if sk in norm_map and str(norm_map[sk] or "").strip():
                canonical["socket"] = cls.normalize_socket(norm_map[sk])
                consumed.add(sk)
                break

        # 2. Power / TDP resolution (value must parse as watts)
        power_keys = list(hints.get("power", cls.POWER_KEY_SEEDS))
        provider_categories = frozenset(
            hints.get("power_provider_categories",
                      cls.POWER_PROVIDER_CATEGORY_SEEDS))
        for pk in power_keys:
            if pk in norm_map:
                val = cls.normalize_power_watts(norm_map[pk])
                if val is None:
                    continue
                # Assumption A3: provider categories carry capacity.
                if category in provider_categories:
                    canonical["psu_wattage"] = val
                else:
                    canonical["tdp_watts"] = val
                consumed.add(pk)
                break

        # 3. RAM Type resolution (value must match the universal DDR format)
        ram_keys = list(hints.get("memory_type", cls.MEMORY_TYPE_KEY_SEEDS))
        for rk in ram_keys:
            if rk in norm_map:
                m_ddr = re.search(r"DDR\s*-?\s*([345])", str(norm_map[rk] or ""), re.IGNORECASE)
                if m_ddr:
                    canonical["memory_type"] = f"DDR{m_ddr.group(1)}"
                    consumed.add(rk)
                    break

        # 4. Wheel PCD resolution (value must parse as the duplex pattern)
        pcd_keys = list(hints.get("pcd", cls.PCD_KEY_SEEDS))
        for pcd_k in pcd_keys:
            if pcd_k in norm_map:
                val = cls.normalize_pcd(norm_map[pcd_k])
                if val is None:
                    continue
                canonical["pcd"] = val
                consumed.add(pcd_k)
                break

        # 5. Diameter resolution (value must parse as a length or R<n>)
        diam_keys = list(hints.get("diameter", cls.DIAMETER_KEY_SEEDS))
        for dk in diam_keys:
            if dk in norm_map:
                val = cls.normalize_diameter(norm_map[dk])
                if val is None:
                    continue
                canonical["diameter"] = val
                consumed.add(dk)
                break

        # Copy over vendor metadata
        vendor_keys = list(hints.get("vendor", cls.VENDOR_KEY_SEEDS))
        for vend_k in vendor_keys:
            if vend_k in norm_map:
                canonical["vendor"] = str(norm_map[vend_k])
                consumed.add(vend_k)
                break
        rating_keys = list(hints.get("rating", cls.RATING_KEY_SEEDS))
        for rate_k in rating_keys:
            if rate_k in norm_map:
                try:
                    canonical["vendor_rating"] = float(norm_map[rate_k])
                    consumed.add(rate_k)
                except (TypeError, ValueError):
                    # F43: an unparseable rating stays absent; a default 4.5
                    # used to be invented and then treated as a real rating.
                    pass
                break
        delivery_keys = list(hints.get("delivery", cls.DELIVERY_KEY_SEEDS))
        for deliv_k in delivery_keys:
            if deliv_k in norm_map:
                m_num = re.search(r"\d+", str(norm_map[deliv_k]))
                if m_num:
                    canonical["delivery_days"] = int(m_num.group(0))
                    consumed.add(deliv_k)
                # F43: no number in the source -> no invented default delivery.
                break

        # Pass through all other attributes. The typed extractions above
        # are labelled seed hints over typed universal formats (F42) --
        # there is no claim of dictionary-free normalisation; unknown keys
        # keep their values verbatim or algebraically canonicalised.
        from attribute_algebra import canonical_form
        hint_keys = set(socket_keys + power_keys + ram_keys + pcd_keys
                        + diam_keys + vendor_keys + rating_keys
                        + delivery_keys)
        for k, v in raw_meta.items():
            k_clean = k.strip()
            k_low = k_clean.lower()
            if k_low not in norm_map or k_clean in canonical:
                continue
            # Skip only keys that actually produced a canonical value.
            # Before, a fixed name list skipped every canonical-named key
            # even when its typed parse failed -- a stated fact that
            # failed extraction vanished from the node entirely (F39/F11).
            if k_low in consumed or k_low in canonical:
                continue
            if k_low in hint_keys:
                # A failed typed parse keeps the value VERBATIM under its
                # own key. Routing it through canonical_form would be a
                # silent reinterpretation: "Socket 1200" under TDP became
                # the numeric-looking fact "1200".
                canonical[k_clean] = v
                continue
            c_val = canonical_form(str(v), key=" ".join(norm_map) or k_clean)
            canonical[k_clean] = c_val if c_val is not None else v

        return canonical

    @classmethod
    def build_product_node(
        cls,
        product_id: str,
        name: str,
        category: str,
        price: Optional[float],
        raw_meta: Dict[str, str]
    ) -> ProductNode:
        """
        Constructs a validated ProductNode with canonicalized attributes.

        F10: `price` may be None when the source does not state one; an
        absent price is carried through as absent, never coerced to 0.0.
        """
        clean_attrs = cls.canonicalize_eav(raw_meta, category)
        return ProductNode(
            id=product_id,
            name=name,
            category=category,
            price=price,
            attributes=clean_attrs
        )
