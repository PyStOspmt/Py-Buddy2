"""
TDD Suite for MarketplaceEAVNormalizer (src/marketplace_normalizer.py).
Verifies canonical resolution of noisy, multilingual, multi-vendor EAV data.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from hypergraph import ProductNode
from marketplace_normalizer import MarketplaceEAVNormalizer


def test_socket_normalization():
    norm = MarketplaceEAVNormalizer.normalize_socket
    assert norm("Socket AM5") == "AM5"
    assert norm("sAM5") == "AM5"
    assert norm("гніздо: AM5") == "AM5"
    assert norm("AMD Socket AM5") == "AM5"
    assert norm("LGA 1700") == "LGA1700"
    assert norm("Socket LGA1700") == "LGA1700"
    assert norm("роз'єм: LGA-1700") == "LGA1700"
    assert norm("Socket AM4") == "AM4"

def test_power_watts_normalization():
    norm = MarketplaceEAVNormalizer.normalize_power_watts
    assert norm("120W") == 120.0
    assert norm("120 Вт") == 120.0
    assert norm("120 watts") == 120.0
    assert norm("253.5 Вт") == 253.5
    assert norm(750) == 750.0
    assert norm("750") == 750.0

def test_pcd_wheel_normalization():
    norm = MarketplaceEAVNormalizer.normalize_pcd
    assert norm("5x112") == "5x112"
    assert norm("5*112 мм") == "5x112"
    assert norm("5/112") == "5x112"
    assert norm("5x120") == "5x120"
    assert norm("4*100") == "4x100"

def test_diameter_normalization():
    norm = MarketplaceEAVNormalizer.normalize_diameter
    assert norm("R18") == 18.0
    assert norm("18\"") == 18.0
    assert norm("19.5 дюймів") == 19.5
    assert norm(17) is None          # F39: bare number, unit unknown

def test_canonicalize_eav_multi_vendor():
    # Vendor A (Rozetka) raw attributes
    raw_rozetka = {
        "Сокет": "AM5",
        "TDP": "120W",
        "Тип пам'яті": "DDR5",
        "Продавець": "Rozetka",
        "Рейтинг": "5.0",
        "Доставка": "1 день"
    }
    canon_rozetka = MarketplaceEAVNormalizer.canonicalize_eav(raw_rozetka, category="cpu")
    assert canon_rozetka["socket"] == "AM5"
    assert canon_rozetka["tdp_watts"] == 120.0
    assert canon_rozetka["memory_type"] == "DDR5"
    assert canon_rozetka["vendor"] == "Rozetka"
    assert canon_rozetka["vendor_rating"] == 5.0
    assert canon_rozetka["delivery_days"] == 1

    # Vendor B (3rd-party IT-Warehouse) raw attributes with dirty keys
    raw_vendor_b = {
        "гніздо": "AMD sAM5",
        "Тепловиділення": "120 Вт",
        "RAM_type": "DDR-5",
        "vendor_name": "IT-Warehouse",
        "rating": "4.6",
        "термін доставки": "3 дні"
    }
    canon_b = MarketplaceEAVNormalizer.canonicalize_eav(raw_vendor_b, category="cpu")
    assert canon_b["socket"] == "AM5"
    assert canon_b["tdp_watts"] == 120.0
    assert canon_b["memory_type"] == "DDR5"
    assert canon_b["vendor"] == "IT-Warehouse"
    assert canon_b["vendor_rating"] == 4.6
    assert canon_b["delivery_days"] == 3

def test_build_product_node():
    node = MarketplaceEAVNormalizer.build_product_node(
        product_id="cpu_test_1",
        name="AMD Ryzen 7 7800X3D",
        category="cpu",
        price=16200.0,
        raw_meta={"роз'єм процесора": "AM5", "споживання": "120W"}
    )
    assert isinstance(node, ProductNode)
    assert node.attributes["socket"] == "AM5"
    assert node.attributes["tdp_watts"] == 120.0
