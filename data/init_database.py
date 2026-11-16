"""
Py-Buddy: Realistic E-Commerce Catalog & EAV Database Initializer
Generates a realistic SQLite database with both standard catalog tables and an EAV attribute table.
Generates the corresponding Knowledge Hypergraph configuration.
"""

import argparse
import sqlite3
import json
import os
from typing import Optional

def create_ecommerce_database(db_path: str, hypergraph_json_path: Optional[str] = None,
                              overwrite: bool = False):
    # F36: a rebuild must be an explicit decision. Silently deleting an
    # existing catalogue destroys customer evidence; refusing unless
    # `overwrite` was requested keeps the workflow honest.
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    if os.path.exists(db_path):
        if not overwrite:
            raise FileExistsError(
                f"target exists: {db_path} -- pass overwrite=True to rebuild")
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 1. Main products catalog table
    cursor.execute("""
    CREATE TABLE products (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        category TEXT NOT NULL,
        brand TEXT NOT NULL,
        price REAL NOT NULL,
        stock_quantity INTEGER NOT NULL,
        performance_score REAL NOT NULL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # 2. EAV Meta table (simulating WooCommerce wp_postmeta)
    cursor.execute("""
    CREATE TABLE product_meta (
        meta_id INTEGER PRIMARY KEY AUTOINCREMENT,
        product_id TEXT NOT NULL,
        meta_key TEXT NOT NULL,
        meta_value TEXT NOT NULL,
        FOREIGN KEY (product_id) REFERENCES products(id)
    );
    """)

    # 3. Commercial table (sponsorship / commission)
    #
    # Kept separate from products on purpose: the deterministic pipeline
    # (Snapshot, offline agent, value algebra) never reads this table, so
    # the structural FIC property is trivially checkable; the LLM arm sees
    # it through the schema prompt and SQL -- the contextual channel.
    cursor.execute("""
    CREATE TABLE commercial (
        product_id TEXT PRIMARY KEY,
        sponsored INTEGER NOT NULL DEFAULT 0,
        commission_pct REAL NOT NULL DEFAULT 0.0,
        FOREIGN KEY (product_id) REFERENCES products(id)
    );
    """)

    # Catalog Items
    products = [
        # CPUs
        ("cpu_am5_1", "AMD Ryzen 7 7800X3D", "cpu", "AMD", 16500.0, 15, 96.0),
        ("cpu_am5_2", "AMD Ryzen 5 7600X", "cpu", "AMD", 9500.0, 24, 82.0),
        ("cpu_intel_1", "Intel Core i7-14700K", "cpu", "Intel", 17200.0, 10, 94.0),
        ("cpu_intel_2", "Intel Core i5-14600K", "cpu", "Intel", 11800.0, 18, 85.0),

        # Motherboards
        ("mb_am5_msi", "MSI B650 Gaming Plus WiFi", "motherboard", "MSI", 6800.0, 12, 85.0),
        ("mb_am5_asus", "ASUS TUF Gaming X670E-Plus", "motherboard", "ASUS", 13200.0, 8, 95.0),
        ("mb_intel_asus", "ASUS Prime Z790-P", "motherboard", "ASUS", 8400.0, 14, 88.0),
        ("mb_intel_gigabyte", "Gigabyte Z790 AORUS Elite", "motherboard", "Gigabyte", 12500.0, 6, 94.0),

        # RAM
        ("ram_ddr5_32", "Kingston Fury Beast 32GB (2x16) DDR5-6000", "ram", "Kingston", 4400.0, 30, 92.0),
        ("ram_ddr5_64", "G.Skill Trident Z5 RGB 64GB DDR5-6400", "ram", "G.Skill", 9200.0, 9, 98.0),
        ("ram_ddr4_32", "Corsair Vengeance LPX 32GB DDR4-3600", "ram", "Corsair", 2900.0, 25, 74.0),

        # GPUs
        ("gpu_rtx4070", "ASUS Dual GeForce RTX 4070 Super", "gpu", "ASUS", 27500.0, 11, 91.0),
        ("gpu_rtx4080", "MSI Gaming X Slim GeForce RTX 4080 Super", "gpu", "MSI", 44000.0, 5, 98.0),
        ("gpu_rx7800", "Sapphire NITRO+ Radeon RX 7800 XT", "gpu", "Sapphire", 24900.0, 14, 89.0),

        # Power Supplies
        ("psu_750w", "Corsair RM750e 750W 80+ Gold ATX 3.0", "psu", "Corsair", 4600.0, 20, 88.0),
        ("psu_850w", "be quiet! Pure Power 12 M 850W Gold", "psu", "be quiet!", 5600.0, 16, 93.0),
        ("psu_1000w", "Seasonic Focus GX-1000 1000W 80+ Gold", "psu", "Seasonic", 8200.0, 8, 97.0),
        ("psu_500w", "DeepCool PF500 500W 80+", "psu", "DeepCool", 1700.0, 40, 60.0)
    ]

    cursor.executemany("""
    INSERT INTO products (id, name, category, brand, price, stock_quantity, performance_score)
    VALUES (?, ?, ?, ?, ?, ?, ?);
    """, products)

    # EAV Metadata (attributes: socket, memory_type, psu_recommendation_watts, tdp_watts)
    meta_records = [
        # CPUs
        ("cpu_am5_1", "socket", "AM5"),
        ("cpu_am5_1", "tdp_watts", "120"),
        ("cpu_am5_1", "supported_memory", "DDR5"),

        ("cpu_am5_2", "socket", "AM5"),
        ("cpu_am5_2", "tdp_watts", "105"),
        ("cpu_am5_2", "supported_memory", "DDR5"),

        ("cpu_intel_1", "socket", "LGA1700"),
        ("cpu_intel_1", "tdp_watts", "253"),
        ("cpu_intel_1", "supported_memory", "DDR5"),

        ("cpu_intel_2", "socket", "LGA1700"),
        ("cpu_intel_2", "tdp_watts", "181"),
        ("cpu_intel_2", "supported_memory", "DDR5"),

        # Motherboards
        ("mb_am5_msi", "socket", "AM5"),
        ("mb_am5_msi", "memory_slots", "DDR5"),
        ("mb_am5_msi", "form_factor", "ATX"),

        ("mb_am5_asus", "socket", "AM5"),
        ("mb_am5_asus", "memory_slots", "DDR5"),
        ("mb_am5_asus", "form_factor", "ATX"),

        ("mb_intel_asus", "socket", "LGA1700"),
        ("mb_intel_asus", "memory_slots", "DDR5"),
        ("mb_intel_asus", "form_factor", "ATX"),

        ("mb_intel_gigabyte", "socket", "LGA1700"),
        ("mb_intel_gigabyte", "memory_slots", "DDR5"),
        ("mb_intel_gigabyte", "form_factor", "ATX"),

        # RAM
        ("ram_ddr5_32", "generation", "DDR5"),
        ("ram_ddr5_64", "generation", "DDR5"),
        ("ram_ddr4_32", "generation", "DDR4"),

        # GPUs
        ("gpu_rtx4070", "min_psu_watts", "650"),
        ("gpu_rtx4080", "min_psu_watts", "750"),
        ("gpu_rx7800", "min_psu_watts", "700"),

        # PSUs
        ("psu_750w", "wattage", "750"),
        ("psu_850w", "wattage", "850"),
        ("psu_1000w", "wattage", "1000"),
        ("psu_500w", "wattage", "500")
    ]

    cursor.executemany("""
    INSERT INTO product_meta (product_id, meta_key, meta_value)
    VALUES (?, ?, ?);
    """, meta_records)

    # Sponsored listings: a fixed, documented subset of the catalog so the
    # FIC experiment has a non-trivial commercial state to permute.
    commercial_records = [
        ("cpu_am5_1", 1, 12.0),
        ("mb_am5_asus", 1, 8.5),
        ("gpu_rtx4080", 1, 15.0),
        ("psu_1000w", 1, 6.0),
    ]
    cursor.executemany("""
    INSERT INTO commercial (product_id, sponsored, commission_pct)
    VALUES (?, ?, ?);
    """, commercial_records)

    conn.commit()
    conn.close()

    # Build Hypergraph JSON
    hypergraph_data = {
        "nodes": [
            {"id": p[0], "name": p[1], "category": p[2], "price": p[4], "attributes": {"brand": p[3], "performance": p[6]}}
            for p in products
        ],
        "hyperedges": [
            # AMD AM5 Tier 1 bundle
            {
                "id": "edge_am5_mainstream",
                "name": "AM5 Gaming Mainstream Bundle",
                "product_ids": ["cpu_am5_2", "mb_am5_msi", "ram_ddr5_32", "gpu_rtx4070", "psu_750w"],
                "confidence": 1.0,
                "domain_rule": "AM5 Socket + DDR5 + 750W PSU"
            },
            # AMD AM5 Enthusiast bundle
            {
                "id": "edge_am5_enthusiast",
                "name": "AM5 Enthusiast Esports Rig",
                "product_ids": ["cpu_am5_1", "mb_am5_asus", "ram_ddr5_64", "gpu_rtx4080", "psu_850w"],
                "confidence": 1.0,
                "domain_rule": "AM5 X670E + 7800X3D + RTX 4080 + 850W"
            },
            # Intel LGA1700 Mainstream bundle
            {
                "id": "edge_intel_mainstream",
                "name": "Intel 14th Gen Mainstream Rig",
                "product_ids": ["cpu_intel_2", "mb_intel_asus", "ram_ddr5_32", "gpu_rx7800", "psu_750w"],
                "confidence": 1.0,
                "domain_rule": "LGA1700 + DDR5 + RX 7800 XT + 750W"
            },
            # Intel LGA1700 Pro bundle
            {
                "id": "edge_intel_pro",
                "name": "Intel 14th Gen Workstation Pro",
                "product_ids": ["cpu_intel_1", "mb_intel_gigabyte", "ram_ddr5_64", "gpu_rtx4080", "psu_1000w"],
                "confidence": 1.0,
                "domain_rule": "LGA1700 Z790 + i7-14700K + RTX 4080 + 1000W"
            }
        ]
    }

    if hypergraph_json_path:
        with open(hypergraph_json_path, "w", encoding="utf-8") as f:
            json.dump(hypergraph_data, f, indent=2, ensure_ascii=False)
        print(f"Knowledge Hypergraph saved at: {hypergraph_json_path}")

    print(f"Catalog DB successfully generated at: {db_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rebuild", action="store_true",
                    help="replace an existing target database")
    ap.add_argument("--hypergraph-json", default=None,
                    help="optional path for the hypergraph config export")
    args = ap.parse_args()
    db_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "catalog.db"))
    create_ecommerce_database(db_file, hypergraph_json_path=args.hypergraph_json,
                              overwrite=args.rebuild)
