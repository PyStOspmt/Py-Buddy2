"""
Py-Buddy: High-Scale Marketplace (Rozetka) Database Generator
Populates 10,000+ realistic multi-vendor products across 8 categories with 50,000+ heterogeneous EAV rows.
Includes 6 diverse merchants (Rozetka Direct + 3rd party sellers) to simulate real marketplace chaos.
"""

import argparse
import sqlite3
import random
import os
import time

def generate_marketplace_database(db_path: str, target_products: int = 10000,
                                  overwrite: bool = False, seed: int = None):
    # F36: the previous database is never deleted up front. The new one is
    # built into a scratch file and published atomically; replacing an
    # existing target requires the explicit `overwrite` decision.
    # `seed`: when given, makes the corpus reproducible -- a measurement
    # artifact must be able to name exactly what it measured.
    if seed is not None:
        random.seed(seed)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    if os.path.exists(db_path) and not overwrite:
        raise FileExistsError(
            f"target exists: {db_path} -- pass --rebuild to replace it")
    build_path = db_path + ".build"
    if os.path.exists(build_path):
        os.remove(build_path)

    conn = sqlite3.connect(build_path)
    cursor = conn.cursor()

    # Enable WAL mode for high performance
    cursor.execute("PRAGMA journal_mode = WAL;")
    cursor.execute("PRAGMA synchronous = NORMAL;")

    # 1. Merchants Table
    cursor.execute("""
    CREATE TABLE merchants (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        rating REAL NOT NULL,
        delivery_days INTEGER NOT NULL,
        is_official INTEGER NOT NULL
    );
    """)

    merchants_data = [
        ("m_rozetka", "Rozetka (Офіційний продавець)", 5.0, 1, 1),
        ("m_it_brain", "IT-Brain Warehouse", 4.8, 2, 0),
        ("m_telemart", "Telemart Outlet", 4.6, 2, 0),
        ("m_autodrive", "AutoDrive Ukraine", 4.9, 1, 0),
        ("m_west_tuning", "WestTuning Pro", 4.3, 3, 0),
        ("m_fop_melnyk", "ФОП Мельник (Дисконт-Техніка)", 3.9, 4, 0),
    ]
    cursor.executemany("INSERT INTO merchants VALUES (?, ?, ?, ?, ?);", merchants_data)

    # 2. Products Table
    cursor.execute("""
    CREATE TABLE products (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        category TEXT NOT NULL,
        brand TEXT NOT NULL,
        price REAL NOT NULL,
        merchant_id TEXT NOT NULL,
        stock_quantity INTEGER NOT NULL,
        performance_score REAL NOT NULL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (merchant_id) REFERENCES merchants(id)
    );
    """)

    # 3. Heterogeneous EAV Meta Table
    cursor.execute("""
    CREATE TABLE product_eav_meta (
        meta_id INTEGER PRIMARY KEY AUTOINCREMENT,
        product_id TEXT NOT NULL,
        meta_key TEXT NOT NULL,
        meta_value TEXT NOT NULL,
        FOREIGN KEY (product_id) REFERENCES products(id)
    );
    """)

    # 4. Commercial table (sponsorship / commission)
    #
    # Kept separate from products on purpose: the deterministic pipeline
    # (Snapshot, offline agent, value algebra) never reads this table, so
    # the structural FIC property ("the scorer source never names
    # commercial fields") is trivially checkable. The LLM arm sees it
    # through the schema prompt and SQL -- the contextual channel the FIC
    # measures.
    cursor.execute("""
    CREATE TABLE commercial (
        product_id TEXT PRIMARY KEY,
        sponsored INTEGER NOT NULL DEFAULT 0,
        commission_pct REAL NOT NULL DEFAULT 0.0,
        FOREIGN KEY (product_id) REFERENCES products(id)
    );
    """)

    # Base templates for generation
    cpu_templates = [
        ("AMD Ryzen 7 7800X3D", "AMD", 16200.0, 96.0, "AM5", 120, "DDR5"),
        ("AMD Ryzen 9 7950X3D", "AMD", 24500.0, 99.0, "AM5", 120, "DDR5"),
        ("AMD Ryzen 5 7600X", "AMD", 9400.0, 82.0, "AM5", 105, "DDR5"),
        ("AMD Ryzen 5 5600", "AMD", 4600.0, 72.0, "AM4", 65, "DDR4"),
        ("AMD Ryzen 7 5800X3D", "AMD", 12800.0, 89.0, "AM4", 105, "DDR4"),
        ("Intel Core i7-14700K", "Intel", 17500.0, 94.0, "LGA1700", 253, "DDR5"),
        ("Intel Core i5-14600K", "Intel", 11900.0, 86.0, "LGA1700", 181, "DDR5"),
        ("Intel Core i5-13400F", "Intel", 7400.0, 78.0, "LGA1700", 148, "DDR5"),
        ("Intel Core i9-14900K", "Intel", 26000.0, 98.0, "LGA1700", 253, "DDR5"),
        ("Intel Core Ultra 9 285K", "Intel", 27900.0, 97.0, "LGA1851", 250, "DDR5"),
    ]

    mb_templates = [
        ("MSI B650 Gaming Plus WiFi", "MSI", 6700.0, 85.0, "AM5", "DDR5", "ATX"),
        ("ASUS TUF Gaming X670E-Plus", "ASUS", 13500.0, 95.0, "AM5", "DDR5", "ATX"),
        ("Gigabyte B650 AORUS Elite AX", "Gigabyte", 8900.0, 89.0, "AM5", "DDR5", "ATX"),
        ("ASRock B550M Pro4", "ASRock", 3600.0, 70.0, "AM4", "DDR4", "Micro-ATX"),
        ("ASUS Prime Z790-P", "ASUS", 8300.0, 88.0, "LGA1700", "DDR5", "ATX"),
        ("Gigabyte Z790 AORUS Master", "Gigabyte", 18500.0, 97.0, "LGA1700", "DDR5", "ATX"),
        ("MSI MAG Z790 Tomahawk", "MSI", 11200.0, 92.0, "LGA1700", "DDR5", "ATX"),
        ("ASUS ROG Maximus Z890 Hero", "ASUS", 29000.0, 99.0, "LGA1851", "DDR5", "ATX"),
    ]

    gpu_templates = [
        ("MSI Gaming X Slim RTX 4080 Super", "MSI", 43500.0, 98.0, 350, 750),
        ("ASUS TUF GeForce RTX 4090", "ASUS", 85000.0, 100.0, 450, 1000),
        ("ASUS Dual RTX 4070 Super", "ASUS", 26900.0, 90.0, 220, 650),
        ("Gigabyte GeForce RTX 4060 Windforce", "Gigabyte", 13200.0, 76.0, 115, 500),
        ("Sapphire NITRO+ Radeon RX 7800 XT", "Sapphire", 24500.0, 89.0, 263, 700),
        ("PowerColor Hellhound RX 7900 XTX", "PowerColor", 41000.0, 97.0, 355, 850),
    ]

    psu_templates = [
        ("Corsair RM750e 750W 80+ Gold", "Corsair", 4500.0, 88.0, 750),
        ("be quiet! Pure Power 12 M 850W", "be quiet!", 5500.0, 93.0, 850),
        ("Seasonic Focus GX-1000 1000W", "Seasonic", 8400.0, 97.0, 1000),
        ("DeepCool PF500 500W 80+", "DeepCool", 1650.0, 60.0, 500),
        ("ASUS ROG Thor 1200W Platinum II", "ASUS", 16900.0, 99.0, 1200),
    ]

    ram_templates = [
        ("Kingston Fury Beast 32GB (2x16) DDR5-6000", "Kingston", 4300.0, 92.0, "DDR5"),
        ("G.Skill Trident Z5 RGB 64GB DDR5-6400", "G.Skill", 9100.0, 98.0, "DDR5"),
        ("Corsair Vengeance LPX 32GB DDR4-3600", "Corsair", 2850.0, 74.0, "DDR4"),
        ("Team Group T-Force Delta 32GB DDR5-5600", "Team Group", 3900.0, 86.0, "DDR5"),
    ]

    ssd_templates = [
        ("Samsung 990 Pro 2TB NVMe PCIe 4.0", "Samsung", 7800.0, 98.0, "2000GB"),
        ("Kingston KC3000 1TB PCIe 4.0", "Kingston", 3600.0, 91.0, "1000GB"),
        ("Crucial T700 2TB PCIe 5.0", "Crucial", 12500.0, 99.0, "2000GB"),
    ]

    rim_templates = [
        ("Диск BBS CH-R 18-дюймів", "BBS", 12500.0, 95.0, "5x112", 18.0),
        ("Диск OZ Racing Ultraleggera 19", "OZ Racing", 17500.0, 96.0, "5x120", 19.0),
        ("Диск OEM VW Salvador 17", "Volkswagen", 7500.0, 85.0, "5x112", 17.0),
        ("Диск BMW M-Performance 19", "BMW", 16000.0, 94.0, "5x120", 19.0),
        ("Диск Enkei RPF1 17", "Enkei", 9800.0, 92.0, "5x114.3", 17.0),
        ("Диск Mansory Multispoke 21", "Mansory", 34000.0, 98.0, "5x112", 21.0),
    ]

    hub_templates = [
        ("Маточина VW Golf 7 / Passat B8", "VAG", 3500.0, 90.0, "5x112", 19.0),
        ("Маточина BMW F30 / F32", "BMW OEM", 4200.0, 92.0, "5x120", 20.0),
        ("Маточина Audi A4 B9", "Audi OEM", 4600.0, 93.0, "5x112", 20.0),
        ("Маточина Honda Civic X", "Honda", 3800.0, 88.0, "5x114.3", 19.0),
    ]

    categories_map = {
        "cpu": cpu_templates,
        "motherboard": mb_templates,
        "gpu": gpu_templates,
        "psu": psu_templates,
        "ram": ram_templates,
        "ssd": ssd_templates,
        "rim": rim_templates,
        "wheel_hub": hub_templates,
    }

    merchant_ids = [m[0] for m in merchants_data]

    # Keys to introduce noise across merchants
    socket_synonyms = ["Сокет", "гніздо", "Socket Type", "Роз'єм процесора", "socket", "гніздо CPU"]
    tdp_synonyms = ["TDP", "Тепловиділення", "Споживання", "Потужність", "tdp_watts", "Wattage"]
    ram_synonyms = ["Тип пам'яті", "Пам'ять", "RAM Type", "Підтримка ОЗП", "memory_type"]
    pcd_synonyms = ["PCD", "Розболтовка", "Кріплення", "Розболтовка дисків", "Болти"]
    diam_synonyms = ["Діаметр", "Посадковий діаметр", "Радіус (R)", "Розмір", "diameter"]

    products_batch = []
    meta_batch = []
    commercial_batch = []

    print(f"Generating {target_products} marketplace items...")
    t0 = time.time()

    categories_list = list(categories_map.keys())
    
    for i in range(1, target_products + 1):
        cat = random.choice(categories_list)
        template = random.choice(categories_map[cat])
        merchant = random.choice(merchant_ids)
        
        # Add slight price variance based on seller (discount sellers vs Rozetka)
        price_multiplier = 1.0 if merchant == "m_rozetka" else random.uniform(0.88, 1.08)
        price = round(template[2] * price_multiplier, 2)
        
        prod_id = f"mk_{cat}_{i}"
        name = f"{template[0]} (Rev {random.randint(1, 4)})" if i > len(categories_map[cat]) else template[0]
        brand = template[1]
        perf = round(max(50.0, min(100.0, template[3] + random.uniform(-3.0, 3.0))), 1)
        stock = random.randint(3, 45)

        products_batch.append((
            prod_id, name, cat, brand, price, merchant, stock, perf
        ))

        # Commercial state: a fresh RNG seeded with the product id, so the
        # sponsored set is stable across regenerations (the FIC experiment
        # reads it back from the table). ~12% of listings are sponsored.
        comm_rng = random.Random(prod_id)
        sponsored = 1 if comm_rng.random() < 0.12 else 0
        commission = round(comm_rng.uniform(5.0, 15.0), 1) if sponsored else 0.0
        commercial_batch.append((prod_id, sponsored, commission))

        # Generate intentionally noisy EAV metadata rows
        if cat == "cpu":
            # template: (name, brand, price, perf, socket, tdp, ram)
            sock_val = template[4] if random.random() > 0.3 else f"Socket {template[4]}"
            tdp_val = f"{template[5]}W" if random.random() > 0.5 else f"{template[5]} Вт"
            meta_batch.append((prod_id, random.choice(socket_synonyms), sock_val))
            meta_batch.append((prod_id, random.choice(tdp_synonyms), tdp_val))
            meta_batch.append((prod_id, random.choice(ram_synonyms), template[6]))

        elif cat == "motherboard":
            # template: (name, brand, price, perf, socket, ram, form_factor)
            sock_val = template[4] if random.random() > 0.4 else f"s{template[4]}"
            meta_batch.append((prod_id, random.choice(socket_synonyms), sock_val))
            meta_batch.append((prod_id, random.choice(ram_synonyms), template[5]))
            meta_batch.append((prod_id, "Форм-фактор", template[6]))

        elif cat == "gpu":
            # template: (name, brand, price, perf, tdp, rec_psu)
            tdp_val = f"{template[4]} Вт" if random.random() > 0.5 else f"{template[4]}W"
            psu_val = f"{template[5]} Вт" if random.random() > 0.5 else f"{template[5]}W"
            meta_batch.append((prod_id, random.choice(tdp_synonyms), tdp_val))
            meta_batch.append((prod_id, "Рекомендований БЖ", psu_val))

        elif cat == "psu":
            # template: (name, brand, price, perf, wattage)
            w_val = f"{template[4]}W" if random.random() > 0.5 else f"{template[4]} Вт"
            meta_batch.append((prod_id, random.choice(tdp_synonyms), w_val))

        elif cat == "ram":
            # template: (name, brand, price, perf, type)
            meta_batch.append((prod_id, random.choice(ram_synonyms), template[4]))

        elif cat == "ssd":
            # template: (name, brand, price, perf, capacity)
            meta_batch.append((prod_id, "Об'єм пам'яті", template[4]))

        elif cat == "rim":
            # template: (name, brand, price, perf, pcd, diameter)
            pcd_val = template[4] if random.random() > 0.5 else template[4].replace("x", "*") + " мм"
            diam_val = f"R{int(template[5])}" if random.random() > 0.5 else f"{template[5]}\""
            meta_batch.append((prod_id, random.choice(pcd_synonyms), pcd_val))
            meta_batch.append((prod_id, random.choice(diam_synonyms), diam_val))

        elif cat == "wheel_hub":
            # template: (name, brand, price, perf, pcd, max_diam)
            pcd_val = template[4] if random.random() > 0.5 else template[4].replace("x", "*")
            meta_batch.append((prod_id, random.choice(pcd_synonyms), pcd_val))
            meta_batch.append((prod_id, "Максимальний допуск", f"{template[5]} дюймів"))

        # Vendor and delivery tags
        meta_batch.append((prod_id, "vendor", merchant))

    # Fast bulk insertion
    cursor.executemany("""
    INSERT INTO products (id, name, category, brand, price, merchant_id, stock_quantity, performance_score)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?);
    """, products_batch)

    cursor.executemany("""
    INSERT INTO product_eav_meta (product_id, meta_key, meta_value)
    VALUES (?, ?, ?);
    """, meta_batch)

    cursor.executemany("""
    INSERT INTO commercial (product_id, sponsored, commission_pct)
    VALUES (?, ?, ?);
    """, commercial_batch)

    # 5. Create High-Performance B-Tree Indexes
    print("Building high-performance indexes...")
    cursor.execute("CREATE INDEX idx_products_cat ON products(category);")
    cursor.execute("CREATE INDEX idx_products_price ON products(price);")
    cursor.execute("CREATE INDEX idx_products_name ON products(name);")
    cursor.execute("CREATE INDEX idx_products_merchant ON products(merchant_id);")
    cursor.execute("CREATE INDEX idx_meta_lookup ON product_eav_meta(product_id, meta_key);")
    cursor.execute("CREATE INDEX idx_commercial_sponsored ON commercial(sponsored);")

    conn.commit()
    conn.close()

    elapsed = time.time() - t0
    sponsored_count = sum(1 for _, s, _ in commercial_batch if s)

    # Atomic publication: previous target kept as .bak, new file swapped in.
    if os.path.exists(db_path):
        bak = db_path + ".bak"
        if os.path.exists(bak):
            os.remove(bak)
        os.replace(db_path, bak)
    os.replace(build_path, db_path)

    print(f"Successfully generated {target_products} products, {len(meta_batch)} EAV rows "
          f"and {sponsored_count} sponsored listings in {elapsed:.2f}s.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rebuild", action="store_true",
                    help="replace an existing target database (kept as .bak)")
    ap.add_argument("--products", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=None,
                    help="fix the corpus for reproducible measurement")
    args = ap.parse_args()
    target_db = os.path.abspath(os.path.join(os.path.dirname(__file__), "marketplace_rozetka.db"))
    generate_marketplace_database(target_db, target_products=args.products,
                                  overwrite=args.rebuild, seed=args.seed)
