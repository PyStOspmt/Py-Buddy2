"""
Initialize the real SQLite database for Auto Parts (Wheels & Tires).
Replaces hardcoded in-code products (resolves C5).
"""

import os
import sqlite3

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
AUTO_DB_PATH = os.path.join(DATA_DIR, "auto_parts.db")

def init_auto_parts_db(db_path: str = AUTO_DB_PATH):
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE merchants (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        rating REAL DEFAULT 5.0,
        delivery_days INT DEFAULT 2
    );
    """)

    cur.execute("""
    CREATE TABLE products (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        category TEXT NOT NULL,
        brand TEXT,
        price REAL NOT NULL,
        merchant_id TEXT,
        stock_quantity INT DEFAULT 10,
        performance_score REAL DEFAULT 82.0,
        FOREIGN KEY (merchant_id) REFERENCES merchants(id)
    );
    """)

    cur.execute("""
    CREATE TABLE product_eav_meta (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        product_id TEXT NOT NULL,
        meta_key TEXT NOT NULL,
        meta_value TEXT NOT NULL,
        FOREIGN KEY (product_id) REFERENCES products(id)
    );
    """)

    # Seed merchants
    merchants = [
        ("m_auto_hub", "AutoHub Ukraine", 4.9, 1),
        ("m_tire_pro", "TirePro Kyiv", 4.8, 2),
        ("m_tuning_store", "Tuning & Wheels", 4.7, 3),
    ]
    cur.executemany("INSERT INTO merchants VALUES (?, ?, ?, ?)", merchants)

    # Seed products (preserving core 10 IDs + realistic catalog assortment)
    products = [
        ("hub_golf", "Маточина VW Golf 7", "wheel_hub", "VAG", 3500.0, "m_auto_hub", 12, 86.0),
        ("hub_bmw", "Маточина BMW F30", "wheel_hub", "BMW OEM", 4200.0, "m_auto_hub", 8, 89.0),
        ("rim_bbs", "Диск BBS CH-R 18-дюймів", "rim", "BBS", 12500.0, "m_tuning_store", 16, 93.5),
        ("rim_oem_vw", "Диск OEM VW Salvador 17-дюймів", "rim", "Volkswagen", 7500.0, "m_auto_hub", 20, 85.0),
        ("rim_bmw_m", "Диск BMW M-Performance 19-дюймів", "rim", "BMW M", 16000.0, "m_tuning_store", 8, 95.0),
        ("rim_mansory", "Диск Mansory 21-дюймів", "rim", "Mansory", 34000.0, "m_tuning_store", 4, 97.5),
        ("tire_michelin", "Шина Michelin Pilot Sport 4 (18\")", "tire", "Michelin", 5800.0, "m_tire_pro", 24, 94.0),
        ("tire_goodyear", "Шина Goodyear UltraGrip (17\")", "tire", "Goodyear", 4400.0, "m_tire_pro", 18, 88.5),
        ("tire_pirelli_19", "Шина Pirelli P Zero (19\")", "tire", "Pirelli", 6900.0, "m_tire_pro", 12, 95.5),
        ("tire_continental_21", "Шина Continental SportContact (21\")", "tire", "Continental", 8900.0, "m_tire_pro", 8, 96.5),
        ("hub_skoda_oct", "Маточина Skoda Octavia A7 (FAG)", "wheel_hub", "FAG", 2950.0, "m_auto_hub", 15, 84.0),
        ("hub_audi_q7", "Маточина Audi Q7 / Cayenne SUV", "wheel_hub", "SKF", 5400.0, "m_auto_hub", 6, 91.5),
        ("hub_toyota_camry", "Маточина Toyota Camry XV70", "wheel_hub", "Koyo", 3850.0, "m_auto_hub", 11, 88.0),
        ("hub_mazda_cx5", "Маточина Mazda CX-5 KF", "wheel_hub", "NTN-SNR", 3620.0, "m_auto_hub", 9, 86.5),
        ("hub_renault_megane", "Маточина Renault Megane III / Logan", "wheel_hub", "NTN-SNR", 2390.0, "m_auto_hub", 19, 80.5),
        ("hub_bmw_x5", "Маточина BMW X5 G05 Посилена", "wheel_hub", "Lemforder", 6100.0, "m_auto_hub", 5, 93.0),
        ("rim_oz_ultra_18", "Диск OZ Ultraleggera 18-дюймів", "rim", "OZ Racing", 13900.0, "m_tuning_store", 12, 95.2),
        ("rim_borbet_y17", "Диск Borbet Y Titan Matt 17-дюймів", "rim", "Borbet", 5650.0, "m_auto_hub", 24, 83.0),
        ("rim_enkei_rpf1_18", "Диск Enkei RPF1 18-дюймів (5x114.3)", "rim", "Enkei", 11800.0, "m_tuning_store", 10, 94.2),
        ("rim_rays_te37_19", "Диск Rays Volk TE37 19-дюймів Кований", "rim", "Rays", 26500.0, "m_tuning_store", 4, 98.2),
        ("rim_dez_tr_16", "Диск Dezent TR Silver 16-дюймів", "rim", "Dezent", 4350.0, "m_auto_hub", 28, 79.0),
        ("rim_ronal_r56_17", "Диск Ronal R56 Black 17-дюймів", "rim", "Ronal", 6150.0, "m_auto_hub", 16, 84.5),
        ("rim_vossen_hf5_20", "Диск Vossen HF-5 Gloss Black 20-дюймів", "rim", "Vossen", 22900.0, "m_tuning_store", 8, 96.0),
        ("rim_kfz_steel_16", "Диск сталевий KFZ 16-дюймів (4x100)", "rim", "KFZ", 2150.0, "m_auto_hub", 32, 73.5),
        ("tire_bridgestone_18", "Шина Bridgestone Blizzak LM005 (18\")", "tire", "Bridgestone", 5450.0, "m_tire_pro", 20, 92.0),
        ("tire_nokian_17", "Шина Nokian Hakkapeliitta R5 (17\")", "tire", "Nokian", 5100.0, "m_tire_pro", 16, 91.5),
        ("tire_hankook_16", "Шина Hankook Ventus Prime 4 (16\")", "tire", "Hankook", 3150.0, "m_tire_pro", 30, 84.0),
        ("tire_kumho_17", "Шина Kumho Ecsta PS71 (17\")", "tire", "Kumho", 3380.0, "m_tire_pro", 26, 82.5),
        ("tire_toyo_19", "Шина Toyo Proxes Sport 2 (19\")", "tire", "Toyo", 6100.0, "m_tire_pro", 14, 90.5),
        ("tire_dunlop_20", "Шина Dunlop SP Sport Maxx RT2 (20\")", "tire", "Dunlop", 7650.0, "m_tire_pro", 10, 93.0),
        ("tire_barum_16", "Шина Barum Bravuris 5HM (16\")", "tire", "Barum", 2580.0, "m_tire_pro", 36, 77.5),
        ("tire_michelin_cc2_18", "Шина Michelin CrossClimate 2 (18\")", "tire", "Michelin", 6250.0, "m_tire_pro", 18, 94.8),
        ("brake_brembo_312", "Гальмівний диск Brembo Xtra 312 мм (VAG)", "brake_disc", "Brembo", 2890.0, "m_auto_hub", 14, 91.0),
        ("brake_zimmermann_340", "Гальмівний диск Zimmermann Sport Z 340 мм", "brake_disc", "Zimmermann", 4150.0, "m_tuning_store", 10, 93.5),
        ("brake_ate_300", "Гальмівний диск ATE PowerDisc 300 мм", "brake_disc", "ATE", 2420.0, "m_auto_hub", 22, 87.5),
        ("brake_trw_280", "Гальмівний диск TRW Lucas 280 мм", "brake_disc", "TRW", 1680.0, "m_auto_hub", 25, 81.0),
    ]
    cur.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?, ?)", products)

    # Seed EAV attributes
    eav_records = [
        ("hub_golf", "pcd", "5x112"),
        ("hub_golf", "center_bore", "57.1"),
        ("hub_golf", "max_diameter", '19"'),

        ("hub_bmw", "pcd", "5x120"),
        ("hub_bmw", "center_bore", "72.6"),
        ("hub_bmw", "max_diameter", '20"'),

        ("rim_bbs", "pcd", "5x112"),
        ("rim_bbs", "diameter", "R18"),
        ("rim_bbs", "weight_kg", "9.5"),

        ("rim_oem_vw", "pcd", "5x112"),
        ("rim_oem_vw", "diameter", "R17"),
        ("rim_oem_vw", "weight_kg", "10.2"),

        ("rim_bmw_m", "pcd", "5x120"),
        ("rim_bmw_m", "diameter", "R19"),
        ("rim_bmw_m", "weight_kg", "11.0"),

        ("rim_mansory", "pcd", "5x112"),
        ("rim_mansory", "diameter", "R21"),
        ("rim_mansory", "weight_kg", "13.5"),

        ("tire_michelin", "diameter", "R18"),
        ("tire_michelin", "season", "summer"),

        ("tire_goodyear", "diameter", "R17"),
        ("tire_goodyear", "season", "winter"),

        ("tire_pirelli_19", "diameter", "R19"),
        ("tire_pirelli_19", "season", "summer"),

        ("tire_continental_21", "diameter", "R21"),
        ("tire_continental_21", "season", "summer"),

        ("hub_skoda_oct", "pcd", "5x112"),
        ("hub_skoda_oct", "center_bore", "57.1"),
        ("hub_skoda_oct", "max_diameter", '18"'),
        ("hub_audi_q7", "pcd", "5x112"),
        ("hub_audi_q7", "center_bore", "66.6"),
        ("hub_audi_q7", "max_diameter", '22"'),
        ("hub_toyota_camry", "pcd", "5x114.3"),
        ("hub_toyota_camry", "center_bore", "60.1"),
        ("hub_toyota_camry", "max_diameter", '19"'),
        ("hub_mazda_cx5", "pcd", "5x114.3"),
        ("hub_mazda_cx5", "center_bore", "67.1"),
        ("hub_mazda_cx5", "max_diameter", '20"'),
        ("hub_renault_megane", "pcd", "4x100"),
        ("hub_renault_megane", "center_bore", "60.1"),
        ("hub_renault_megane", "max_diameter", '17"'),
        ("hub_bmw_x5", "pcd", "5x112"),
        ("hub_bmw_x5", "center_bore", "66.6"),
        ("hub_bmw_x5", "max_diameter", '22"'),

        ("rim_oz_ultra_18", "pcd", "5x112"),
        ("rim_oz_ultra_18", "diameter", "R18"),
        ("rim_oz_ultra_18", "weight_kg", "8.4"),
        ("rim_borbet_y17", "pcd", "5x112"),
        ("rim_borbet_y17", "diameter", "R17"),
        ("rim_borbet_y17", "weight_kg", "9.9"),
        ("rim_enkei_rpf1_18", "pcd", "5x114.3"),
        ("rim_enkei_rpf1_18", "diameter", "R18"),
        ("rim_enkei_rpf1_18", "weight_kg", "8.2"),
        ("rim_rays_te37_19", "pcd", "5x114.3"),
        ("rim_rays_te37_19", "diameter", "R19"),
        ("rim_rays_te37_19", "weight_kg", "8.8"),
        ("rim_dez_tr_16", "pcd", "5x112"),
        ("rim_dez_tr_16", "diameter", "R16"),
        ("rim_dez_tr_16", "weight_kg", "9.6"),
        ("rim_ronal_r56_17", "pcd", "5x114.3"),
        ("rim_ronal_r56_17", "diameter", "R17"),
        ("rim_ronal_r56_17", "weight_kg", "10.1"),
        ("rim_vossen_hf5_20", "pcd", "5x120"),
        ("rim_vossen_hf5_20", "diameter", "R20"),
        ("rim_vossen_hf5_20", "weight_kg", "11.4"),
        ("rim_kfz_steel_16", "pcd", "4x100"),
        ("rim_kfz_steel_16", "diameter", "R16"),
        ("rim_kfz_steel_16", "weight_kg", "9.1"),

        ("tire_bridgestone_18", "diameter", "R18"),
        ("tire_bridgestone_18", "season", "winter"),
        ("tire_nokian_17", "diameter", "R17"),
        ("tire_nokian_17", "season", "winter"),
        ("tire_hankook_16", "diameter", "R16"),
        ("tire_hankook_16", "season", "summer"),
        ("tire_kumho_17", "diameter", "R17"),
        ("tire_kumho_17", "season", "summer"),
        ("tire_toyo_19", "diameter", "R19"),
        ("tire_toyo_19", "season", "summer"),
        ("tire_dunlop_20", "diameter", "R20"),
        ("tire_dunlop_20", "season", "summer"),
        ("tire_barum_16", "diameter", "R16"),
        ("tire_barum_16", "season", "summer"),
        ("tire_michelin_cc2_18", "diameter", "R18"),
        ("tire_michelin_cc2_18", "season", "all_season"),

        ("brake_brembo_312", "pcd", "5x112"),
        ("brake_zimmermann_340", "pcd", "5x112"),
        ("brake_ate_300", "pcd", "5x120"),
        ("brake_trw_280", "pcd", "4x100"),
    ]
    for pid, k, v in eav_records:
        cur.execute("INSERT INTO product_eav_meta (product_id, meta_key, meta_value) VALUES (?, ?, ?)", (pid, k, v))

    conn.commit()
    conn.close()
    print(f"Created auto_parts.db at {db_path} with {len(products)} products and {len(eav_records)} EAV rows.")

if __name__ == "__main__":
    init_auto_parts_db()
