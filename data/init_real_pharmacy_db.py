import sqlite3
import os

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "real_pharmacy.db"))

def build_pharmacy_db(db_path: str = DB_PATH):
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except Exception:
            pass

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Raw, uncurated client schema with natural pharmaceutical naming
    cur.execute("""
    CREATE TABLE IF NOT EXISTS products (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        category TEXT NOT NULL,
        brand TEXT,
        price REAL NOT NULL,
        stock_quantity INTEGER NOT NULL,
        performance_score REAL DEFAULT 85.0
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS product_eav_meta (
        product_id TEXT NOT NULL,
        meta_key TEXT NOT NULL,
        meta_value TEXT NOT NULL,
        PRIMARY KEY (product_id, meta_key)
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS merchants (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        rating REAL DEFAULT 4.8,
        delivery_days INTEGER DEFAULT 1,
        is_official INTEGER DEFAULT 1
    );
    """)

    cur.execute("DELETE FROM merchants;")
    cur.execute("INSERT INTO merchants VALUES ('pharma_central', 'Аптека Низьких Цін (ФармХолдинг)', 4.9, 1, 1);")

    # Demo pharmaceutical assortment (preserving med_01..med_10 IDs & constraints)
    drugs = [
        # Analgesics / NSAIDs
        ("med_01", "Парацетамол-Дарниця 500 мг табл. №10", "analgesic", "Дарниця", 24.50, 120, 82.0, {
            "active_substance": "paracetamol", "dose_mg": "500", "max_daily_dose_mg": "4000",
            "contraindicated_with": "alcohol,warfarin", "form": "tablets", "prescription": "no"
        }),
        ("med_02", "Панадол Екстра 500 мг/65 мг табл. №12", "analgesic", "GlaxoSmithKline", 89.00, 45, 88.5, {
            "active_substance": "paracetamol", "dose_mg": "500", "max_daily_dose_mg": "4000",
            "contraindicated_with": "alcohol,warfarin", "form": "tablets", "prescription": "no"
        }),
        ("med_03", "Ібупрофен 400 мг капсули №20", "nsaid", "Фармак", 68.20, 80, 86.0, {
            "active_substance": "ibuprofen", "dose_mg": "400", "max_daily_dose_mg": "1200",
            "contraindicated_with": "aspirin,warfarin,gastric_ulcer", "form": "capsules", "prescription": "no"
        }),
        ("med_04", "Нурофен Форте 400 мг табл. №12", "nsaid", "Reckitt Benckiser", 115.00, 60, 91.0, {
            "active_substance": "ibuprofen", "dose_mg": "400", "max_daily_dose_mg": "1200",
            "contraindicated_with": "aspirin,warfarin,gastric_ulcer", "form": "tablets", "prescription": "no"
        }),
        ("med_05", "Аспірин Кардіо 100 мг табл. №28", "antiplatelet", "Bayer", 142.00, 110, 92.5, {
            "active_substance": "acetylsalicylic_acid", "dose_mg": "100", "max_daily_dose_mg": "300",
            "contraindicated_with": "ibuprofen,warfarin,active_bleeding", "form": "tablets", "prescription": "no"
        }),
        ("med_06", "Варфарин 2.5 мг табл. №100", "anticoagulant", "Orion Pharma", 210.00, 25, 93.0, {
            "active_substance": "warfarin", "dose_mg": "2.5", "max_daily_dose_mg": "10",
            "contraindicated_with": "aspirin,ibuprofen,paracetamol_high_dose", "form": "tablets", "prescription": "yes"
        }),
        # Cardiovascular / Nitrates & ED
        ("med_07", "Нітрогліцерин 0.5 мг табл. сублінгв. №40", "cardiovascular", "Здоров'я", 32.00, 95, 89.0, {
            "active_substance": "nitroglycerin", "dose_mg": "0.5", "max_daily_dose_mg": "2.0",
            "contraindicated_with": "sildenafil,tadalafil,severe_hypotension", "form": "sublingual", "prescription": "yes"
        }),
        ("med_08", "Силденафіл 50 мг табл. №4", "urological", "Тева", 275.00, 40, 90.0, {
            "active_substance": "sildenafil", "dose_mg": "50", "max_daily_dose_mg": "100",
            "contraindicated_with": "nitroglycerin,isosorbide,nitrates", "form": "tablets", "prescription": "yes"
        }),
        # Antibiotics & Probiotics
        ("med_09", "Амоксиклав 875/125 мг табл. №14", "antibiotic", "Sandoz", 340.00, 30, 95.5, {
            "active_substance": "amoxicillin_clavulanate", "dose_mg": "1000", "max_daily_dose_mg": "2000",
            "contraindicated_with": "penicillin_allergy", "form": "tablets", "prescription": "yes"
        }),
        ("med_10", "Лінекс Форте капсули №14", "probiotic", "Lek", 285.00, 50, 88.0, {
            "active_substance": "lactobacillus_acidophilus", "dose_mg": "0", "max_daily_dose_mg": "0",
            "contraindicated_with": "none", "form": "capsules", "prescription": "no"
        }),
        ("med_11", "Німесил 100 мг гранули саше №30", "nsaid", "Berlin-Chemie", 365.00, 42, 93.5, {
            "active_substance": "nimesulide", "dose_mg": "100", "max_daily_dose_mg": "200",
            "contraindicated_with": "paracetamol,hepatic_impairment,warfarin", "form": "granules", "prescription": "yes"
        }),
        ("med_12", "Дексалгін 25 мг табл. №10", "nsaid", "Menarini", 178.00, 55, 92.0, {
            "active_substance": "dexketoprofen", "dose_mg": "25", "max_daily_dose_mg": "75",
            "contraindicated_with": "aspirin,warfarin,gastric_ulcer", "form": "tablets", "prescription": "no"
        }),
        ("med_13", "Мефенамінова кислота 500 мг табл. №20", "nsaid", "Дарниця", 74.50, 90, 79.5, {
            "active_substance": "mefenamic_acid", "dose_mg": "500", "max_daily_dose_mg": "1500",
            "contraindicated_with": "aspirin,gastric_ulcer", "form": "tablets", "prescription": "no"
        }),
        ("med_14", "Цитрамон-Дарниця табл. №10", "analgesic", "Дарниця", 21.80, 150, 76.0, {
            "active_substance": "acetylsalicylic_acid_paracetamol_caffeine", "dose_mg": "420", "max_daily_dose_mg": "1680",
            "contraindicated_with": "warfarin,hypertension", "form": "tablets", "prescription": "no"
        }),
        ("med_15", "Солпадеїн Актив табл. розчинні №12", "analgesic", "Perrigo", 156.00, 38, 90.5, {
            "active_substance": "paracetamol_caffeine", "dose_mg": "500", "max_daily_dose_mg": "4000",
            "contraindicated_with": "alcohol,warfarin", "form": "effervescent", "prescription": "no"
        }),
        ("med_16", "Цетрин 10 мг табл. №20", "antihistamine", "Dr. Reddy's", 118.00, 72, 87.5, {
            "active_substance": "cetirizine", "dose_mg": "10", "max_daily_dose_mg": "10",
            "contraindicated_with": "alcohol", "form": "tablets", "prescription": "no"
        }),
        ("med_17", "Еріус 5 мг табл. №10", "antihistamine", "Bayer", 248.00, 34, 93.0, {
            "active_substance": "desloratadine", "dose_mg": "5", "max_daily_dose_mg": "5",
            "contraindicated_with": "none", "form": "tablets", "prescription": "no"
        }),
        ("med_18", "Лоратадин-Стома 10 мг табл. №10", "antihistamine", "Стома", 28.40, 110, 78.0, {
            "active_substance": "loratadine", "dose_mg": "10", "max_daily_dose_mg": "10",
            "contraindicated_with": "alcohol", "form": "tablets", "prescription": "no"
        }),
        ("med_19", "Омепразол-Астрафарм 20 мг капсули №30", "gastro", "Астрафарм", 59.00, 85, 85.5, {
            "active_substance": "omeprazole", "dose_mg": "20", "max_daily_dose_mg": "40",
            "contraindicated_with": "clopidogrel", "form": "capsules", "prescription": "no"
        }),
        ("med_20", "Нексіум 20 мг табл. №14", "gastro", "AstraZeneca", 229.00, 40, 94.0, {
            "active_substance": "esomeprazole", "dose_mg": "20", "max_daily_dose_mg": "40",
            "contraindicated_with": "atazanavir", "form": "tablets", "prescription": "no"
        }),
        ("med_21", "Фосфалюгель гель 16 г саше №20", "gastro", "Sanofi", 268.00, 48, 89.5, {
            "active_substance": "aluminum_phosphate", "dose_mg": "10400", "max_daily_dose_mg": "62400",
            "contraindicated_with": "severe_renal_failure", "form": "gel", "prescription": "no"
        }),
        ("med_22", "Смекта порошок 3 г саше №12", "gastro", "Ipsen", 164.00, 65, 86.5, {
            "active_substance": "diosmectite", "dose_mg": "3000", "max_daily_dose_mg": "9000",
            "contraindicated_with": "intestinal_obstruction", "form": "powder", "prescription": "no"
        }),
        ("med_23", "Креон 10000 капсули №20", "gastro", " Abbott", 235.00, 52, 92.5, {
            "active_substance": "pancreatin", "dose_mg": "150", "max_daily_dose_mg": "600",
            "contraindicated_with": "none", "form": "capsules", "prescription": "no"
        }),
        ("med_24", "Бісопролол-Тева 5 мг табл. №30", "cardiovascular", "Тева", 92.00, 64, 90.0, {
            "active_substance": "bisoprolol", "dose_mg": "5", "max_daily_dose_mg": "10",
            "contraindicated_with": "verapamil,bradycardia,asthma", "form": "tablets", "prescription": "yes"
        }),
        ("med_25", "Амлодипін-КВ 5 мг табл. №30", "cardiovascular", "Київський вітамінний завод", 68.50, 75, 86.0, {
            "active_substance": "amlodipine", "dose_mg": "5", "max_daily_dose_mg": "10",
            "contraindicated_with": "severe_hypotension", "form": "tablets", "prescription": "yes"
        }),
        ("med_26", "Конкор 5 мг табл. №30", "cardiovascular", "Merck", 196.00, 44, 94.5, {
            "active_substance": "bisoprolol", "dose_mg": "5", "max_daily_dose_mg": "10",
            "contraindicated_with": "verapamil,bradycardia", "form": "tablets", "prescription": "yes"
        }),
        ("med_27", "Азитроміцин-БХФЗ 500 мг капсули №3", "antibiotic", "БХФЗ", 96.00, 36, 87.0, {
            "active_substance": "azithromycin", "dose_mg": "500", "max_daily_dose_mg": "500",
            "contraindicated_with": "macrolide_allergy,qt_prolongation", "form": "capsules", "prescription": "yes"
        }),
        ("med_28", "Сумамед 500 мг табл. №3", "antibiotic", "Pliva", 312.00, 22, 94.0, {
            "active_substance": "azithromycin", "dose_mg": "500", "max_daily_dose_mg": "500",
            "contraindicated_with": "macrolide_allergy,qt_prolongation", "form": "tablets", "prescription": "yes"
        }),
        ("med_29", "Цефікс 400 мг капсули №5", "antibiotic", "Kusum", 289.00, 28, 93.0, {
            "active_substance": "cefixime", "dose_mg": "400", "max_daily_dose_mg": "400",
            "contraindicated_with": "cephalosporin_allergy", "form": "capsules", "prescription": "yes"
        }),
        ("med_30", "Ентерожерміна Форте суспензія 5 мл №10", "probiotic", "Sanofi", 358.00, 40, 93.5, {
            "active_substance": "bacillus_clausii", "dose_mg": "0", "max_daily_dose_mg": "0",
            "contraindicated_with": "none", "form": "suspension", "prescription": "no"
        }),
        ("med_31", " Йогурт Розель капсули №30", "probiotic", "Pharmascience", 162.00, 58, 84.0, {
            "active_substance": "lactobacillus_rhamnosus", "dose_mg": "0", "max_daily_dose_mg": "0",
            "contraindicated_with": "none", "form": "capsules", "prescription": "no"
        }),
        ("med_32", "Хлоргексидин біглюконат 0.05% 100 мл", "antiseptic", "Віола", 26.00, 140, 81.0, {
            "active_substance": "chlorhexidine", "dose_mg": "50", "max_daily_dose_mg": "200",
            "contraindicated_with": "dermatitis", "form": "solution", "prescription": "no"
        }),
        ("med_33", "Декасан розчин 0.02% 200 мл", "antiseptic", "Юрія-Фарм", 138.00, 70, 88.5, {
            "active_substance": "decamethoxine", "dose_mg": "40", "max_daily_dose_mg": "160",
            "contraindicated_with": "none", "form": "solution", "prescription": "no"
        }),
        ("med_34", "Магне-B6 табл. №50", "vitamins", "Sanofi", 295.00, 50, 91.0, {
            "active_substance": "magnesium_lactate_pyridoxine", "dose_mg": "470", "max_daily_dose_mg": "2820",
            "contraindicated_with": "levodopa,severe_renal_failure", "form": "tablets", "prescription": "no"
        }),
        ("med_35", "Аквадетрим Вітамін D3 15000 МО/мл 10 мл", "vitamins", "Polpharma", 184.00, 62, 90.0, {
            "active_substance": "colecalciferol", "dose_mg": "0", "max_daily_dose_mg": "0",
            "contraindicated_with": "hypercalcemia", "form": "drops", "prescription": "no"
        }),
        ("med_36", "Аскорбінова кислота-КВ 500 мг табл. жувальні №30", "vitamins", "Київський вітамінний завод", 64.00, 105, 83.5, {
            "active_substance": "ascorbic_acid", "dose_mg": "500", "max_daily_dose_mg": "1000",
            "contraindicated_with": "nephrolithiasis", "form": "tablets", "prescription": "no"
        }),
    ]

    cur.execute("DELETE FROM products;")
    cur.execute("DELETE FROM product_eav_meta;")

    for m_id, name, cat, brand, price, qty, perf_score, meta in drugs:
        cur.execute(
            "INSERT INTO products (id, name, category, brand, price, stock_quantity, performance_score) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (m_id, name.strip(), cat, brand.strip(), price, qty, perf_score)
        )
        for k, v in meta.items():
            cur.execute(
                "INSERT INTO product_eav_meta (product_id, meta_key, meta_value) VALUES (?, ?, ?)",
                (m_id, k, v)
            )

    conn.commit()
    conn.close()
    print(f"Created real pharmacy database at {db_path}")

if __name__ == "__main__":
    build_pharmacy_db()
