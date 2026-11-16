"""Deterministic jury-demo source: an UNSEEN schema the system has
never been tuned on.

Deliberate properties (mirroring the evaluation polygon):

- entity table ``tvaryny`` is NOT the largest table (``vlastyvosti``
  EAV rows outnumber it ~4x) -- a row-count heuristic picks the EAV
  store and fails;
- table/column names are Ukrainian transliteration, never "products";
- FK dimension ``seredovyshcha`` (habitats), EAV ``vlastyvosti``,
  junction ``tvaryny_vystavky`` -> ``vystavky`` (exhibits), and a 1:1
  detail ``tvaryny_medyka`` (vet records) that must NOT win candidacy;
- names/descriptions carry enough free text for semantic search but
  stay synthetic and disclosed.

Output: ``data/jury_zoo.db`` (~420 entities). Run:
    python build_jury_demo.py
"""
import os
import random
import sqlite3
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
OUT = os.path.join(ROOT, "data", "jury_zoo.db")

HABITATS = ["савана", "тайга", "джунглі", "пустеля", "полярний лід",
            "прісні води", "океан", "гірські луки"]
EXHIBITS = [("Нічний світ", "павільйон 1"), ("Великі кішки", "павільйон 2"),
            ("Птахи Півдня", "авіарій"), ("Діти савани", "відкритий вольєр"),
            ("Мешканці льоду", "павільйон 3"), ("Рептилії та амфібії", "терраріум")]

# (base_name, habitat, eav traits, description template)
SPECIES = [
    ("Лев", "савана", {"вага": "190 кг", "активність": "полює вночі"},
     "великий хижак, що живе прайдами та спить до 20 годин на добу"),
    ("Жираф", "савана", {"зріст": "5.5 м", "язик": "45 см"},
     "найвища тварина, дістає листя акацій довгим синім язиком"),
    ("Зебра", "савана", {"смуги": "унікальний малюнок"},
     "смугаста стадна тварина — кожна особина несе власний візерунок смуг"),
    ("Слон африканський", "савана", {"вага": "6000 кг", "хобот": "40000 м'язів"},
     "гігант із хоботом-рукою, чує низькочастотний гуркіт за кілометри"),
    ("Тигр амурський", "тайга", {"вага": "220 кг", "смуги": "помаранчеві"},
     "найбільший кіт, територія одного самця сягає сотень кілометрів"),
    ("Ведмідь бурий", "тайга", {"вага": "350 кг", "зима": "спить у барлозі"},
     "всеїдний мешканець хвойних лісів, восени нагулює жир на ягодах"),
    ("Рись", "тайга", {"вуха": "пучки на вухах", "погляд": "помічає мишу за 75 м"},
     "скритний хижак з коротким хвостом і вушними пензликами"),
    ("Ягуар", "джунглі", {"щелепи": "пробиває панцир черепахи"},
     "найсильніші щелепи серед котів, полює біля води"),
    ("Мандрил", "джунглі", {"обличчя": "яскраве забарвлення морди"},
     "найбарвистіший примат, забарвлення морди служить сигналом статусу"),
    ("Тапір", "джунглі", {"хоботок": "короткий хоботок"},
     "тихий травоїдний ссавець із маленьким хоботком, тримається водойм"),
    ("Верблюд", "пустеля", {"горб": "запас жиру", "вода": "обходиться тижнями"},
     "корабель пустелі, витримує спеку й пилові бурі"),
    ("Фенек", "пустеля", {"вуха": "величезні", "активність": "нічна"},
     "крихітна лисиця з вухами-радарами, охолоджується через них"),
    ("Скорпіон імператорський", "пустеля", {"жало": "на кінці хвоста"},
     "світиться в ультрафіолеті, полює на комах уночі"),
    ("Ведмідь білий", "полярний лід", {"вага": "450 кг", "шерсть": "порожниста"},
     "найбільший хижак суші, плаває десятки кілометрів за тюленями"),
    ("Пінгвін імператорський", "полярний лід", {"висота": "1.2 м", "пір'я": "водонепроникне"},
     "єдиний птах, що виводить пташеня серед льоду взимку"),
    ("Тюлень Ведделя", "полярний лід", {"занурення": "до 600 м"},
     "підлідний пірнальник, протримується під водою до години"),
    ("Кайман", "прісні води", {"зуби": "конічні", "засідка": "нерухомо чекає"},
     "хижак із міцними щелепами, непомітно чатує на мілководді"),
    ("Капібара", "прісні води", {"вага": "55 кг", "характер": "найспокійніший"},
     "найбільший гризун світу, веде напівводний спосіб життя біля водойм"),
    ("Скат електричний", "прісні води", {"розряд": "600 В"},
     "генерує електричний розряд, що вражає здобич і захисників"),
    ("Косатка", "океан", {"мозок": "другий за розміром", "мова": "діалекти зграй"},
     "вершина морського харчового ланцюга, полює зграями"),
    ("Черепаха морська", "океан", {"вік": "до 100 років", "міграція": "тис. км"},
     "давня мандрівниця океанів, повертається на рідний пляж"),
    ("Восьминіг", "океан", {"кров": "синя", "розум": "розв'язує головоломки"},
     "безхребетний геній: три серця, синя кров, камуфляж за секунду"),
    ("Коза гірська", "гірські луки", {"копита": "чіпкі для скель"},
     "альпініст природи, упевнено стоїть на прямовисних уступах"),
    ("Орел беркут", "гірські луки", {"розмах": "2.3 м", "зір": "бачить зайця за 3 км"},
     "король гірського неба, влучає в здобич з висоти польоту"),
    ("Сурикат", "пустеля", {"черга": "завжди дозорний"},
     "колоніальний сторож, співпрацює з десятками родичів"),
    ("Гієна плямиста", "савана", {"щелепа": "роздрібнює кістки"},
     "мисливець-кооператор із «сміхом», не лише падальник"),
]

# Per-species distinguishing detail -- each species gets its own second
# sentence so documents are not flattened into one vector neighbourhood.
SPECIES_DETAIL = {
    "Лев": "грива самця темнішає з віком, рев чути на кілька кілометрів",
    "Жираф": "серце важить понад десять кілограмів, щоб гнати кров у голову",
    "Зебра": "чорно-білі смуги збивають з пантелику мух цеце та хижаків",
    "Слон африканський": "бивні ростуть усе життя, спілкується інфразвуком",
    "Тигр амурський": "смугастий візерунок у кожного як відбиток пальця",
    "Ведмідь бурий": "після зимового сну в барлозі втрачає третину ваги",
    "Рись": "полює з засідки на деревах, стрибає на здобич згори",
    "Ягуар": "на відміну від інших котів полюбляє воду й плаває",
    "Мандрил": "самці демонструють синьо-червону морду як знак сили",
    "Тапір": "дитинча народжується в кавунову смужку для маскування",
    "Верблюд": "шерсть скидає згустками навесні, губи не бояться кактусів",
    "Фенек": "найменша лисиця світу, вуха більші за голову",
    "Скорпіон імператорський": "за темряви полохлива тварина з м'яким отруйним жалом",
    "Ведмідь білий": "чорна шкіра під білою шерстю, полює з льоду",
    "Пінгвін імператорський": "батько висиджує яйце на лапах два місяці без їжі",
    "Тюлень Ведделя": "прогризає у льоду дихальні отвори зубами",
    "Кайман": "очі й ніздрі на верхівці морди -- видно над водою",
    "Капібара": "напівводний спосіб життя, пірнає, щоб втекти від ягуара",
    "Скат електричний": "батареї м'язів дають розряд для полювання й оборони",
    "Косатка": "зграї передають техніку полювання через покоління",
    "Черепаха морська": "дитинчата йдуть до океану за відблиском місяця на хвилях",
    "Восьминіг": "міняє колір і фактуру шкіри миттю, прослизує крізь щілину з монету",
    "Коза гірська": "інстинкти змушують козенят стрибати за матір'ю з першого дня",
    "Орел беркут": "піке на здобич розвиває швидкість гоночного авто",
    "Сурикат": "черговий стоїть стовпчиком на високій точці й попереджає гурт",
    "Гієна плямиста": "самиці керують кланом і крупніші за самців",
}


def build(out: str = OUT, n_animals: int = 420, seed: int = 20261112) -> None:
    rng = random.Random(seed)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    if os.path.exists(out):
        os.unlink(out)
    conn = sqlite3.connect(out)
    c = conn.cursor()
    c.executescript(
        """
        CREATE TABLE seredovyshcha (
            id INTEGER PRIMARY KEY, nazva TEXT NOT NULL);
        CREATE TABLE tvaryny (
            id INTEGER PRIMARY KEY,
            imya TEXT NOT NULL,
            opys TEXT,
            sered_id INTEGER REFERENCES seredovyshcha(id));
        CREATE TABLE vlastyvosti (
            tvaryna_id INTEGER NOT NULL REFERENCES tvaryny(id),
            kliuch TEXT NOT NULL,
            znachennya TEXT,
            PRIMARY KEY (tvaryna_id, kliuch));
        CREATE TABLE vystavky (
            id INTEGER PRIMARY KEY, nazva TEXT NOT NULL, zal TEXT);
        CREATE TABLE tvaryny_vystavky (
            tvaryna_id INTEGER NOT NULL REFERENCES tvaryny(id),
            vystavka_id INTEGER NOT NULL REFERENCES vystavky(id),
            PRIMARY KEY (tvaryna_id, vystavka_id));
        CREATE TABLE tvaryny_medyka (
            tvaryna_id INTEGER PRIMARY KEY REFERENCES tvaryny(id),
            zapis TEXT);
        """)
    c.executemany("INSERT INTO seredovyshcha(nazva) VALUES (?)",
                  [(h,) for h in HABITATS])
    c.executemany("INSERT INTO vystavky(nazva, zal) VALUES (?, ?)", EXHIBITS)

    used = set()
    for i in range(1, n_animals + 1):
        sp = SPECIES[i % len(SPECIES)]
        base, hab, eav, desc = sp
        name = base if i <= len(SPECIES) else f"{base} №{i // len(SPECIES)}"
        if name in used:
            name = f"{base} особина {i}"
        used.add(name)
        extra = SPECIES_DETAIL.get(base, desc)
        c.execute(
            "INSERT INTO tvaryny(id, imya, opys, sered_id) VALUES (?, ?, ?, ?)",
            (i, name, f"{desc}; {extra}", HABITATS.index(hab) + 1))
        for k, v in eav.items():
            c.execute("INSERT INTO vlastyvosti VALUES (?, ?, ?)", (i, k, v))
        # 2-3 extra generic traits so EAV stays the biggest table
        for k in ("спадок", "режим дня"):
            c.execute("INSERT INTO vlastyvosti VALUES (?, ?, ?)",
                      (i, k, extra.split(",")[0].split(";")[0][:40]))
        # exhibit membership: 1-2 exhibits per animal
        for vx in rng.sample(range(1, len(EXHIBITS) + 1), rng.choice((1, 2))):
            c.execute("INSERT INTO tvaryny_vystavky VALUES (?, ?)", (i, vx))
        c.execute("INSERT INTO tvaryny_medyka VALUES (?, ?)",
                  (i, f"огляд {rng.randint(1, 12)}.{rng.randint(1, 28)}: норма"))

    conn.commit()
    counts = {t: c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
              for t in ("seredovyshcha", "tvaryny", "vlastyvosti",
                        "vystavky", "tvaryny_vystavky", "tvaryny_medyka")}
    conn.close()
    print(f"wrote {out}")
    for t, n in counts.items():
        print(f"  {t}: {n}")


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else OUT)
