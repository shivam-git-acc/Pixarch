"""Synthetic identity and financial document specimens.

Why these nine subtypes are generated rather than fetched
---------------------------------------------------------
Wikimedia Commons holds nine usable Indian ID specimens in total and nothing
more (verified, not inherited from a comment). Everything else on the open web
is a photograph of a *real person's* card: name, father's name, address, photo
and the number itself. `data/raw/` is gitignored but `data/manifest.csv` is
committed and carries `source_url`, and `fetch_dataset.py` rebuilds the corpus
from it -- so fetching those would publish a permanent, public index of real
people's identity documents and have every cloner re-download them. The same
argument applies verbatim to cards, passbooks and bank statements.

Generating them instead is also better data: exact OCR ground truth for free,
balanced class counts, and controllable difficulty.

Safety properties of the generated values
-----------------------------------------
* Aadhaar numbers begin with 0 or 1. UIDAI never issues those, so no generated
  number can collide with a real Aadhaar.
* Card numbers are built to **fail** the Luhn checksum, so none can be a valid
  payment card.
* Every person, bank and school name is fictional.

Deliberately **no** "SPECIMEN" watermark: a token present in every synthetic
document and no real one is a label leak straight into BM25 -- the same bug as
the filename leak already documented in schema.sql.
"""
from __future__ import annotations

import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .fetchers import Record

# ------------------------------------------------------------------ fonts --
_CANDIDATES = {
    "sans": ["/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
             "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
    "sans_bold": ["/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
                  "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"],
    "serif_bold": ["/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
                   "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"],
    "mono": ["/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
             "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"],
    "deva": ["/usr/share/fonts/truetype/lohit-devanagari/Lohit-Devanagari.ttf",
             "/usr/share/fonts/truetype/samyak/Samyak-Devanagari.ttf"],
}


def _font(kind: str, size: int) -> ImageFont.FreeTypeFont:
    for path in _CANDIDATES[kind]:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


# ------------------------------------------------------------ fake values --
_FIRST = ["Arun", "Meera", "Rohit", "Kavya", "Imran", "Sneha", "Vikram", "Anita",
          "Farhan", "Divya", "Rajesh", "Pooja", "Sanjay", "Nisha", "Aditya"]
_LAST = ["Sharma", "Iyer", "Khan", "Reddy", "Bose", "Nair", "Mehta", "Gupta",
         "Pillai", "Chawla", "Verma", "Das", "Joshi", "Kulkarni"]
_CITY = ["Pune", "Indore", "Kochi", "Jaipur", "Nagpur", "Surat", "Bhopal", "Patna"]
_BANKS = ["Meridian Union Bank", "Deccan Commercial Bank", "Sahyadri Bank",
          "Northern Crest Bank", "Vindhya Cooperative Bank"]
_SCHOOLS = ["Greenfield Public School", "St. Aldwyn High School",
            "Sunrise Vidya Mandir", "Lakeview Academy"]


def _name(rng): return f"{rng.choice(_FIRST)} {rng.choice(_LAST)}"


def _dob(rng): return f"{rng.randint(1,28):02d}/{rng.randint(1,12):02d}/{rng.randint(1960,2004)}"


def _aadhaar_no(rng):
    """Starts with 0 or 1 -- UIDAI never issues those, so it cannot collide."""
    d = [str(rng.choice([0, 1]))] + [str(rng.randint(0, 9)) for _ in range(11)]
    s = "".join(d)
    return f"{s[0:4]} {s[4:8]} {s[8:12]}"


def _pan_no(rng):
    up = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return ("".join(rng.choice(up) for _ in range(5))
            + "".join(str(rng.randint(0, 9)) for _ in range(4)) + rng.choice(up))


def _card_no(rng):
    """16 digits engineered to FAIL Luhn, so it is never a valid card."""
    base = [rng.randint(0, 9) for _ in range(15)]
    tot = 0
    for i, d in enumerate(reversed(base)):
        d = d * 2 if i % 2 == 0 else d
        tot += d - 9 if d > 9 else d
    good = (10 - tot % 10) % 10
    bad = (good + 5) % 10                      # guaranteed wrong check digit
    s = "".join(map(str, base)) + str(bad)
    return " ".join(s[i:i + 4] for i in range(0, 16, 4))


def _txn_rows(rng, n):
    kinds = ["UPI/P2M/", "NEFT/", "ATM WDL ", "IMPS/", "CHQ DEP ", "POS "]
    bal = rng.randint(20000, 90000)
    out = []
    for _ in range(n):
        amt = rng.randint(120, 9000)
        dr = rng.random() < 0.6
        bal += -amt if dr else amt
        out.append((f"{rng.randint(1,28):02d}/{rng.randint(1,12):02d}/2025",
                    f"{rng.choice(kinds)}{rng.randint(10**7, 10**8)}",
                    f"{amt:,.2f}" if dr else "", f"{amt:,.2f}" if not dr else "",
                    f"{bal:,.2f}"))
    return out


# --------------------------------------------------------------- renderers --
def _card_base(rng, w=1012, h=638, top=(228, 236, 248), bot=(250, 238, 240)):
    """Card-sized canvas with a soft vertical gradient, like a printed card."""
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    for y in range(h):
        t = y / h
        d.line([(0, y), (w, y)], fill=tuple(int(top[i] + (bot[i] - top[i]) * t) for i in range(3)))
    return img, d


def _aadhaar(rng):
    img, d = _card_base(rng, top=(236, 240, 250), bot=(252, 236, 232))
    d.text((34, 20), "भारत सरकार", font=_font("deva", 34), fill=(20, 20, 20))
    d.text((34, 64), "GOVERNMENT OF INDIA", font=_font("sans_bold", 30), fill=(20, 20, 20))
    d.text((34, 104), "विशिष्ट पहचान प्राधिकरण", font=_font("deva", 24), fill=(60, 60, 60))
    d.text((34, 138), "UNIQUE IDENTIFICATION AUTHORITY OF INDIA",
           font=_font("sans", 21), fill=(60, 60, 60))
    d.rectangle([(740, 190), (960, 470)], outline=(120, 120, 120), width=2)
    d.text((790, 320), "PHOTO", font=_font("sans", 26), fill=(130, 130, 130))
    nm, dob = _name(rng), _dob(rng)
    y = 220
    for label, val in (("Name", nm), ("DOB", dob),
                       ("Gender", rng.choice(["MALE", "FEMALE"]))):
        d.text((40, y), f"{label}:", font=_font("sans", 25), fill=(70, 70, 70))
        d.text((190, y), val, font=_font("sans_bold", 27), fill=(10, 10, 10))
        y += 48
    no = _aadhaar_no(rng)
    d.text((40, 470), no, font=_font("mono", 52), fill=(10, 10, 10))
    d.text((40, 548), "आधार - आम आदमी का अधिकार", font=_font("deva", 26), fill=(40, 40, 40))
    txt = (f"भारत सरकार GOVERNMENT OF INDIA विशिष्ट पहचान प्राधिकरण "
           f"UNIQUE IDENTIFICATION AUTHORITY OF INDIA Name: {nm} DOB: {dob} {no} आधार")
    return img, txt


def _pan(rng):
    img, d = _card_base(rng, top=(226, 238, 250), bot=(250, 234, 236))
    d.text((30, 18), "आयकर विभाग", font=_font("deva", 32), fill=(20, 20, 20))
    d.text((30, 60), "INCOME TAX DEPARTMENT", font=_font("serif_bold", 31), fill=(20, 20, 20))
    d.text((640, 18), "भारत सरकार", font=_font("deva", 32), fill=(20, 20, 20))
    d.text((620, 60), "GOVT. OF INDIA", font=_font("serif_bold", 31), fill=(20, 20, 20))
    nm, fa, dob = _name(rng), _name(rng), _dob(rng)
    pan = _pan_no(rng)
    d.rectangle([(760, 300), (960, 540)], outline=(120, 120, 120), width=2)
    d.text((812, 410), "PHOTO", font=_font("sans", 24), fill=(130, 130, 130))
    y = 170
    for label, val in (("Name", nm), ("Father's Name", fa), ("Date of Birth", dob)):
        d.text((36, y), label, font=_font("sans", 22), fill=(80, 80, 80))
        d.text((36, y + 28), val, font=_font("sans_bold", 28), fill=(10, 10, 10))
        y += 78
    d.text((36, 418), "Permanent Account Number", font=_font("sans", 24), fill=(40, 70, 150))
    d.text((36, 452), pan, font=_font("mono", 46), fill=(10, 10, 10))
    d.text((36, 548), "Signature", font=_font("sans", 22), fill=(90, 90, 90))
    txt = (f"आयकर विभाग INCOME TAX DEPARTMENT भारत सरकार GOVT. OF INDIA Name {nm} "
           f"Father's Name {fa} Date of Birth {dob} Permanent Account Number {pan}")
    return img, txt


def _dl(rng):
    img, d = _card_base(rng, top=(232, 246, 234), bot=(246, 246, 232))
    d.text((30, 18), "INDIAN UNION DRIVING LICENCE", font=_font("serif_bold", 30), fill=(15, 15, 15))
    d.text((30, 58), f"TRANSPORT DEPARTMENT, GOVT OF {rng.choice(['MAHARASHTRA','KERALA','KARNATAKA'])}",
           font=_font("sans", 21), fill=(60, 60, 60))
    nm, dob = _name(rng), _dob(rng)
    dlno = f"{rng.choice(['MH','KL','KA'])}{rng.randint(10,99)} {rng.randint(10**10, 10**11-1)}"
    d.rectangle([(770, 160), (965, 400)], outline=(120, 120, 120), width=2)
    d.text((822, 270), "PHOTO", font=_font("sans", 24), fill=(130, 130, 130))
    d.text((30, 110), "DL No.", font=_font("sans", 22), fill=(80, 80, 80))
    d.text((140, 104), dlno, font=_font("mono", 34), fill=(10, 10, 10))
    y = 170
    for label, val in (("Name", nm), ("Date of Birth", dob),
                       ("Valid Till", f"31/12/{rng.randint(2027,2040)}"),
                       ("COV", rng.choice(["LMV, MCWG", "MCWG", "LMV, MCWG, TRANS"]))):
        d.text((30, y), f"{label}:", font=_font("sans", 22), fill=(80, 80, 80))
        d.text((250, y - 2), val, font=_font("sans_bold", 25), fill=(10, 10, 10))
        y += 48
    d.text((30, 470), f"Blood Group: {rng.choice(['A+','B+','O+','AB+'])}",
           font=_font("sans", 22), fill=(40, 40, 40))
    d.text((30, 520), f"Address: {rng.randint(1,99)}, {rng.choice(_CITY)}",
           font=_font("sans", 22), fill=(40, 40, 40))
    txt = (f"INDIAN UNION DRIVING LICENCE TRANSPORT DEPARTMENT DL No. {dlno} Name {nm} "
           f"Date of Birth {dob} COV LMV MCWG")
    return img, txt


def _passport(rng):
    img = Image.new("RGB", (1000, 1400), (246, 243, 236))
    d = ImageDraw.Draw(img)
    d.rectangle([(0, 0), (1000, 150)], fill=(232, 226, 212))
    d.text((40, 24), "भारत गणराज्य", font=_font("deva", 34), fill=(25, 25, 25))
    d.text((40, 72), "REPUBLIC OF INDIA", font=_font("serif_bold", 38), fill=(25, 25, 25))
    d.text((700, 72), "PASSPORT", font=_font("serif_bold", 34), fill=(25, 25, 25))
    sur, giv = rng.choice(_LAST), rng.choice(_FIRST)
    dob, city = _dob(rng), rng.choice(_CITY)
    pno = f"{rng.choice('ZPMLKJ')}{rng.randint(1000000, 9999999)}"
    d.rectangle([(60, 200), (320, 540)], outline=(120, 120, 120), width=2)
    d.text((140, 350), "PHOTO", font=_font("sans", 28), fill=(130, 130, 130))
    rows = [("Type / प्रकार", "P"), ("Country Code", "IND"), ("Passport No. / पासपोर्ट सं.", pno),
            ("Surname / उपनाम", sur.upper()), ("Given Name / दिया गया नाम", giv.upper()),
            ("Nationality / राष्ट्रीयता", "INDIAN"), ("Sex / लिंग", rng.choice(["M", "F"])),
            ("Date of Birth / जन्म तिथि", dob), ("Place of Birth / जन्म स्थान", city.upper()),
            ("Place of Issue / जारी करने का स्थान", city.upper()),
            ("Date of Issue", f"12/0{rng.randint(1,9)}/2021"),
            ("Date of Expiry", f"11/0{rng.randint(1,9)}/2031")]
    y = 200
    for label, val in rows:
        d.text((370, y), label, font=_font("sans", 18), fill=(90, 90, 90))
        d.text((370, y + 22), val, font=_font("sans_bold", 26), fill=(10, 10, 10))
        y += 62
    mrz1 = f"P<IND{sur.upper()}<<{giv.upper()}" + "<" * max(0, 30 - len(sur) - len(giv))
    mrz2 = f"{pno}<IND{dob.replace('/','')}" + "<" * 14
    d.rectangle([(0, 1230), (1000, 1400)], fill=(238, 235, 228))
    d.text((40, 1265), mrz1[:44], font=_font("mono", 30), fill=(10, 10, 10))
    d.text((40, 1320), mrz2[:44], font=_font("mono", 30), fill=(10, 10, 10))
    txt = (f"भारत गणराज्य REPUBLIC OF INDIA PASSPORT Type P Passport No. {pno} Surname "
           f"{sur.upper()} Given Name {giv.upper()} Nationality INDIAN Date of Birth {dob} "
           f"Place of Issue {city.upper()} {mrz1[:44]} {mrz2[:44]}")
    return img, txt


def _payment_card(rng, debit: bool):
    kind = "DEBIT CARD" if debit else "CREDIT CARD"
    img, d = _card_base(rng, top=(30, 52, 92), bot=(86, 46, 110)) if not debit else \
        _card_base(rng, top=(22, 78, 70), bot=(40, 110, 92))
    bank = rng.choice(_BANKS)
    no, nm = _card_no(rng), _name(rng).upper()
    exp = f"{rng.randint(1,12):02d}/{rng.randint(27,33)}"
    d.text((40, 36), bank, font=_font("sans_bold", 36), fill=(255, 255, 255))
    d.text((40, 84), kind, font=_font("sans", 24), fill=(214, 214, 224))
    d.rounded_rectangle([(52, 170), (152, 250)], radius=10, fill=(214, 182, 96))
    d.text((48, 300), no, font=_font("mono", 50), fill=(255, 255, 255))
    d.text((48, 404), "VALID THRU", font=_font("sans", 18), fill=(206, 206, 216))
    d.text((48, 428), exp, font=_font("mono", 34), fill=(255, 255, 255))
    d.text((300, 404), "CARDHOLDER", font=_font("sans", 18), fill=(206, 206, 216))
    d.text((300, 428), nm, font=_font("sans_bold", 30), fill=(255, 255, 255))
    d.text((760, 540), rng.choice(["RuPay", "GlobalPay"]), font=_font("sans_bold", 34),
           fill=(238, 238, 245))
    txt = f"{bank} {kind} {no} VALID THRU {exp} CARDHOLDER {nm}"
    return img, txt


def _passbook(rng):
    img = Image.new("RGB", (1240, 1754), (252, 251, 246))
    d = ImageDraw.Draw(img)
    bank = rng.choice(_BANKS)
    d.rectangle([(0, 0), (1240, 170)], fill=(28, 68, 126))
    d.text((40, 32), bank, font=_font("sans_bold", 44), fill=(255, 255, 255))
    d.text((40, 92), "SAVINGS BANK ACCOUNT PASSBOOK", font=_font("sans", 28), fill=(220, 228, 244))
    nm = _name(rng)
    acc = f"{rng.randint(10**10, 10**11-1)}"
    ifsc = f"{rng.choice(['MERU','DECC','SAHY'])}0{rng.randint(100000,999999)}"
    y = 210
    for label, val in (("Account Holder", nm), ("Account Number", acc), ("IFSC Code", ifsc),
                       ("Branch", f"{rng.choice(_CITY)} Main Branch"),
                       ("Account Type", "SAVINGS"), ("Customer ID", str(rng.randint(10**7, 10**8)))):
        d.text((48, y), f"{label}", font=_font("sans", 24), fill=(90, 90, 90))
        d.text((420, y - 2), val, font=_font("sans_bold", 27), fill=(15, 15, 15))
        y += 46
    y += 20
    hdr = ["Date", "Particulars", "Withdrawal", "Deposit", "Balance"]
    xs = [48, 230, 700, 890, 1060]
    d.rectangle([(40, y), (1200, y + 44)], fill=(226, 234, 246))
    for x, h in zip(xs, hdr):
        d.text((x, y + 10), h, font=_font("sans_bold", 24), fill=(20, 20, 20))
    y += 52
    rows = _txn_rows(rng, 20)
    for r in rows:
        for x, cell in zip(xs, r):
            d.text((x, y), cell, font=_font("sans", 22), fill=(25, 25, 25))
        d.line([(40, y + 34), (1200, y + 34)], fill=(222, 222, 222))
        y += 42
    txt = (f"{bank} SAVINGS BANK ACCOUNT PASSBOOK Account Holder {nm} Account Number {acc} "
           f"IFSC Code {ifsc} Date Particulars Withdrawal Deposit Balance "
           + " ".join(" ".join(c for c in r if c) for r in rows))
    return img, txt


def _bank_statement(rng):
    img = Image.new("RGB", (1240, 1754), "white")
    d = ImageDraw.Draw(img)
    bank = rng.choice(_BANKS)
    nm, acc = _name(rng), f"{rng.randint(10**10, 10**11-1)}"
    d.text((48, 44), bank, font=_font("sans_bold", 40), fill=(20, 40, 90))
    d.text((48, 98), "STATEMENT OF ACCOUNT", font=_font("sans_bold", 30), fill=(20, 20, 20))
    d.line([(48, 146), (1192, 146)], fill=(20, 40, 90), width=3)
    period = f"01/01/2025 to 31/03/2025"
    y = 172
    for label, val in (("Account Holder", nm), ("Account Number", acc),
                       ("Statement Period", period),
                       ("Branch", f"{rng.choice(_CITY)}"),
                       ("Opening Balance", f"{rng.randint(10000,80000):,.2f}")):
        d.text((48, y), label, font=_font("sans", 23), fill=(85, 85, 85))
        d.text((420, y - 2), val, font=_font("sans_bold", 25), fill=(15, 15, 15))
        y += 42
    y += 24
    hdr = ["Txn Date", "Description", "Debit", "Credit", "Balance"]
    xs = [48, 220, 700, 890, 1060]
    d.rectangle([(40, y), (1200, y + 42)], fill=(236, 240, 248))
    for x, h in zip(xs, hdr):
        d.text((x, y + 9), h, font=_font("sans_bold", 23), fill=(20, 20, 20))
    y += 50
    rows = _txn_rows(rng, 24)
    for r in rows:
        for x, cell in zip(xs, r):
            d.text((x, y), cell, font=_font("sans", 21), fill=(25, 25, 25))
        y += 40
    d.text((48, y + 30), "This is a computer generated statement.",
           font=_font("sans", 20), fill=(110, 110, 110))
    txt = (f"{bank} STATEMENT OF ACCOUNT Account Holder {nm} Account Number {acc} "
           f"Statement Period {period} Txn Date Description Debit Credit Balance "
           + " ".join(" ".join(c for c in r if c) for r in rows))
    return img, txt


def _report_card(rng):
    img = Image.new("RGB", (1240, 1754), (253, 253, 250))
    d = ImageDraw.Draw(img)
    school = rng.choice(_SCHOOLS)
    nm = _name(rng)
    cls = f"{rng.randint(6,12)}-{rng.choice('ABC')}"
    roll = str(rng.randint(1, 60))
    d.rectangle([(0, 0), (1240, 160)], fill=(120, 30, 50))
    d.text((48, 28), school, font=_font("serif_bold", 42), fill=(255, 255, 255))
    d.text((48, 90), "PROGRESS REPORT CARD  |  ACADEMIC SESSION 2025-26",
           font=_font("sans", 26), fill=(240, 226, 230))
    y = 200
    for label, val in (("Student Name", nm), ("Class / Section", cls),
                       ("Roll Number", roll), ("Admission No.", str(rng.randint(10000, 99999)))):
        d.text((48, y), label, font=_font("sans", 24), fill=(85, 85, 85))
        d.text((420, y - 2), val, font=_font("sans_bold", 27), fill=(15, 15, 15))
        y += 46
    y += 24
    subs = ["English", "Hindi", "Mathematics", "Science", "Social Science",
            "Computer Science", "Physical Education"]
    hdr = ["Subject", "Max Marks", "Marks Obtained", "Grade"]
    xs = [48, 520, 760, 1060]
    d.rectangle([(40, y), (1200, y + 44)], fill=(244, 232, 236))
    for x, h in zip(xs, hdr):
        d.text((x, y + 10), h, font=_font("sans_bold", 24), fill=(20, 20, 20))
    y += 54
    total = 0
    lines = []
    for s in subs:
        m = rng.randint(45, 99)
        total += m
        g = "A+" if m >= 90 else "A" if m >= 80 else "B" if m >= 65 else "C"
        for x, cell in zip(xs, [s, "100", str(m), g]):
            d.text((x, y), cell, font=_font("sans", 23), fill=(25, 25, 25))
        d.line([(40, y + 36), (1200, y + 36)], fill=(226, 226, 226))
        lines.append(f"{s} 100 {m} {g}")
        y += 46
    pct = total / len(subs)
    d.text((48, y + 24), f"Total: {total} / {len(subs)*100}",
           font=_font("sans_bold", 28), fill=(15, 15, 15))
    d.text((48, y + 68), f"Percentage: {pct:.1f}%   Result: {'PASS' if pct >= 40 else 'FAIL'}",
           font=_font("sans_bold", 28), fill=(15, 15, 15))
    d.text((48, y + 140), "Class Teacher", font=_font("sans", 22), fill=(100, 100, 100))
    d.text((900, y + 140), "Principal", font=_font("sans", 22), fill=(100, 100, 100))
    txt = (f"{school} PROGRESS REPORT CARD ACADEMIC SESSION 2025-26 Student Name {nm} "
           f"Class {cls} Roll Number {roll} Subject Max Marks Marks Obtained Grade "
           + " ".join(lines) + f" Total {total} Percentage {pct:.1f}")
    return img, txt


GENERATORS = {
    "aadhaar": _aadhaar,
    "pan": _pan,
    "dl": _dl,
    "passport": _passport,
    "credit_card": lambda rng: _payment_card(rng, debit=False),
    "debit_card": lambda rng: _payment_card(rng, debit=True),
    "passbook": _passbook,
    "bank_statement": _bank_statement,
    "report_card": _report_card,
}

# Subtypes whose real counterparts already exist in the corpus from Wikimedia
# Commons. Synthetic ids start after the real ones so the two never collide,
# and the real ones stay available as a held-out realism check.
_REAL_OFFSETS = {"aadhaar": 100, "pan": 100, "dl": 100, "passport": 100}


def generate(raw_dir: Path, n: int = 20, subtypes=None, seed: int = 0,
             gt_path: Path | None = None) -> list[Record]:
    """Render `n` specimens per subtype into data/raw/documents/<subtype>/."""
    from .fetchers import _merge_gt

    records: list[Record] = []
    gt_lines: list[dict] = []
    for subtype in (subtypes or GENERATORS):
        fn = GENERATORS[subtype]
        start = _REAL_OFFSETS.get(subtype, 0)
        made = 0
        for i in range(n):
            rng = random.Random(f"{seed}:{subtype}:{i}")
            img, text = fn(rng)
            item_id = f"doc_{subtype}_{start + i:03d}"
            dest = raw_dir / "documents" / subtype / f"{item_id}.jpg"
            dest.parent.mkdir(parents=True, exist_ok=True)
            img.save(dest, quality=94)
            gt_lines.append({"item_id": item_id, "text": text, "source": "pixarch-synthetic"})
            records.append(Record(
                item_id=item_id, path=str(dest.relative_to(raw_dir)),
                bucket="document", subtype=subtype,
                source_url="",
                license="CC0 -- generated by scripts/make_synthetic.py",
                attribution="Pixarch synthetic specimen",
                origin="synthetic",
                notes="fictional data; no real PII. Aadhaar starts 0/1 (never issued); "
                      "card numbers fail Luhn",
            ))
            made += 1
        print(f"  documents/{subtype:15} {made:3d}/{n}  (synthetic)")
    if gt_path is not None:
        _merge_gt(gt_path, gt_lines)
    return records
