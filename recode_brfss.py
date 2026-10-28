"""Recode BRFSS 2015 survey codes into meaningful values, driven by codebook_values.json.

Rules, applied per column in this order:
  1. "Don't know / Not sure / Refused / Missing" codes          -> NaN
  2. Hand-written decoders for packed encodings:
       ALCDAY5                       1xx days/week, 2xx days/30d     -> days per 30 days
       EXEROFT1/2, STRENGTH          1xx per week, 2xx per month     -> times per week
       FRUITJU1, FRUIT1, FVBEANS,    1xx/day, 2xx/week, 3xx/month    -> times per day
       FVGREEN, FVORANG, VEGETAB1
       BLDSUGAR, FEETCHK2            1xx/day ... 4xx/year            -> times per year
       EXERHMM1/2                    H:MM                            -> minutes
       WEIGHT2                       pounds or 9xxx kg               -> kg
       HEIGHT3                       F'II or 9xxx cm                 -> cm
       FLSHTMY2, HIVTSTD3            MMYYYY                          -> months before interview
       DROCDY3_, _DRNKWEK            stored x100                     -> /100
  3. Ordinal scales where "Never"/"None" sits at an out-of-order code (ORDINAL_FIXES)
  4. "None" / "Zero" / "Never" codes of count variables           -> 0
  5. Yes/No variables (either orientation)                         -> Yes = 1, No = 0

preprocess() then
  - adds UNSURE_CHRONIC: "don't know" to any chronic-condition question (from the raw codes),
  - merges HHADULT into NUMADULT (same question, asked of cell-phone vs landline respondents),
  - fills skip-logic NaNs (fill_skip_logic): a follow-up skipped because of the answer to its
    filter question gets the value that answer implies, e.g. BPMEDS = 0 without high blood pressure,
  - reorders codes whose numbers are not ordered by severity (REORDER) and folds PREDIAB1
    (told pre-diabetic) into DIABETE3 = 1 (borderline),
  - drops the administrative / survey-design columns in DROP_COLUMNS (after recoding, because the
    date decoders need IYEAR and IMONTH) and the redundant columns in REDUNDANT_COLUMNS.

select_columns() (NaN % per class, near-constant columns), fit_caps() (99th-percentile caps
of the continuous columns) and fit_impute() (median / mode fill values) are data-dependent:
fit them on the training set only and apply the result to the test set.

Usage:
  from recode_brfss import preprocess, select_columns, fit_caps, apply_caps, fit_impute, apply_impute
  x_tr, names = preprocess(x_tr, names)   # (n, p) float array with NaN for blanks, column names
  keep = select_columns(x_tr, y_tr, names)
  x_tr = x_tr[:, [names.index(n) for n in keep]]
  caps = fit_caps(x_tr, keep)
  x_tr = apply_caps(x_tr, keep, caps)
  fills = fit_impute(x_tr, keep)
  x_tr = apply_impute(x_tr, keep, fills)
  # test: preprocess -> same `keep` columns -> apply_caps(x_te, keep, caps)
  #       -> apply_impute(x_te, keep, fills)

  # Model override: compute on the RAW data, before preprocess(), and force those predictions to 1
  force = forced_positive(x_te_raw, names_raw)
  y_pred[force] = 1
"""
import json
import os
import re

import numpy as np

CODEBOOK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "codebook_values.json")

MISSING_RE = re.compile(r"^(don.?t\s*know|not\s*sure|refused|not asked|missing$)", re.I)
ZERO_RE = re.compile(r"^(none|zero|never|no drinks|did not drink)", re.I)

# Interview bookkeeping, phone-screening questions and survey weights: they describe how the
# interview was run, not the respondent.
DROP_COLUMNS = (
    "_STATE", "FMONTH", "IDATE", "IMONTH", "IDAY", "IYEAR", "DISPCODE", "SEQNO", "_PSU",
    "CTELENUM", "PVTRESD1", "COLGHOUS", "STATERES", "CELLFON3", "LADULT", "CTELNUM1",
    "CELLFON2", "CADULT", "PVTRESD2", "CCLGHOUS", "CSTATE", "LANDLINE",
    "QSTVER", "QSTLANG", "_STSTR", "_STRWT", "_RAWRAKE", "_WT2RAKE", "_CLLCPWT",
    "_DUALUSE", "_DUALCOR", "_LLCPWT",
    # Only asked on one of the two phone questionnaires; their missingness is a phone-type flag,
    # which adds nothing over age. HHADULT is merged into NUMADULT before dropping.
    "NUMMEN", "NUMWOMEN", "NUMHHOL2", "CPDEMO1", "MSCODE", "HHADULT",
    # Optional state modules without a filter question to fill from: blank simply because the
    # state did not run the module (63-84% NaN).
    "SCNTMNY1", "SCNTMEL1", "CAREGIV1", "CRGVEXPT", "SXORIENT", "TRNSGNDR",
    # About the respondent's child, not the respondent.
    "_CHISPNC", "_CRACE1", "_CPRACE", "RCSGENDR", "RCSRLTN2", "CASTHDX2",
    # CDC processing flags ("answers missing / out of range"), i.e. missing indicators.
    "_MISFRTN", "_MISVEGN", "_FRUITEX", "_VEGETEX", "PAMISS1_",
    # Activity type, ~75 nominal codes; the activity amount columns are kept instead.
    "EXRACT11", "EXRACT21",
    # Optional modules whose answers and missingness add < 1.5 pp heart-disease rate once age
    # (and diabetes) are accounted for. PREDIAB1 is merged into DIABETE3 before dropping.
    "BLDSTOOL", "HADSIGM3", "PDIABTST", "PREDIAB1",
    "EYEEXAM", "DIABEDU",                          # diabetes module, < 2 pp among diabetics
    "SCNTPAID", "SCNTWRK1", "SCNTLWK1",            # social context module, 84% NaN, < 4 pp
    # Cognitive-decline follow-ups: CDHOUSE, CDASSIST, CDSOCIAL are combined into CDSCORE;
    # CDHELP (< 1 pp) and CDDISCUS are dropped.
    "CDASSIST", "CDSOCIAL", "CDHELP", "CDDISCUS",
    # Label leak, not a feature: only asked of heart-attack survivors. Used by forced_positive()
    # to override the model's prediction instead.
    "HAREHAB1",
)

# Columns whose (recoded) value is fully determined by a column that is kept; checked on the
# training set, at most 0.5% of observed rows not recoverable (rounding of metric answers etc.).
# Raw survey answers replaced by a calculated variable:
RAW_REDUNDANT = (
    "HEIGHT3", "WEIGHT2",                          # HTM4, WTKG3
    "TOLDHI2",                                     # _RFCHOL
    "ASTHMA3", "ASTHNOW",                          # _ASTHMS1 (current / former / never)
    "HAVARTH3",                                    # _DRDXAR1
    "SMOKE100", "SMOKDAY2",                        # _SMOKER3
    "ALCDAY5",                                     # DROCDY3_
    "FRUITJU1", "FRUIT1", "FVBEANS", "FVGREEN", "FVORANG", "VEGETAB1",  # *DAY_ / *DA1_ per day
    "EXERANY2",                                    # _TOTINDA
    "EXERHMM1", "EXERHMM2", "EXEROFT1", "EXEROFT2", "STRENGTH",  # PADUR*_, PAFREQ*_, STRFREQ_
    "HIVTST6",                                     # _AIDTST3
)
# Calculated variables that are a coarsening or copy of another kept column:
DERIVED_REDUNDANT = (
    "HTIN4",                                       # HTM4
    "_BMI5CAT", "_RFBMI5",                         # _BMI5
    "_AGEG5YR", "_AGE65YR", "_AGE_G",              # _AGE80
    "_RFHLTH",                                     # GENHLTH
    "_HCVU651",                                    # HLTHPLN1 (restricted to age 18-64)
    "_RFHYPE5",                                    # BPHIGH4
    "_LTASTH1", "_CASTHM1",                        # _ASTHMS1
    "_RACEG21", "_RACEGR3", "_RACE_G1",            # _RACE
    "_CHLDCNT", "_EDUCAG", "_INCOMG",              # CHILDREN, EDUCA, INCOME2
    "_RFSMOK3",                                    # _SMOKER3
    "DRNKANY5",                                    # DROCDY3_
    "_FRTLT1", "_VEGLT1", "_FRT16", "_VEG23",      # _FRUTSUM, _VEGESUM, _FRUITEX, _VEGETEX
    "_FRTRESP", "_VEGRESP",                        # _MISFRTN, _MISVEGN
    "_PAINDX1", "_PA300R2", "_PA30021", "_PASTAE1",  # _PA150R2, _PACAT1, _PAREC1
    "_CHOLCHK",                                    # CHOLCHK (after the "never checked" fill)
    "MAXVO2_", "FC60_",                            # formula of age and sex (r = -0.97 with _AGE80)
    # Per-activity pieces and threshold flags of the physical-activity summary; kept are
    # _TOTINDA, PA1MIN_ (total minutes), PA1VIGM_ (vigorous minutes), STRFREQ_, _PACAT1.
    "METVL11_", "METVL21_", "ACTIN11_", "ACTIN21_", "PADUR1_", "PADUR2_", "PAFREQ1_",
    "PAFREQ2_", "_MINAC11", "_MINAC21", "PAMIN11_", "PAMIN21_", "PAVIG11_", "PAVIG21_",
    "_PA150R2", "_PASTRNG", "_PAREC1",
    "_RFSEAT2", "_RFSEAT3",                        # SEATBELT
    "_FLSHOT6", "_PNEUMO2",                        # FLUSHOT6, PNEUVAC3 (restricted to age 65+)
)
REDUNDANT_COLUMNS = RAW_REDUNDANT + DERIVED_REDUNDANT

# Codes whose numeric order is not an order of severity; applied after fill_skip_logic (which
# uses the original codes).
REORDER = {
    # 1 yes, 2 yes but only during pregnancy, 3 no, 4 borderline -> 0 no, 1 borderline / pregnancy only, 2 yes
    "BPHIGH4": {1: 2, 2: 1, 3: 0, 4: 1},
    "DIABETE3": {1: 2, 2: 1, 3: 0, 4: 1},
    # 1 one personal doctor, 2 more than one, 3 none -> 0 none, 1 one, 2 more than one
    "PERSDOC2": {1: 1, 2: 2, 3: 0},
}

# code -> new value, for categorical scales whose "never/none" answer is coded 8 or 98
ORDINAL_FIXES = {
    "CHECKUP1": {8: 5},        # 1 <1y ... 4 5+ years, 8 never   -> 5
    "EYEEXAM": {8: 5},         # 1 <1 month ... 4 2+ years, 8 never -> 5
    "ASNOSLEP": {8: 0},        # 1 one-two ... 5 more than ten, 8 none -> 0
    "ASTHMED3": {8: 0},        # 1 1-14 days ... 3 25-30 days, 8 never -> 0
    "ASINHALR": {8: 0},        # 1 1-4 times ... 6 100+ times, 8 never -> 0
    "SEATBELT": {8: np.nan},   # 8 never drive or ride in a car: not applicable
    "CHKHEMO3": {88: 0, 98: 0},  # 88 none, 98 never heard of the A1C test -> 0 tests
}


def _packed(v, units):
    """Decode 'prefix digit + two-digit amount' codes. units maps prefix -> factor, special
    maps exact codes -> value. Returns a new array; unknown codes become NaN."""
    out = np.full_like(v, np.nan)
    prefix = np.floor(v / 100)
    amount = v - prefix * 100
    for p, factor in units.items():
        m = prefix == p
        out[m] = amount[m] * factor
    return out


def _alcday(v):
    out = _packed(v, {1: 30 / 7, 2: 1})
    out[v == 888] = 0
    return out


def _per_week(v):
    out = _packed(v, {1: 1, 2: 7 / 30})
    out[v == 888] = 0
    return out


def _per_day(v):
    out = _packed(v, {1: 1, 2: 1 / 7, 3: 1 / 30})
    out[v == 300] = 0.5 / 30   # "less than once a month"
    out[v == 555] = 0          # never
    return out


def _per_year(v):
    out = _packed(v, {1: 365, 2: 52, 3: 12, 4: 1})
    out[v == 888] = 0          # never
    out[v == 555] = np.nan     # FEETCHK2: "no feet"
    return out


def _minutes(v):
    hours = np.floor(v / 100)
    return hours * 60 + (v - hours * 100)


def _weight(v):
    return np.where(v >= 9000, v - 9000, v * 0.45359237)


def _height(v):
    feet = np.floor(v / 100)
    return np.where(v >= 9000, v - 9000, (feet * 12 + (v - feet * 100)) * 2.54)


DECODERS = {
    "ALCDAY5": _alcday,
    "EXEROFT1": _per_week, "EXEROFT2": _per_week, "STRENGTH": _per_week,
    "FRUITJU1": _per_day, "FRUIT1": _per_day, "FVBEANS": _per_day,
    "FVGREEN": _per_day, "FVORANG": _per_day, "VEGETAB1": _per_day,
    "BLDSUGAR": _per_year, "FEETCHK2": _per_year,
    "EXERHMM1": _minutes, "EXERHMM2": _minutes,
    "WEIGHT2": _weight,
    "HEIGHT3": _height,
    "DROCDY3_": lambda v: v / 100, "_DRNKWEK": lambda v: v / 100,
}
DATE_COLUMNS = ("FLSHTMY2", "HIVTSTD3")


def _months_before_interview(v, iyear, imonth):
    month = np.floor(v / 10000)
    year = v - month * 10000
    month = np.where(month == 77, 6, month)                     # month unknown, year known
    year = np.where((year < 1900) | (year > 2100), np.nan, year)  # 7777/9999 and typos
    month = np.where(month > 12, np.nan, month)
    months = (iyear * 12 + imonth) - (year * 12 + month)
    return np.where(months < 0, np.nan, months)                 # dated after the interview


def load_codebook(path=CODEBOOK):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def recode(x, names, codebook=None):
    cb = load_codebook() if codebook is None else codebook
    x = x.copy()
    idx = {n: j for j, n in enumerate(names)}
    for j, name in enumerate(names):
        entry = cb.get(name)
        if not entry or not entry["values"]:
            continue
        scale = 10.0 ** -entry["implied_decimals"]
        col = x[:, j]
        rows = entry["values"]

        # 1. missing codes
        for lo, hi, label in rows:
            if MISSING_RE.search(label):
                col[(col >= lo * scale - 1e-9) & (col <= hi * scale + 1e-9)] = np.nan

        # 2. packed encodings
        if name in DECODERS:
            col[:] = DECODERS[name](col)
            continue
        if name in DATE_COLUMNS and "IYEAR" in idx and "IMONTH" in idx:
            col[:] = _months_before_interview(col, x[:, idx["IYEAR"]], x[:, idx["IMONTH"]])
            continue

        # 3. ordinal fixes
        if name in ORDINAL_FIXES:
            orig = col.copy()
            for code, new in ORDINAL_FIXES[name].items():
                col[orig == code] = new
            continue

        # 4. "none" codes of count variables (those with a numeric range row) -> 0
        if any(lo != hi for lo, hi, _ in rows):
            for lo, hi, label in rows:
                if lo == hi and lo != 0 and ZERO_RE.match(label):
                    col[col == lo * scale] = 0

        # 5. yes/no -> 1/0
        kept = [(lo, label.strip().lower()) for lo, hi, label in rows
                if lo == hi and not MISSING_RE.search(label)]
        if sorted(lab for _, lab in kept) == ["no", "yes"]:
            yes = next(lo for lo, lab in kept if lab == "yes")
            no = next(lo for lo, lab in kept if lab == "no")
            orig = col.copy()
            col[orig == yes] = 1
            col[orig == no] = 0
    return x


# Physical-activity variables computed from the first / second reported activity
PA_FIRST = ("METVL11_", "ACTIN11_", "PADUR1_", "PAFREQ1_", "_MINAC11", "PAMIN11_", "PAVIG11_",
            "PA1MIN_", "PA1VIGM_")
PA_SECOND = ("METVL21_", "ACTIN21_", "PADUR2_", "PAFREQ2_", "_MINAC21", "PAMIN21_", "PAVIG21_")
HIV_RECENT_MONTHS = 60
DIABETES_MODULE = ("INSULIN", "BLDSUGAR", "FEETCHK2", "DOCTDIAB", "CHKHEMO3", "FEETCHK", "DIABEYE")


# Fill NaNs implied by a filter question (no high BP -> BPMEDS 0); builds DIABYRS, HIV5YR, CDSCORE
def fill_skip_logic(x, names):
    x = x.copy()
    names = list(names)
    c = {n: x[:, j] for j, n in enumerate(names)}

    def fill(child, condition, value):
        col = c[child]
        m = np.isnan(col) & condition
        col[m] = value[m] if isinstance(value, np.ndarray) else value

    # 0 bad days of either kind -> 0 days poor health kept from usual activities
    fill("POORHLTH", (c["PHYSHLTH"] == 0) & (c["MENTHLTH"] == 0), 0)
    # no (or pregnancy-only / borderline) high blood pressure -> not on BP medication
    fill("BPMEDS", np.isin(c["BPHIGH4"], (2, 3, 4)), 0)
    # cholesterol never checked -> CHOLCHK "never" (5, after "5+ years ago"), not told it is high
    fill("CHOLCHK", c["BLOODCHO"] == 0, 5)
    fill("_RFCHOL", c["BLOODCHO"] == 0, 0)
    # men and women over 44 are not asked about pregnancy
    fill("PREGNANT", (c["SEX"] == 1) | (c["_AGE80"] > 44), 0)
    # LASTSMK2 becomes "distance from smoking": 0 current smoker, 1-7 time since quitting, 8 never
    fill("LASTSMK2", c["SMOKE100"] == 0, 8)
    fill("LASTSMK2", np.isin(c["SMOKDAY2"], (1, 2)), 0)
    # STOPSMK2 (tried to quit in the past year) is only asked of current smokers
    fill("STOPSMK2", (c["SMOKE100"] == 0) | (c["SMOKDAY2"] == 3), 0)
    # no alcohol in the past 30 days -> 0 drinks, 0 binges, 0 max drinks
    for child in ("AVEDRNK2", "DRNK3GE5", "MAXDRNKS"):
        fill(child, c["ALCDAY5"] == 0, 0)
    # no arthritis -> no arthritis-related limitation / pain; ARTHSOCL 3 = "not at all"
    no_arth = c["HAVARTH3"] == 0
    for child in ("LMTJOIN3", "ARTHDIS2", "JOINPAIN"):
        fill(child, no_arth, 0)
    fill("ARTHSOCL", no_arth, 3)
    # no exercise -> activity type 0 ("none") and all activity amounts 0;
    # EXRACT21 = 88 means "no other activity", so the second-activity amounts are 0 as well
    no_ex = c["EXERANY2"] == 0
    no_second = no_ex | (c["EXRACT21"] == 88)
    c["EXRACT21"][c["EXRACT21"] == 88] = 0
    fill("EXRACT11", no_ex, 0)
    fill("EXRACT21", no_ex, 0)
    for child in PA_FIRST:
        fill(child, no_ex, 0)
    for child in PA_SECOND:
        fill(child, no_second, 0)
    # no flu shot in the past 12 months -> 13 months ago (beyond the observed 0-12), place 0 ("none")
    fill("FLSHTMY2", c["FLUSHOT6"] == 0, 13)
    fill("IMFVPLAC", c["FLUSHOT6"] == 0, 0)
    # HIV test in the last 5 years (never tested -> 0), place 0 ("none")
    never_hiv = c["HIVTST6"] == 0
    hiv = c["HIVTSTD3"]
    hiv[:] = np.where(np.isnan(hiv), np.nan, (hiv <= HIV_RECENT_MONTHS).astype(float))
    fill("HIVTSTD3", never_hiv, 0)
    fill("WHRTST10", never_hiv, 0)
    names[names.index("HIVTSTD3")] = "HIV5YR"
    # diabetes module is only asked of diabetics: non-diabetics take no insulin, do not check
    # their blood sugar / feet, have no diabetes visits or A1C tests and no diabetic eye damage.
    # NaNs left are diabetics in states without the module: impute them from diabetics only.
    for child in DIABETES_MODULE:
        fill(child, np.isin(c["DIABETE3"], (2, 3, 4)), 0)
    # memory-loss question is only asked from age 45
    fill("CIMEMLOS", c["_AGE80"] < 45, 0)
    # CDHOUSE / CDASSIST / CDSOCIAL (1 always ... 5 never: memory loss interferes with housework,
    # needs assistance, limits social life) are only asked after memory loss. Combined into
    # CDSCORE = mean of (5 - code) over the answered items: 0 never / no memory loss ... 4 always.
    items = np.column_stack([5 - c[k] for k in ("CDHOUSE", "CDASSIST", "CDSOCIAL")])
    answered = ~np.isnan(items).all(axis=1)
    score = np.full(len(items), np.nan)
    score[answered] = np.nanmean(items[answered], axis=1)
    score[~answered & (c["CIMEMLOS"] == 0)] = 0
    c["CDHOUSE"][:] = score
    names[names.index("CDHOUSE")] = "CDSCORE"
    # age at diabetes diagnosis -> years with diabetes; non-diabetics have 0
    diab = c["DIABAGE2"]
    diab[:] = np.where(c["DIABETE3"] == 1, np.maximum(c["_AGE80"] - diab, 0), diab)
    fill("DIABAGE2", np.isin(c["DIABETE3"], (2, 3, 4)), 0)
    names[names.index("DIABAGE2")] = "DIABYRS"
    return x, names


# Chronic-condition questions whose "don't know" answers (raw code 7) feed UNSURE_CHRONIC
CHRONIC_CONDITIONS = ("CVDSTRK3", "CHCCOPD1", "CHCKIDNY", "CHCSCNCR", "CHCOCNCR", "ADDEPEV2",
                      "ASTHMA3", "HAVARTH3", "DIABETE3", "BPHIGH4", "TOLDHI2")


# 1 if any CHRONIC_CONDITIONS answer is "don't know" (raw code 7), on the raw data
def unsure_chronic(x, names):
    cols = [names.index(n) for n in CHRONIC_CONDITIONS if n in names]
    return np.any(x[:, cols] == 7, axis=1).astype(float)


# Row-wise preprocessing: UNSURE_CHRONIC, recode, skip-logic fills, REORDER, drop columns
def preprocess(x, names, codebook=None):
    flag = unsure_chronic(x, names)
    x = recode(x, names, codebook)
    x, names = np.column_stack([x, flag]), list(names) + ["UNSURE_CHRONIC"]
    if "NUMADULT" in names and "HHADULT" in names:
        j, k = names.index("NUMADULT"), names.index("HHADULT")
        x[:, j] = np.where(np.isnan(x[:, j]), x[:, k], x[:, j])
    x, names = fill_skip_logic(x, names)
    for name, mapping in REORDER.items():
        if name in names:
            col = x[:, names.index(name)]
            orig = col.copy()
            for code, new in mapping.items():
                col[orig == code] = new
    # non-diabetics told they have pre-diabetes (PREDIAB1 = 1) count as borderline in DIABETE3
    if "DIABETE3" in names and "PREDIAB1" in names:
        d = x[:, names.index("DIABETE3")]
        d[(d == 0) & (x[:, names.index("PREDIAB1")] == 1)] = 1
    drop = set(DROP_COLUMNS) | set(REDUNDANT_COLUMNS)
    keep = [j for j, n in enumerate(names) if n not in drop]
    return x[:, keep], [names[j] for j in keep]


# Rows forced to _MICHD = 1: HAREHAB1 (heart-attack rehab) is only asked of heart-attack survivors
def forced_positive(x, names):
    return ~np.isnan(x[:, names.index("HAREHAB1")])


# Kept columns: < max_nan % NaN or class NaN gap > min_gap pp, and not near-constant
def select_columns(x, y, names, max_nan=90.0, min_gap=5.0, max_top_share=99.0):
    nan = np.isnan(x)
    classes = np.unique(y)
    nan_pct = 100.0 * nan.mean(axis=0)
    gap = 100.0 * (nan[y == classes[-1]].mean(axis=0) - nan[y == classes[0]].mean(axis=0))
    keep = (nan_pct < max_nan) | (np.abs(gap) > min_gap)
    for j in range(x.shape[1]):
        v = x[~nan[:, j], j]
        if v.size == 0 or 100.0 * np.unique(v, return_counts=True)[1].max() / v.size >= max_top_share:
            keep[j] = False
    return [n for n, k in zip(names, keep) if k]


# q-th percentile caps of the continuous columns, zeros excluded (mostly-0 skip-logic fills)
def fit_caps(x, names, q=99.0, max_categories=15):
    caps = {}
    for j, name in enumerate(names):
        v = x[~np.isnan(x[:, j]), j]
        if v.size and (np.any(v != np.round(v)) or np.unique(v).size > max_categories):
            pos = v[v != 0]
            caps[name] = float(np.percentile(pos if pos.size else v, q))
    return caps


def apply_caps(x, names, caps):
    x = x.copy()
    for name, cap in caps.items():
        if name in names:
            j = names.index(name)
            x[:, j] = np.minimum(x[:, j], cap)   # NaN stays NaN
    return x


# Imputation subgroups: the column is filled with the median/mode of the rows where the
# condition holds, because those are the rows its NaNs come from.
IMPUTE_SUBGROUPS = {
    # diabetics in states that did not run the diabetes module (non-diabetics are already 0)
    **{name: ("DIABETE3", lambda v: v == 2) for name in DIABETES_MODULE},
    # people 45+ in states that did not run the cognitive module (under-45s are already 0)
    "CIMEMLOS": ("_AGE80", lambda v: v >= 45),
    "CDSCORE": ("_AGE80", lambda v: v >= 45),
}


# Fill values: mode for categoricals, median otherwise (no column is near-normal), module subgroups
def fit_impute(x, names, max_categories=15):
    fills = {}
    for j, name in enumerate(names):
        rows = ~np.isnan(x[:, j])
        if name in IMPUTE_SUBGROUPS and IMPUTE_SUBGROUPS[name][0] in names:
            by, cond = IMPUTE_SUBGROUPS[name]
            rows &= cond(x[:, names.index(by)])
        v = x[rows, j]
        if v.size == 0:
            continue
        values, counts = np.unique(v, return_counts=True)
        if np.all(v == np.round(v)) and values.size <= max_categories:
            fills[name] = float(values[counts.argmax()])
        else:
            fills[name] = float(np.median(v))
    return fills


def apply_impute(x, names, fills):
    x = x.copy()
    for name, value in fills.items():
        if name in names:
            col = x[:, names.index(name)]
            col[np.isnan(col)] = value
    return x
