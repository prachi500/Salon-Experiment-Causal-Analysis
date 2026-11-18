"""
Simulated data generator for the "Salon Experiment & Causal Analysis" project.

The internship brief describes a fictional Coimbatore salon & spa chain with
~300,000 rows across appointment, service, stylist and product-sales records
from 22 outlets. No download was attached, so this script builds a dataset with
exactly that structure (including the data-quality problems named in the brief):

  * no-shows are stored as status = "Cancelled"
  * service names differ from outlet to outlet
  * walk-ins have no client_id

Run:  python generate_data.py      (fixed seed -> fully reproducible)

If you have the official dataset, skip this script and put your files in data/
with the same column names (see DATA_DICTIONARY.md).
"""
import numpy as np
import pandas as pd
from pathlib import Path

SEED = 20261106
rng = np.random.default_rng(SEED)
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

START, END = pd.Timestamp("2025-01-01"), pd.Timestamp("2026-09-30")
LAUNCH = pd.Timestamp("2026-07-01")          # "last quarter's change" (Coimbatore only)

# ----------------------------------------------------------------------------
# 1. Outlets: 14 in Coimbatore (change launched), 8 in Tiruppur/Erode (control)
# ----------------------------------------------------------------------------
cbe_areas = ["RS Puram", "Gandhipuram", "Peelamedu", "Saibaba Colony", "Race Course",
             "Singanallur", "Vadavalli", "Ganapathy", "Kuniyamuthur", "Thudiyalur",
             "Saravanampatti", "Podanur", "Town Hall", "Avinashi Road"]
ctl_areas = [("Tiruppur", "Kumaran Road"), ("Tiruppur", "Avinashi Rd"), ("Tiruppur", "Pushpa Junction"),
             ("Tiruppur", "Palladam Road"), ("Tiruppur", "Khaderpet"), ("Erode", "Perundurai Road"),
             ("Erode", "Brough Road"), ("Erode", "Bhavani Road")]
fmt_cbe = ["Salon"] * 6 + ["Salon & Spa"] * 6 + ["Spa"] * 2
fmt_ctl = ["Salon"] * 3 + ["Salon & Spa"] * 3 + ["Spa"] * 2
rng.shuffle(fmt_cbe)
rng.shuffle(fmt_ctl)

rows = []
for i, a in enumerate(cbe_areas):
    rows.append(dict(outlet_id=f"O{i+1:02d}", city="Coimbatore", area=a, format=fmt_cbe[i],
                     change_launched="Y", treated=True,
                     base=rng.uniform(14, 24), price_index=rng.uniform(1.02, 1.15),
                     ns_base=rng.uniform(0.08, 0.14)))
for j, (c, a) in enumerate(ctl_areas):
    rows.append(dict(outlet_id=f"O{15+j:02d}", city=c, area=a, format=fmt_ctl[j],
                     change_launched="N", treated=False,
                     base=rng.uniform(10, 17), price_index=rng.uniform(0.88, 1.0),
                     ns_base=rng.uniform(0.08, 0.14)))
outlets = pd.DataFrame(rows)
outlets["seats"] = (outlets.base / 2.2).round().astype(int) + 2
outlets["open_year"] = rng.integers(2014, 2024, len(outlets))
(outlets.drop(columns=["treated", "base", "price_index", "ns_base"])
        .to_csv(DATA / "outlets.csv", index=False))

# ----------------------------------------------------------------------------
# 2. Stylists
# ----------------------------------------------------------------------------
LEVELS = ["Junior", "Senior", "Master"]
LEVEL_MULT = np.array([0.9, 1.0, 1.2])
st_rows, st_levels = [], {}
sid = 0
for _, o in outlets.iterrows():
    n_st = int(rng.integers(6, 12))
    lv = rng.choice(3, n_st, p=[0.35, 0.45, 0.20])
    ids = []
    for k in range(n_st):
        sid += 1
        ids.append(f"S{sid:03d}")
        st_rows.append(dict(stylist_id=f"S{sid:03d}", outlet_id=o.outlet_id, level=LEVELS[lv[k]],
                            tenure_months=int(rng.integers(3, 120))))
    st_levels[o.outlet_id] = (ids, lv)
pd.DataFrame(st_rows).to_csv(DATA / "stylists.csv", index=False)

# ----------------------------------------------------------------------------
# 3. Appointments
# ----------------------------------------------------------------------------
SERVICES = ["Haircut", "Threading", "Hair Colour", "Hair Spa", "Facial",
            "Mani-Pedi", "Waxing", "Bridal Makeup", "Massage", "Keratin Treatment"]
BASE_PRICE = np.array([400, 120, 1800, 1200, 1300, 900, 600, 9000, 1800, 4500])
W_FMT = {
    "Salon":       [.30, .12, .09, .08, .12, .10, .12, .006, .01, .02],
    "Salon & Spa": [.24, .09, .08, .09, .13, .10, .10, .005, .10, .02],
    "Spa":         [.06, .03, .04, .14, .18, .10, .06, .004, .38, .02],
}
FORMATS = list(W_FMT)
FESTIVE = {1: 1.0, 2: 1.0, 3: 1.05, 4: 1.25, 5: 1.3, 6: 1.0, 7: 1.0, 8: 1.12,
           9: 1.2, 10: 1.35, 11: 1.2, 12: 1.15}
TILT_IDX = [2, 4, 5, 7]          # colour, facial, mani-pedi, bridal respond to festive demand

# service-weights lookup  [format, month(1..12), walk-in(0/1)] -> cumulative probs
CUM = np.zeros((3, 13, 2, len(SERVICES)))
for fi, f in enumerate(FORMATS):
    for m in range(1, 13):
        for wk in (0, 1):
            w = np.array(W_FMT[f], dtype=float)
            w[TILT_IDX] *= FESTIVE[m]
            if wk:
                w[[0, 1]] *= 1.5
                w[7] *= 0.3
                w[9] *= 0.5
            CUM[fi, m, wk] = np.cumsum(w / w.sum())

# Raw (messy) service names: each outlet uses one of four naming styles
STYLES = [
    ["Haircut", "Threading", "Hair Colour", "Hair Spa", "Facial", "Mani-Pedi", "Waxing",
     "Bridal Makeup", "Massage", "Keratin Treatment"],
    ["Hair Cut", "Eyebrow Threading", "Hair Color", "Hair Spa Treatment", "Cleanup/Facial",
     "Manicure & Pedicure", "Full Body Wax", "Bridal Package", "Body Massage", "Hair Smoothening"],
    ["Cut & Style", "Brow Threading", "Global Colour", "Spa (Hair)", "Fruit Facial", "Mani/Pedi",
     "Waxing (Rica)", "Bride Makeup", "Aroma Massage", "Keratin"],
    ["HAIRCUT-W", "THREADING", "COLOR-GLOBAL", "HAIR SPA", "FACIAL-GOLD", "PEDICURE+MANICURE",
     "WAX-FULL", "BRIDAL MAKEUP", "MASSAGE-SWEDISH", "SMOOTHENING"],
]

DOW = np.array([0.85, 0.85, 0.90, 0.95, 1.10, 1.45, 1.30])
WALK = np.array([0.22, 0.22, 0.24, 0.25, 0.28, 0.33, 0.33])
MONTH = {1: .95, 2: .93, 3: 1.0, 4: 1.10, 5: 1.12, 6: .96, 7: .98, 8: 1.04, 9: 1.08,
         10: 1.20, 11: 1.10, 12: 1.08}

days = pd.date_range(START, END)
nd = len(days)
dow = days.dayofweek.values
mon = days.month.values
trend = 1 + 0.004 * ((days - START).days.values / 30.4)
post = (days >= LAUNCH)
POOL = 3600

frames, prod_frames = [], []
for oi, o in outlets.iterrows():
    treated = bool(o.treated)
    fi = FORMATS.index(o.format)
    lam = o.base * DOW[dow] * np.array([MONTH[m] for m in mon]) * trend * rng.lognormal(0, 0.08, nd)
    w = WALK[dow]
    tp = post & treated
    nb = rng.poisson(lam * (1 - w) * np.where(tp, 1.03, 1.0))
    nw = rng.poisson(lam * w * np.where(tp, 0.97, 1.0))
    d_idx = np.concatenate([np.repeat(np.arange(nd), nb), np.repeat(np.arange(nd), nw)])
    is_walk = np.concatenate([np.zeros(nb.sum(), int), np.ones(nw.sum(), int)])
    N = len(d_idx)
    m_row, tp_row = mon[d_idx], tp[d_idx]
    wkend = (dow[d_idx] >= 5)

    # services
    cs = CUM[fi, m_row, is_walk]
    svc = np.minimum((rng.random(N)[:, None] > cs).sum(1), len(SERVICES) - 1)

    # stylists & price
    ids, lv = st_levels[o.outlet_id]
    st = rng.integers(0, len(ids), N)
    price = np.round(BASE_PRICE[svc] * o.price_index * LEVEL_MULT[lv[st]] / 10) * 10

    # clients (booked only)
    wts = rng.lognormal(0, 0.9, POOL)
    cum = np.cumsum(wts)
    cidx = np.searchsorted(cum, rng.random(N) * cum[-1])
    cid = np.array([f"C{oi+1:02d}{c:05d}" for c in cidx], dtype=object)
    cid[is_walk == 1] = None

    # status
    p_ns = o.ns_base * np.where(wkend, 1.25, 1.0) * np.where(tp_row, 0.95, 1.0)
    p_cn = 0.06
    u = rng.random(N)
    booked = is_walk == 0
    no_show = booked & (u < p_ns)
    canc_notice = booked & (~no_show) & (u < p_ns + p_cn)
    status = np.where(no_show | canc_notice, "Cancelled", "Completed")
    lead = np.full(N, np.nan)
    lead[canc_notice] = np.round(rng.uniform(1, 72, canc_notice.sum()), 1)
    auto = no_show & (rng.random(N) < 0.30)               # some no-shows auto-closed after the slot
    lead[auto] = np.round(-rng.uniform(0.2, 4, auto.sum()), 1)
    completed = status == "Completed"

    # add-on (premium upsell). Loyalty + add-on push in treated outlets after launch, mainly for booked/ID clients
    spa_f = 0.5 if o.format == "Spa" else 1.0
    p_add = np.where(is_walk == 1, 0.10, 0.12)
    p_add = np.where(tp_row & booked, 0.12 + 0.14 * spa_f, p_add)
    addon = completed & (rng.random(N) < p_add)
    addon_price = np.round(350 * o.price_index / 10) * 10

    # raw names
    style = int(rng.integers(0, 4))
    raw = np.array(STYLES[style], dtype=object)[svc]
    alt = rng.random(N) < 0.10
    alt_style = rng.integers(0, 4, N)
    raw = np.where(alt, np.array([STYLES[s][k] for s, k in zip(alt_style, svc)], dtype=object), raw)
    r = rng.random(N)
    raw = np.where(r < 0.10, np.char.upper(raw.astype(str)), raw)
    raw = np.where((r >= 0.10) & (r < 0.16), np.char.lower(raw.astype(str)), raw)
    raw = np.where((r >= 0.16) & (r < 0.24), np.char.add(raw.astype(str), " "), raw)

    df = pd.DataFrame(dict(
        appt_date=days[d_idx].strftime("%Y-%m-%d"),
        appt_hour=rng.choice(np.arange(9, 21), N, p=np.array([.05, .08, .10, .10, .08, .08, .09, .10, .11, .11, .07, .03]) / 1.0),
        outlet_id=o.outlet_id, client_id=cid,
        booking_type=np.where(is_walk == 1, "Walk-in", "Booked"),
        service_raw=raw, stylist_id=np.array(ids, dtype=object)[st],
        status=status, cancel_lead_hours=lead,
        service_price=np.where(completed, price, np.nan),
        addon_price=np.where(completed, np.where(addon, addon_price, 0.0), np.nan),
    ))
    frames.append(df)

    # product sales linked to appointments
    attach = completed & (rng.random(N) < (0.14 + 0.02 * (tp_row & booked)))
    n_att = int(attach.sum())
    linked = pd.DataFrame(dict(
        sale_date=df.appt_date.values[attach], outlet_id=o.outlet_id,
        _row=np.where(attach)[0],
        amount=np.round(rng.lognormal(np.log(380 * o.price_index), 0.5, n_att) / 10) * 10))
    prod_frames.append(("linked", oi, linked))
    # standalone retail (no appointment)
    ns = rng.poisson(0.5 * np.array([MONTH[m] for m in mon]))
    sd = np.repeat(np.arange(nd), ns)
    standalone = pd.DataFrame(dict(
        sale_date=days[sd].strftime("%Y-%m-%d"), outlet_id=o.outlet_id, _row=-1,
        amount=np.round(rng.lognormal(np.log(420 * o.price_index), 0.55, len(sd)) / 10) * 10))
    prod_frames.append(("standalone", oi, standalone))

appt = pd.concat(frames, ignore_index=True)
appt = appt.sort_values(["appt_date", "outlet_id", "appt_hour"], kind="mergesort").reset_index(drop=True)
# row ids: need mapping from (outlet, local row) -> appt_id; rebuild via original order
orig = pd.concat(frames, ignore_index=True)
orig["_gidx"] = np.arange(len(orig))
order = orig.sort_values(["appt_date", "outlet_id", "appt_hour"], kind="mergesort")["_gidx"].values
rank = np.empty(len(order), int)
rank[order] = np.arange(len(order))
appt.insert(0, "appt_id", [f"A{i+1:07d}" for i in range(len(appt))])
offsets = np.cumsum([0] + [len(f) for f in frames])

prod_list = []
for kind, oi, p in prod_frames:
    if kind == "linked":
        g = offsets[oi] + p["_row"].values
        p = p.assign(appt_id=appt.appt_id.values[rank[g]])
    else:
        p = p.assign(appt_id=None)
    prod_list.append(p.drop(columns="_row"))
prod = pd.concat(prod_list, ignore_index=True).sort_values("sale_date", kind="mergesort").reset_index(drop=True)
prod.insert(0, "sale_id", [f"P{i+1:06d}" for i in range(len(prod))])
cats = np.array(["Shampoo", "Conditioner", "Serum", "Skincare", "Styling"])
prod["product_category"] = cats[rng.integers(0, 5, len(prod))]
prod = prod[["sale_id", "sale_date", "outlet_id", "appt_id", "product_category", "amount"]]

# exact duplicate rows (system double-posts) ~0.4%
dups = appt.sample(frac=0.004, random_state=SEED)
appt_out = pd.concat([appt, dups]).sort_values("appt_id", kind="mergesort").reset_index(drop=True)
appt_out.to_csv(DATA / "appointments.csv.gz", index=False)
prod.to_csv(DATA / "product_sales.csv.gz", index=False)

# ----------------------------------------------------------------------------
# 4. Experiment dataset: pilot of "WhatsApp reminder + one-tap reschedule"
#    (designed in Part A; each client randomised 50/50 at enrolment)
# ----------------------------------------------------------------------------
N_EXP = 14000
EXP_START, EXP_WEEKS = pd.Timestamp("2026-03-02"), 8
oid = outlets.outlet_id.values
w_o = (outlets.base / outlets.base.sum()).values
e_out = rng.choice(oid, N_EXP, p=w_o)
enrol = EXP_START + pd.to_timedelta(rng.integers(0, EXP_WEEKS * 7, N_EXP), unit="D")
idx_date = enrol + pd.to_timedelta(rng.integers(1, 8, N_EXP), unit="D")
arm = np.where(rng.random(N_EXP) < 0.5, "treatment", "control")
is_new = (rng.random(N_EXP) < 0.28).astype(int)
age = rng.choice(["18-24", "25-34", "35-44", "45+"], N_EXP, p=[.22, .38, .26, .14])
gender = rng.choice(["F", "M"], N_EXP, p=[.74, .26])
chan = rng.choice(["App", "Phone", "WhatsApp chat", "Website"], N_EXP, p=[.38, .27, .20, .15])
svc_cat = rng.choice(["Haircut", "Facial", "Hair Colour", "Hair Spa", "Mani-Pedi", "Waxing", "Massage", "Threading"],
                     N_EXP, p=[.26, .16, .10, .09, .10, .12, .09, .08])
weekend = (idx_date.dayofweek >= 5).astype(int)
fmt_map = outlets.set_index("outlet_id")["format"]
pidx_map = outlets.set_index("outlet_id")["price_index"]
pidx = pidx_map.loc[e_out].values

age_eff = pd.Series(age).map({"18-24": .35, "25-34": .10, "35-44": 0.0, "45+": -.25}).values
ch_eff = pd.Series(chan).map({"App": 0, "Phone": -.10, "WhatsApp chat": -.20, "Website": .15}).values
logit = -2.50 + 0.60 * is_new + age_eff + ch_eff + 0.20 * weekend + rng.normal(0, 0.55, N_EXP)
u_ns = 1 / (1 + np.exp(-logit))                          # latent no-show propensity

# pre-period covariates (returning clients only)
pre_vis = np.where(is_new == 1, 0, 1 + rng.poisson(2.2, N_EXP))
pre_ns = rng.binomial(pre_vis, u_ns)
pre_rate = np.where(pre_vis > 0, pre_ns / np.maximum(pre_vis, 1), np.nan)
pre_spend = np.where(is_new == 1, np.nan, np.round(rng.lognormal(np.log(950), 0.40, N_EXP) * pidx / 10) * 10)

# treatment effect: relative reduction in no-show probability (bigger for younger / new clients)
r_age = pd.Series(age).map({"18-24": .22, "25-34": .16, "35-44": .11, "45+": .06}).values
r = np.minimum(r_age * np.where(is_new == 1, 1.15, 1.0), 0.5)
v = rng.random(N_EXP)                                    # coupled potential outcomes
ns_ctrl = v < u_ns
ns_trt = v < u_ns * (1 - r)
is_t = arm == "treatment"
no_show = np.where(is_t, ns_trt, ns_ctrl).astype(int)
avoided = is_t & ns_ctrl & (~ns_trt)
w2 = rng.random(N_EXP)
resched_conv = avoided & (rng.random(N_EXP) < 0.5)
status = np.full(N_EXP, "completed", dtype=object)
status[(no_show == 0) & (w2 < 0.06)] = "cancelled_with_notice"
status[(no_show == 0) & (w2 >= 0.06) & (w2 < 0.072)] = "rescheduled"
status[resched_conv] = "rescheduled"
status[no_show == 1] = "no_show"
completed = status == "completed"

typical = 950
base_sp = np.where(np.isnan(pre_spend), typical * pidx, pre_spend)
rev = np.round(np.exp(0.55 * np.log(base_sp) + 0.45 * np.log(typical) + rng.normal(0, 0.45, N_EXP)) / 10) * 10
revenue = np.where(completed, rev, 0.0)

opt_out = (rng.random(N_EXP) < np.where(is_t, 0.011, 0.008)).astype(int)

p_ret = np.where(is_new == 1, 0.36, 0.66)
p_ret = p_ret + pd.Series(status).map({"no_show": -0.20, "cancelled_with_notice": -0.08,
                                       "rescheduled": 0.12, "completed": 0.0}).values
p_ret = p_ret - 0.15 * np.nan_to_num(pre_rate, nan=0.0)
retained = (rng.random(N_EXP) < np.clip(p_ret, 0.05, 0.95)).astype(int)

exp = pd.DataFrame(dict(
    client_id=[f"E{i+1:06d}" for i in range(N_EXP)], outlet_id=e_out,
    enrol_date=enrol.strftime("%Y-%m-%d"), index_appt_date=idx_date.strftime("%Y-%m-%d"),
    arm=arm, is_new_client=is_new, age_group=age, gender=gender, booking_channel=chan,
    service_category=svc_cat, weekend_appt=weekend,
    pre_visits_12m=pre_vis, pre_avg_spend_inr=pre_spend, pre_noshow_rate=np.round(pre_rate, 4),
    status=status, no_show=no_show, completed=completed.astype(int),
    revenue_inr=revenue, opted_out=opt_out, retained_60d=retained))
exp.to_csv(DATA / "experiment_clients.csv", index=False)

print("appointments rows:", len(appt_out), "| product rows:", len(prod),
      "| stylists:", len(st_rows), "| outlets:", len(outlets),
      "| experiment rows:", len(exp), "| total rows:", len(appt_out) + len(prod) + len(st_rows) + len(outlets))
print("exp no-show control/treatment:", exp.groupby("arm").no_show.mean().round(4).to_dict())
