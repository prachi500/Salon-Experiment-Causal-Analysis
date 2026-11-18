"""Builds the three Word deliverables from results.json (so every number matches the notebook)."""
import json
from pathlib import Path
import numpy as np
from scipy import stats
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from statsmodels.stats.power import NormalIndPower

ROOT = Path(".")
R = json.load(open(ROOT / "results.json"))
FIG = ROOT / "figures"
OUT = ROOT / "deliverables"
OUT.mkdir(exist_ok=True)

AUTHOR = "Soni Prachi Kishanbhai"
INTERN = "Data Science with Python Analyst Internship, Coimbatore (YuvaIntern)"
DATE = "6 November 2026"
NAVY, GREYTXT = RGBColor(0x1F, 0x3A, 0x5F), RGBColor(0x55, 0x5B, 0x66)

# ----------------------------------------------------------------- number helpers
pct = lambda x, d=1: f"{x*100:+.{d}f}%"
pp = lambda x, d=1: f"{x*100:+.{d}f} pp"
rate = lambda x, d=1: f"{x*100:.{d}f}%"
ci_pct = lambda lo, hi, d=1: f"{lo*100:+.{d}f}% to {hi*100:+.{d}f}%"
ci_pp = lambda lo, hi, d=1: f"{lo*100:+.{d}f} to {hi*100:+.{d}f} pp"
pv = lambda p: "<0.001" if p < 0.001 else f"{p:.3f}"
inr = lambda x: f"INR {x:,.0f}"

AB, DID, D, NV, PAR = R["ab"], R["did"], R["design"], R["naive"], R["parallel"]
NS = AB["no_show"]
rv, rb = AB["rev_per_completed_visit"], AB["rev_per_booked_appt"]
cb, ct = NV["Coimbatore"], NV["Control"]
N_ACTUAL = R["srm"]["n_control"] + R["srm"]["n_treatment"]
min_arm = min(R["srm"]["n_control"], R["srm"]["n_treatment"])
z = stats.norm.ppf(0.975) + stats.norm.ppf(0.8)
mde_actual = z * np.sqrt(2 * D["p0"] * (1 - D["p0"]) / min_arm)
seg = {s["segment"]: s for s in R["did_segments"]}
sab = {s["segment"]: s for s in R["ab_segments"]}
fam = lambda f_, s_: next(x for x in R["did_segments"] if x["family"] == f_ and x["segment"] == s_)
walk_vis = fam("Visits", "Walk-ins"); book_vis = fam("Visits", "Booked clients")
book_rpv = fam("Revenue per visit", "Booked clients"); walk_rpv = fam("Revenue per visit", "Walk-ins")
cup = R["cuped"]["table"]
assert R["ab_decision"]["decision"] == "GO" and R["did_decision"].startswith("GO"), "text below assumes GO / GO"

# ----------------------------------------------------------------- docx helpers
def new_doc(margin=2.0, base=10.5):
    d = Document()
    s = d.sections[0]
    s.page_width, s.page_height = Cm(21.0), Cm(29.7)
    s.left_margin = s.right_margin = Cm(margin)
    s.top_margin = s.bottom_margin = Cm(margin)
    st = d.styles["Normal"]; st.font.name = "Calibri"; st.font.size = Pt(base)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
    st.paragraph_format.space_after = Pt(5); st.paragraph_format.line_spacing = 1.08
    for name, size, before, after in [("Heading 1", 16, 16, 6), ("Heading 2", 12.5, 10, 4), ("Heading 3", 11, 8, 3)]:
        h = d.styles[name]; h.font.name = "Calibri"; h.font.size = Pt(size); h.font.bold = True; h.font.color.rgb = NAVY
        h.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri"); h.element.rPr.rFonts.set(qn("w:ascii"), "Calibri"); h.element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
        h.paragraph_format.space_before = Pt(before); h.paragraph_format.space_after = Pt(after); h.paragraph_format.keep_with_next = True
    d.core_properties.author = AUTHOR
    footer(d)
    return d

def footer(d):
    p = d.sections[0].footer.paragraphs[0]; p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(f"{AUTHOR} | Salon Experiment & Causal Analysis | Page "); r.font.size = Pt(8.5); r.font.color.rgb = GREYTXT
    r2 = p.add_run(); r2.font.size = Pt(8.5); r2.font.color.rgb = GREYTXT
    for t, txt in [("begin", None), (None, "PAGE"), ("end", None)]:
        if t:
            e = OxmlElement("w:fldChar"); e.set(qn("w:fldCharType"), t); r2._r.append(e)
        else:
            e = OxmlElement("w:instrText"); e.set(qn("xml:space"), "preserve"); e.text = txt; r2._r.append(e)

def para(d, text="", bold=False, italic=False, size=None, color=None, align=None, after=None, style=None):
    p = d.add_paragraph(style=style)
    rich(p, text, bold, italic, size, color)
    if align: p.alignment = align
    if after is not None: p.paragraph_format.space_after = Pt(after)
    return p

def rich(p, text, bold=False, italic=False, size=None, color=None):
    """**bold** segments inside text."""
    parts = text.split("**")
    for i, t in enumerate(parts):
        if not t: continue
        r = p.add_run(t); r.bold = bold or (i % 2 == 1); r.italic = italic
        if size: r.font.size = Pt(size)
        if color: r.font.color.rgb = color

def bullets(d, items, size=None):
    for t in items:
        p = d.add_paragraph(style="List Bullet"); rich(p, t, size=size); p.paragraph_format.space_after = Pt(2)

def numbered(d, items):
    for t in items:
        p = d.add_paragraph(style="List Number"); rich(p, t); p.paragraph_format.space_after = Pt(2)

def shade(cell, hexcolor):
    tcPr = cell._tc.get_or_add_tcPr(); shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), hexcolor); tcPr.append(shd)

def table(d, header, rows, widths, size=9, zebra=True, bold_first=False):
    t = d.add_table(rows=1, cols=len(header)); t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    for i, h in enumerate(header):
        c = t.rows[0].cells[i]; c.text = ""; shade(c, "1F3A5F")
        r = c.paragraphs[0].add_run(h); r.bold = True; r.font.size = Pt(size); r.font.color.rgb = RGBColor(255, 255, 255)
    for ri, row in enumerate(rows):
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            rich(cells[i].paragraphs[0], str(v), bold=(bold_first and i == 0), size=size)
            cells[i].paragraphs[0].paragraph_format.space_after = Pt(1)
            if zebra and ri % 2 == 1: shade(cells[i], "F1F5F9")
    for row in t.rows:
        for i, w in enumerate(widths):
            row.cells[i].width = Cm(w)
    d.add_paragraph().paragraph_format.space_after = Pt(2)
    return t

def figure(d, name, caption, width=15.5):
    p = d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(FIG / name), width=Cm(width))
    c = para(d, caption, italic=True, size=9, color=GREYTXT, align=WD_ALIGN_PARAGRAPH.CENTER, after=8)

def callout(d, text, fill="E8F0FE"):
    t = d.add_table(rows=1, cols=1); t.alignment = WD_TABLE_ALIGNMENT.CENTER; t.style = "Table Grid"
    c = t.rows[0].cells[0]; shade(c, fill); c.text = ""
    rich(c.paragraphs[0], text, size=10); c.width = Cm(17)
    d.add_paragraph().paragraph_format.space_after = Pt(2)

def title_block(d, title, subtitle):
    p = para(d, title, bold=True, size=24, color=NAVY, after=2)
    para(d, subtitle, size=12, color=GREYTXT, after=6)
    para(d, f"**Prepared by:** {AUTHOR}   |   **Programme:** {INTERN}   |   **Date:** {DATE}", size=9.5, color=GREYTXT, after=10)

# ================================================================= content blocks
def sec_data_note(d):
    callout(d, "**About the data.** The brief describes a fictional Coimbatore salon and spa chain (22 outlets, about 300,000 rows) but no data file "
               "came with it. I therefore generated a simulated dataset with exactly the described structure and problems "
               "(no-shows stored as cancellations, service names that differ by outlet, walk-ins without client ID) using a fixed random seed "
               "(`generate_data.py`). All numbers in this report come from that simulated data and would be re-computed by re-running the notebook on the real files.",
            fill="FFF4E5")

def sec_design(d, h="heading"):
    d.add_heading("Part A. Experiment design: WhatsApp reminder with one-tap reschedule", 1)
    d.add_heading("A1. Objective and hypotheses", 2)
    para(d, "The chain wants to test a new initiative: replace the SMS-only appointment reminder with a **WhatsApp reminder sent 24 hours and 3 hours before the slot, with a one-tap reschedule link**. "
            "No-shows are the most expensive leak in a salon, because the stylist's hour is lost and cannot be resold.")
    bullets(d, ["**H0:** the WhatsApp reminder does not change the no-show rate.",
                "**H1:** the WhatsApp reminder lowers the no-show rate. The test is two-sided (5% significance) so that we also notice if the change makes things worse."])
    d.add_heading("A2. Who is in the test and how they are assigned", 2)
    bullets(d, ["**Unit of randomisation:** the client, enrolled at their first eligible booking during the test window. One client is in one arm only, so a client never sees both reminders.",
                "**Assignment:** random 50/50 split at enrolment. Control keeps the existing SMS reminder.",
                "**Eligible:** clients with a client ID and a confirmed future booking. Walk-ins have no client ID, so they cannot receive a reminder or be tracked, and are out of scope.",
                "**Analysis principle:** intention-to-treat (everyone is analysed in the arm they were assigned to, whether or not the WhatsApp message was opened)."])
    d.add_heading("A3. Metrics", 2)
    para(d, f"Of the three candidate metrics, **no-show rate is the primary metric**. A reminder acts on it directly, the outcome is known within days of the appointment, and it has plenty of events "
            f"(baseline {rate(D['p0'])} of booked appointments). Client retention needs 60 days of waiting and revenue per visit is not what a reminder is supposed to change, so both are kept as guardrails.")
    table(d, ["Role", "Metric", "Definition", "Fail / success rule"], [
        ["Primary", "No-show rate", "Booked index appointments where the client did not show and did not cancel (status 'Cancelled' with no cancellation time or a time after the slot)", "Success if the 95% CI for (treatment minus control) is entirely below 0"],
        ["Guardrail", "Opt-out of messages", "Share of clients who opt out of reminders", "Fail if upper CI bound of the increase is above +1.0 pp"],
        ["Guardrail", "Revenue per completed visit", "Service + add-on + product revenue of completed visits", "Fail if lower CI bound of the change is below -3%"],
        ["Guardrail", "60-day retention", "Client returns for a booking within 60 days of the index appointment", "Fail if lower CI bound of the change is below -2 pp"],
        ["Guardrail", "Cancellations with notice", "Cancelled with a cancellation time before the slot", "Fail if upper CI bound of the increase is above +2 pp"],
        ["Secondary", "Rescheduled share", "Client used the reschedule link", "Descriptive: shows the mechanism"],
    ], [2.0, 3.3, 7.2, 4.5], size=8.5)
    d.add_heading("A4. Power analysis and sample size", 2)
    para(d, f"Baseline no-show rate from the last six months of booked appointments (Jan to Jun 2026): **{rate(D['p0'],2)}**. "
            f"I chose a minimum detectable effect (MDE) of **1.5 percentage points** (from {rate(D['p0'])} to {rate(D['p1'])}, a 12% relative reduction): smaller than that would not repay the cost of the WhatsApp messaging.")
    table(d, ["Input", "Value"], [
        ["Baseline no-show rate (p0)", rate(D["p0"], 2)], ["Target rate (p1)", rate(D["p1"], 2)], ["MDE (absolute)", "1.5 percentage points"],
        ["Significance level (two-sided)", "5%"], ["Power", "80%"], ["Allocation", "50 / 50"],
        ["Clients per arm (exact, two-proportion z-test)", f"{D['n_per_arm_exact']:,.0f}"], ["Closed-form cross-check", f"{D['n_per_arm_closed']:,.0f}"],
        ["Add 5% for clients with no usable outcome", f"{D['n_per_arm_adj']:,} per arm"], ["**Total clients to enrol**", f"**{D['n_total']:,}**"],
    ], [10, 7], size=9)
    para(d, "Sample size is very sensitive to the effect you want to detect: halving the MDE roughly quadruples the clients needed.", after=3)
    sens = D["sensitivity"]
    table(d, ["MDE (pp)", "Power 80%: clients per arm", "Power 90%: clients per arm"],
          [[f"{m:.1f}", f"{next(s['n per arm'] for s in sens if s['MDE (pp)']==m and s['power']==0.8):,}", f"{next(s['n per arm'] for s in sens if s['MDE (pp)']==m and s['power']==0.9):,}"] for m in [1.0, 1.5, 2.0, 2.5]],
          [4, 6.5, 6.5], size=9)
    figure(d, "fig_power.png", "Figure 1. Clients needed per arm for different effect sizes (includes the 5% allowance).", 12.5)
    d.add_heading("A5. Test duration", 2)
    et = {r["weeks"]: r["unique_clients_available"] for r in D["enrol_table"]}
    para(d, f"A client can only be enrolled once, so the number of **unique** booked clients available is what limits the duration, not the number of appointments. "
            f"Using the same calendar window of 2026 (starting 2 March), the chain would reach {et[7]:,} unique clients after 7 weeks, {et[8]:,} after 8 weeks and **{et[9]:,} after 9 weeks**, "
            f"which is the first whole week above the {D['n_total']:,} needed.")
    bullets(d, [f"**Enrolment: {D['weeks']} full weeks**, a whole number so that every weekday and the weekend appear equally often in both arms.",
                "**Follow-up:** the no-show outcome is known right after each index appointment. The retention guardrail needs a further 60 days, so the full read-out is about 18 weeks after the start.",
                "**Fixed horizon:** the primary result is read once, at the end. Guardrails (especially opt-outs) are monitored weekly, and the test is stopped early only for harm, never because the primary metric looks good early (repeated peeking inflates false positives)."])
    d.add_heading("A6. Analysis plan (fixed before looking at outcomes)", 2)
    numbered(d, ["Sample ratio mismatch: chi-square test of the actual split against 50/50, overall, by outlet (Bonferroni) and by enrolment week. Flag at p < 0.01 overall.",
                 "Randomisation balance: standardised mean differences on pre-experiment covariates (target |SMD| < 0.1) and an omnibus test.",
                 "Primary effect: difference in proportions with a 95% Wald confidence interval, cross-checked with a 2,000-resample bootstrap.",
                 "Guardrails and secondary metrics: the same, plus Welch t-tests for revenue.",
                 "Segments (pre-listed: age group, new vs returning, weekday vs weekend, booking channel): exploratory, Holm-corrected.",
                 "Variance reduction (stretch): regression adjustment / CUPED using pre-period behaviour.",
                 "Decision: GO if the primary CI is below zero and all guardrails pass; NO-GO if the estimate is not an improvement or any guardrail fails; EXTEND if the CI still includes both zero and the MDE (under-powered)."])

def sec_data(d):
    d.add_heading("1. Data and cleaning", 1)
    a, c = R["audit"], R["clean"]
    n_rec = c["no-shows recovered from 'Cancelled'"]
    para(d, f"The four tables hold **{a['total rows (4 tables)']:,} rows**: {a['appointment rows']:,} appointments, {a['product-sale rows']:,} product sales, {a['stylist rows']} stylists and {a['outlet rows']} outlets "
            f"(14 in Coimbatore, where the change launched, and 8 in Tiruppur and Erode, which did not). The data covers 1 Jan 2025 to 30 Sep 2026.")
    table(d, ["Problem found", "Evidence", "What I did"], [
        ["No-shows stored as 'Cancelled'", f"{rate(a['status = Cancelled (share)'])} of rows are Cancelled; {a['Cancelled with no cancellation time or after slot (hidden no-shows)']:,} of those have no cancellation time or one after the slot",
         f"Rebuilt a real outcome column: {n_rec:,} no-shows and {c['true cancellations with notice']:,} true cancellations. No-show rate among booked appointments is {rate(c['no-show rate among booked appointments'],2)}"],
        ["Service names differ by outlet", f"{a['distinct raw service names']} different spellings for {c['canonical services']} services (for example 'Hair Cut', 'Cut & Style', 'HAIRCUT-W')",
         "Lower-cased, removed punctuation, mapped with keyword rules; a check confirmed zero unmapped rows"],
        ["Walk-ins without client ID", f"{rate(a['walk-ins without client_id (share)'])} of visits",
         "Kept for visit-level revenue; excluded from anything per client (retention, experiment); analysed as their own segment"],
        ["Duplicate rows", f"{a['exact duplicate appointment rows']:,} exact duplicates", "Dropped"],
    ], [3.6, 6.4, 7.0], size=8.5)
    para(d, "Revenue per visit is the service price plus any add-on plus product sold in the same visit. Standalone retail sales (no appointment) are excluded from per-visit figures.")

def sec_ab(d):
    d.add_heading("Part B. Analysis of the experiment dataset", 1)
    s, b = R["srm"], R["balance"]
    para(d, f"The experiment dataset has **{N_ACTUAL:,} clients** ({s['n_control']:,} control, {s['n_treatment']:,} treatment), enrolled over 8 weeks from 2 March 2026. That is {N_ACTUAL/D['n_total']*100:.0f}% of the planned {D['n_total']:,}, "
            f"so the smallest effect it can reliably detect is about {mde_actual*100:.1f} pp instead of 1.5 pp.")
    d.add_heading("B1. Did randomisation work?", 2)
    table(d, ["Check", "Result", "Verdict"], [
        ["Sample ratio mismatch (overall)", f"{rate(s['share_treatment'],2)} in treatment; chi-square p = {s['p']:.2f}", "Pass (no mismatch)"],
        ["SRM by outlet (22 tests)", f"smallest p = {s['min_p_outlet']:.3f} vs Bonferroni threshold {s['bonferroni_threshold_outlet']:.4f}", "Pass (0 outlets flagged)"],
        ["SRM by enrolment week", f"smallest p = {s['min_p_week']:.3f}", "Pass"],
        ["Covariate balance (14 pre-experiment covariates)", f"largest |SMD| = {b['max_abs_smd']:.3f} (limit 0.1)", "Pass"],
        ["Omnibus test: can covariates predict the arm?", f"likelihood-ratio p = {b['omnibus_lr_p']:.2f}", "Pass"],
    ], [5.5, 8.0, 3.5], size=9)
    figure(d, "fig_balance.png", "Figure 2. Standardised mean differences between arms, all inside the +/-0.1 band.", 11.5)
    d.add_heading("B2. Effects with confidence intervals", 2)
    para(d, f"**Primary result.** The no-show rate fell from **{rate(NS['control'])} in control to {rate(NS['treatment'])} in treatment**, a difference of **{pp(NS['diff'],2)}** "
            f"(95% CI {ci_pp(NS['lo'], NS['hi'])}; bootstrap CI {ci_pp(NS['boot_lo'], NS['boot_hi'])}; p {pv(NS['p'])}). In relative terms, the risk of a no-show is {NS['rr']:.2f} times as high "
            f"(95% CI {NS['rr_lo']:.2f} to {NS['rr_hi']:.2f}), a {abs(1-NS['rr'])*100:.0f}% reduction. The whole interval is below zero and also below the 1.5 pp design effect, so the primary criterion is met.")
    rows = []
    for lab, k, kind in [("No-show rate (primary)", "no_show", "p"), ("Cancelled with notice", "cancelled_notice", "p"), ("Rescheduled", "rescheduled", "p"),
                         ("Opted out of messages", "opted_out", "p"), ("Retained within 60 days", "retained_60d", "p")]:
        a_ = AB[k]; rows.append([lab, rate(a_["control"], 2), rate(a_["treatment"], 2), pp(a_["diff"], 2), ci_pp(a_["lo"], a_["hi"], 2), pv(a_["p"])])
    rv = AB["rev_per_completed_visit"]; rb = AB["rev_per_booked_appt"]
    rows.append(["Revenue per completed visit", inr(rv["control"]), inr(rv["treatment"]), pct(rv["pct"]), ci_pct(rv["pct_lo"], rv["pct_hi"]), pv(rv["p"])])
    rows.append(["Revenue per booked appointment (incl. no-shows as 0)", inr(rb["control"]), inr(rb["treatment"]), pct(rb["pct"]), ci_pct(rb["pct_lo"], rb["pct_hi"]), pv(rb["p"])])
    table(d, ["Metric", "Control", "Treatment", "Difference", "95% CI", "p"], rows, [4.9, 2.2, 2.2, 2.4, 4.1, 1.2], size=8.5)
    figure(d, "fig_ab_effects.png", "Figure 3. Treatment minus control with 95% confidence intervals. The dashed line marks the -3% revenue guardrail.", 16)
    d.add_heading("B3. Guardrails", 2)
    g = R["ab_decision"]["guardrails"]
    table(d, ["Guardrail", "Estimate (95% CI)", "Rule", "Result"], [
        ["Opt-out of messages", f"{pp(AB['opted_out']['diff'],2)} ({ci_pp(AB['opted_out']['lo'], AB['opted_out']['hi'], 2)})", "upper bound below +1.0 pp", "PASS"],
        ["Revenue per completed visit", f"{pct(rv['pct'])} ({ci_pct(rv['pct_lo'], rv['pct_hi'])})", "lower bound above -3%", "PASS"],
        ["60-day retention", f"{pp(AB['retained_60d']['diff'],2)} ({ci_pp(AB['retained_60d']['lo'], AB['retained_60d']['hi'], 2)})", "lower bound above -2 pp", "PASS"],
        ["Cancellations with notice", f"{pp(AB['cancelled_notice']['diff'],2)} ({ci_pp(AB['cancelled_notice']['lo'], AB['cancelled_notice']['hi'], 2)})", "upper bound below +2 pp", "PASS"],
    ], [4.2, 6.3, 4.5, 2.0], size=9)
    assert all(g.values())
    para(d, f"The mechanism is visible in the data: the share of clients who **rescheduled** doubled ({rate(AB['rescheduled']['control'],1)} to {rate(AB['rescheduled']['treatment'],1)}), "
            "so part of the avoided no-shows became rebooked appointments instead of lost hours. Opt-outs rose slightly (about a quarter of a percentage point) but stay clearly inside the 1 pp limit. "
            "Revenue per completed visit is a guardrail, not a causal ticket-size estimate: it only includes clients who showed up, and the treatment changes who shows up.")
    callout(d, f"**Experiment verdict: GO.** The primary CI is entirely below zero, and all four guardrails pass. Retention is statistically unchanged "
               f"({pp(AB['retained_60d']['diff'],1)}, CI {ci_pp(AB['retained_60d']['lo'], AB['retained_60d']['hi'])}); it does not show a retention gain, so the business case rests on recovered slots, not on loyalty.")

def sec_did(d):
    d.add_heading("Part C. Impact of last quarter's change (no experiment): difference-in-differences", 1)
    d.add_heading("C1. Why a simple before/after is not enough", 2)
    para(d, f"Management compares Coimbatore's July to September 2026 quarter with April to June: revenue per completed visit moved from {inr(cb['rpv_q2'])} to {inr(cb['rpv_q3'])}, i.e. **{pct(cb['rpv_change'])}**. "
            f"That number cannot be trusted in either direction. In the control region, which did not launch anything, revenue per visit moved **{pct(ct['rpv_change'])}** over the same weeks. "
            f"Even in 2025, when nothing was launched anywhere, the same two quarters moved {pct(cb['rpv_change_2025_same_quarters'])} in Coimbatore and {pct(ct['rpv_change_2025_same_quarters'])} in the control region. "
            "Seasonal and random swings between quarters are therefore as large as the claimed gain, which is exactly why a control group is needed.")
    d.add_heading("C2. Method", 2)
    para(d, "I use a two-way fixed-effects difference-in-differences on a panel of 22 outlets and 89 weeks (77 before launch, 12 after; the launch week and the two partial edge weeks are removed). "
            "The model is `log(revenue per visit) = outlet effect + week effect + beta x (Coimbatore x after launch)`. Week effects absorb everything that hits both regions at once, including festive-season demand and general growth. "
            "Standard errors are clustered by outlet; with only 22 clusters I use t critical values (21 degrees of freedom) and add a stratified cluster bootstrap and a permutation test.")
    d.add_heading("C3. Results", 2)
    r_ = DID["rpv"]; rr = DID["rpv_rupees"]; vi = DID["visits"]; re_ = DID["revenue"]; nsd = DID["noshow"]; col = DID["rpv_collapsed"]
    table(d, ["Outcome", "Estimated effect", "95% CI", "p"], [
        ["**Revenue per visit (primary)**", f"**{pct(r_['pct'])}** (about {inr(rr['b'])} per visit)", f"{ci_pct(r_['pct_lo'], r_['pct_hi'])} ({inr(rr['lo'])} to {inr(rr['hi'])})", pv(r_["p"])],
        ["Completed visits per outlet-week", pct(vi["pct"]), ci_pct(vi["pct_lo"], vi["pct_hi"]), pv(vi["p"])],
        ["Total revenue per outlet-week", pct(re_["pct"]), ci_pct(re_["pct_lo"], re_["pct_hi"]), pv(re_["p"])],
        ["No-show rate (percentage points)", pp(nsd["b"], 2), ci_pp(nsd["lo"], nsd["hi"], 2), pv(nsd["p"])],
    ], [5.3, 4.7, 5.5, 1.5], size=9)
    para(d, f"**Reading.** After removing what the control region experienced, the change is associated with **{pct(r_['pct'])} higher revenue per visit** (95% CI {ci_pct(r_['pct_lo'], r_['pct_hi'])}). "
            f"The lower end of the interval is only just above zero, so the effect is real but modest and not precisely measured. There is no clear change in the number of visits ({pct(vi['pct'])}, CI {ci_pct(vi['pct_lo'], vi['pct_hi'])}), "
            f"and total revenue rises by {pct(re_['pct'])} (CI {ci_pct(re_['pct_lo'], re_['pct_hi'])}), whose lower bound touches zero.")
    para(d, f"**Robustness.** Weighting outlet-weeks by visits gives {pct(DID['rpv_weighted_pct'])}. The stratified cluster bootstrap (5,000 resamples of outlets) gives {pct(col['est_pct'])} with a CI of {ci_pct(col['boot_lo_pct'], col['boot_hi_pct'])}. "
            f"A permutation test (re-assigning the 'treated' label across outlets at random) gives p = {col['perm_p']:.3f}. All three agree with the main estimate.")
    figure(d, "fig_trends.png", "Figure 4. Revenue per completed visit (4-week average). Both regions move together before launch; Coimbatore pulls away afterwards.", 15.5)
    d.add_heading("C4. Is the parallel-trends assumption believable?", 2)
    para(d, "DiD is only valid if, without the change, Coimbatore would have moved like the control region. That cannot be proven, but it can be tested on the period before launch:")
    table(d, ["Check", "Result", "Reading"], [
        ["Differential pre-trend (treated x time, pre-launch data only)", f"{PAR['pretrend_slope_pct_per_10wk']*100:+.2f}% per 10 weeks (CI {ci_pct(PAR['pretrend_slope_lo'], PAR['pretrend_slope_hi'], 2)}); p = {PAR['pretrend_slope_p']:.2f}", "No drift apart before launch"],
        ["Placebo launch on 5 Jan 2026 (pre-launch data only)", f"{pct(PAR['placebo_pct'])} (CI {ci_pct(PAR['placebo_lo'], PAR['placebo_hi'])}); p = {PAR['placebo_p']:.2f}", "A fake launch finds no effect"],
        ["Joint test: five pre-launch quarters equal zero (event study)", f"F-test p = {PAR['joint_lead_p']:.2f}", "Cannot reject parallel trends"],
        ["Monthly event study leads", f"{PAR['pre_months_ci_excl_zero']} of {PAR['pre_months_total']} pre-launch months have a CI excluding zero (about 1 expected by chance); largest |lead| = {PAR['event_pre_max_abs_pct']*100:.1f}%", "Consistent with noise"],
    ], [6.0, 7.5, 3.5], size=8.5)
    figure(d, "fig_event_study.png", "Figure 5. Event study: Coimbatore minus control by month, relative to the average of all pre-launch months. Shaded area is the 95% CI.", 15)
    para(d, "Passing these tests makes the assumption plausible, not certain. They cannot detect a Coimbatore-specific shock that happened at the same time as the launch.")

def sec_seg(d):
    d.add_heading("Part D. Who gained and who lost?", 1)
    d.add_heading("D1. Last quarter's change, by segment", 2)
    rows = []
    for s in R["did_segments"]:
        rows.append([s["family"], s["segment"], pct(s["pct"]), ci_pct(s["lo"], s["hi"]), pv(s["p"]), pv(s["p_holm"]), f"{s['n_treated_outlets']} / {s['G']-s['n_treated_outlets']}"])
    table(d, ["Outcome", "Segment", "Effect", "95% CI", "p", "p (Holm)", "Outlets treated / control"], rows, [2.6, 4.2, 1.7, 3.3, 1.3, 1.6, 2.3], size=8)
    figure(d, "fig_did_segments.png", "Figure 6. Segment effects of the quarterly change. Blue: CI above zero. Orange: CI below zero. Grey: cannot tell.", 16)
    bullets(d, [f"**Gainers:** booked (ID) clients: revenue per visit {pct(book_rpv['pct'])} and visits {pct(book_vis['pct'])}. Hair services ({pct(fam('Revenue per visit','Service: Hair')['pct'])}) and skin and beauty services ({pct(fam('Revenue per visit','Service: Skin & beauty')['pct'])}) both gained, consistent with the loyalty points and premium add-on menu targeting registered clients.",
                f"**Possible loser:** walk-ins. Their visits fell {pct(walk_vis['pct'])} (CI {ci_pct(walk_vis['lo'], walk_vis['hi'])}) and revenue per visit changed {pct(walk_rpv['pct'])} (CI {ci_pct(walk_rpv['lo'], walk_rpv['hi'])}). The visit drop is only borderline and does not survive the Holm correction (p = {pv(walk_vis['p_holm'])}), so treat it as a warning to watch, not a finding.",
                "**Cannot tell:** spa and bridal services and individual outlet formats. They have wide intervals because few outlets (as few as 2 treated and 2 control Spa outlets) support each estimate."])
    d.add_heading("D2. WhatsApp reminder experiment, by segment (effect on no-show rate)", 2)
    rows = [[s["segment"], f"{s['n']:,}", rate(s["control"]), rate(s["treatment"]), pp(s["diff"]), ci_pp(s["lo"], s["hi"]), pv(s["p_holm"])] for s in R["ab_segments"]]
    table(d, ["Segment", "Clients", "Control", "Treatment", "Difference", "95% CI", "p (Holm)"], rows, [3.8, 1.6, 1.8, 2.0, 2.2, 3.7, 1.9], size=8)
    figure(d, "fig_ab_segments.png", "Figure 7. Change in no-show rate by segment (treatment minus control).", 13.5)
    para(d, f"Every segment moves in the same direction. The effect is largest for clients aged 18 to 24 ({pp(sab['Age 18-24']['diff'])}) and smallest for clients aged 45 and above ({pp(sab['Age 45+']['diff'])}), "
            f"and larger for returning than for new clients. The formal test of whether the effect differs across age groups is not significant (likelihood-ratio p = {R['ab_age_interaction_p']:.2f}), "
            "so the pattern is suggestive and should not be used to target the roll-out by age.")

def sec_cuped(d):
    d.add_heading("Part E (stretch). CUPED and regression adjustment", 1)
    para(d, f"CUPED removes the part of the outcome that is predictable from pre-experiment behaviour: `Y_adj = Y - theta x (X - mean of X)`, with `theta = cov(Y, X) / var(X)`. "
            f"For revenue per booked appointment, X is the client's average spend before the experiment (theta = {R['cuped']['theta_revenue']:.2f}, correlation {R['cuped']['rho_revenue']:.2f}). "
            "For the no-show and retention metrics I use regression adjustment with pre-period no-show rate, pre-period visits and spend, new-client flag, age group, booking channel, weekend and outlet. "
            "Sample size is proportional to variance, so the weeks needed shrink by the same share.")
    rows = [[c["metric"], c["method"], f"{c['se_raw']*100 if c['metric'].startswith(('No','60')) else c['se_raw']:.3f}" + (" pp" if c["metric"].startswith(("No", "60")) else ""),
             f"{c['se_adj']*100 if c['metric'].startswith(('No','60')) else c['se_adj']:.3f}" + (" pp" if c["metric"].startswith(("No", "60")) else ""),
             f"{c['variance_reduction']*100:.1f}%", f"{c['weeks_needed']:.1f}", f"{c['weeks_saved']:.1f}"] for c in cup]
    table(d, ["Metric", "Method", "SE before", "SE after", "Variance cut", "Weeks needed (plan 9)", "Weeks saved"], rows, [4.3, 3.5, 1.9, 1.9, 1.7, 2.1, 1.6], size=8)
    figure(d, "fig_cuped.png", "Figure 8. Weeks of enrolment needed with and without adjustment.", 13.5)
    para(d, f"**Honest conclusion:** the gain is modest. Adjustment saves about {cup[0]['weeks_saved']:.1f} weeks on the primary no-show metric (a client's past behaviour predicts only 1% of the variance in a yes/no outcome) "
            f"and about {cup[1]['weeks_saved']:.1f} weeks on revenue per booked appointment, where CUPED lowers the standard error by about {cup[1]['variance_reduction']*100:.0f}% in variance terms. "
            "CUPED helps most for continuous metrics that are stable per client (spend), and little for rare binary events. The point estimates barely move, which is what a valid adjustment should do.")

def sec_wrong(d):
    d.add_heading("What could make my causal estimates wrong", 1)
    d.add_heading("Difference-in-differences (last quarter's change)", 2)
    bullets(d, ["**Parallel trends could fail after launch.** A Coimbatore-only event in July to September (a festival, a competitor, a local price change) would be credited to the change. The pre-launch tests only show the regions moved together before.",
                "**Seasonality may hit the regions differently.** If festive demand lifts Coimbatore more than Tiruppur and Erode, the control undercorrects. The 2025 comparison and event-study leads are my safeguard.",
                "**Spillovers.** Clients who travel between regions or shared advertising would contaminate the control and bias the effect toward zero.",
                "**Only 22 outlets and 12 post weeks.** Standard errors can be too small, so I used t critical values, a bootstrap and a permutation test; the lower bound of the interval is still close to zero.",
                "**Composition change.** If the add-on menu changes who visits (more registered clients, fewer walk-ins), revenue per visit can rise without any individual spending more. The visit counts and segment results are the check, and they do show fewer walk-ins.",
                "**Outlets were not randomly selected.** Coimbatore outlets are larger and pricier. Outlet effects remove level differences but not different trends."])
    d.add_heading("A/B test (WhatsApp reminder)", 2)
    bullets(d, ["**Broken randomisation** would invalidate the comparison; SRM and balance checks both passed.",
                "**Non-delivery.** Some clients do not receive or open WhatsApp. The estimate is intention-to-treat, so it understates the effect on clients who do receive it.",
                "**Interference.** Friends or family booking together may see each other's reminders.",
                "**Novelty.** Reminders may work best at first; eight weeks cannot show whether the effect fades.",
                "**Selection in revenue per completed visit.** It only includes clients who showed up, and the treatment changes who shows up.",
                "**Multiple comparisons.** Segment results are exploratory; I report Holm-adjusted p-values.",
                f"**Under-enrolment.** The pilot has {N_ACTUAL/D['n_total']*100:.0f}% of the planned sample, so small effects below about {mde_actual*100:.1f} pp are not reliably detectable."])

def sec_recommend(d):
    d.add_heading("Recommendation", 1)
    table(d, ["Decision", "Verdict", "Why"], [
        ["Roll out the WhatsApp reminder with one-tap reschedule", "**GO**", f"No-show rate {pp(NS['diff'],1)} (CI {ci_pp(NS['lo'], NS['hi'])}); all four guardrails pass; the effect shows up in every segment. Roll out in stages and keep monitoring opt-outs."],
        ["Keep last quarter's loyalty and add-on change in Coimbatore", "**GO, with monitoring**", f"Revenue per visit {pct(DID['rpv']['pct'])} against the control (CI {ci_pct(DID['rpv']['pct_lo'], DID['rpv']['pct_hi'])}). Real, but about {DID['rpv']['pct']*100:.0f}% and not precisely measured. Re-measure after the October to December festive quarter and watch walk-in visits."],
    ], [5.6, 3.0, 8.4], size=9)

def sec_deliverables(d):
    d.add_heading("Deliverables and reproducibility", 1)
    table(d, ["File", "What it is"], [
        ["Salon_Project_Report.docx", "This report (for submission)"],
        ["Experiment_Design_Document.docx", "Part A as a stand-alone document"],
        ["Decision_Memo.docx", "One-page decision memo"],
        ["Salon_Experiment_Causal_Analysis.ipynb", "Executed analysis notebook (Python), with all outputs"],
        ["generate_data.py", "Builds the simulated dataset with a fixed seed"],
        ["results.json, figures/", "Every number and chart used in the documents"],
    ], [7.0, 10.0], size=9)
    para(d, "To reproduce: `pip install -r requirements.txt`, then `python generate_data.py`, `python build_notebook.py` and `python build_docs.py`.")

def memo(d):
    para(d, "DECISION MEMO", bold=True, size=18, color=NAVY, after=1)
    para(d, "**To:** Operations Head, Coimbatore salon & spa chain    **From:** " + AUTHOR + "    **Date:** " + DATE, size=9.5, color=GREYTXT, after=1)
    para(d, "**Subject:** WhatsApp reminder test and the true impact of last quarter's change", size=9.5, color=GREYTXT, after=6)
    callout(d, f"**Recommendation.** (1) **GO**: roll out the WhatsApp reminder with one-tap reschedule to all 22 outlets in stages. "
               f"(2) **GO with monitoring**: keep last quarter's loyalty and add-on change, but plan on about **{pct(DID['rpv']['pct'],0)} revenue per visit, not the headline before/after figure**, and re-measure after the festive quarter.")
    d.add_heading("1. WhatsApp reminder (randomised test, 14,000 clients)", 2)
    bullets(d, [f"No-show rate fell from **{rate(NS['control'])} to {rate(NS['treatment'])}** ({pp(NS['diff'])}, 95% CI {ci_pp(NS['lo'], NS['hi'])}). That is {abs(1-NS['rr'])*100:.0f}% fewer no-shows.",
                f"Guardrails pass: opt-outs {pp(AB['opted_out']['diff'],2)} (limit +1.0), revenue per completed visit {pct(rv['pct'])}, 60-day retention {pp(AB['retained_60d']['diff'])} (not significant), cancellations {pp(AB['cancelled_notice']['diff'])}.",
                f"Randomisation is clean (split {rate(R['srm']['share_treatment'])}, SRM p = {R['srm']['p']:.2f}, largest balance gap {R['balance']['max_abs_smd']:.2f}). Gains appear in every age group, largest for ages 18 to 24 ({pp(sab['Age 18-24']['diff'])})."], size=9.5)
    d.add_heading("2. Last quarter's change (no experiment: difference-in-differences vs Tiruppur and Erode)", 2)
    bullets(d, [f"A simple before/after is misleading: Coimbatore revenue per visit {pct(cb['rpv_change'])} vs Q2, but the control region moved {pct(ct['rpv_change'])} with no change, so quarter-to-quarter swings are as big as the claim.",
                f"Controlling for that, the change added **{pct(DID['rpv']['pct'])}** revenue per visit (95% CI {ci_pct(DID['rpv']['pct_lo'], DID['rpv']['pct_hi'])}, about {inr(DID['rpv_rupees']['b'])} per visit). Pre-launch trends were parallel and a placebo launch found nothing.",
                f"Gains come from booked clients (visits {pct(book_vis['pct'])}). Walk-in visits fell {pct(walk_vis['pct'])} (CI {ci_pct(walk_vis['lo'], walk_vis['hi'])}), which is borderline; watch it."], size=9.5)
    table(d, ["Question", "Estimate", "95% CI", "Verdict"], [
        ["WhatsApp: no-show rate", pp(NS["diff"], 2), ci_pp(NS["lo"], NS["hi"]), "GO"],
        ["Quarterly change: revenue per visit", pct(DID["rpv"]["pct"]), ci_pct(DID["rpv"]["pct_lo"], DID["rpv"]["pct_hi"]), "GO, monitor"],
        ["Quarterly change: visits", pct(DID["visits"]["pct"]), ci_pct(DID["visits"]["pct_lo"], DID["visits"]["pct_hi"]), "No clear effect"],
    ], [6.2, 2.6, 5.2, 3.0], size=8.5)
    d.add_heading("3. What could make this wrong", 2)
    bullets(d, ["A Coimbatore-only event in July to September would be credited to the change; only 22 outlets and 12 post-launch weeks support the estimate.",
                "WhatsApp: results are intention-to-treat (some clients never see the message), eight weeks cannot show novelty fading, and the pilot reached 93% of the planned sample.",
                "Data are simulated to the brief's description; re-run the notebook on the real files before acting."], size=9.5)
    d.add_heading("4. Next steps", 2)
    bullets(d, ["Roll out WhatsApp reminders in three waves; track opt-outs weekly and stop a wave if opt-outs rise more than 1 pp.",
                "Re-run the difference-in-differences on October to December data; consider a randomised holdout for the next loyalty change.",
                "Investigate walk-in experience at loyalty outlets."], size=9.5)

# ================================================================= assemble documents
# 1) full report
d = new_doc()
title_block(d, "Salon Experiment & Causal Analysis", "Designing an A/B test and measuring the true impact of a business change")
d.add_heading("Executive summary", 1)
para(d, "A Coimbatore salon and spa chain launched a change last quarter and says it improved results, but nobody checked whether the improvement was real or just seasonal. "
        "It also wants to test a new reminder initiative properly. This project answers both questions.")
callout(d, f"**1. WhatsApp reminder test: GO.** The no-show rate fell from {rate(NS['control'])} to {rate(NS['treatment'])} ({pp(NS['diff'],2)}, 95% CI {ci_pp(NS['lo'], NS['hi'])}). "
           f"All four guardrails pass. The experiment was valid: no sample ratio mismatch (p = {R['srm']['p']:.2f}) and balanced groups.\n"
           f"**2. Last quarter's change: real but modest, GO with monitoring.** Against the Tiruppur and Erode control region, revenue per visit rose **{pct(DID['rpv']['pct'])}** "
           f"(95% CI {ci_pct(DID['rpv']['pct_lo'], DID['rpv']['pct_hi'])}). A naive before/after shows {pct(cb['rpv_change'])} while the control region moved {pct(ct['rpv_change'])}, so it is not a reliable guide.")
bullets(d, [f"**Who gained:** booked clients (revenue per visit {pct(book_rpv['pct'])}, visits {pct(book_vis['pct'])}). **Possible loser:** walk-ins (visits {pct(walk_vis['pct'])}, borderline and not significant after correction).",
            f"**Stretch (CUPED):** the test could have been about {cup[1]['weeks_saved']:.1f} weeks shorter for revenue and only {cup[0]['weeks_saved']:.1f} weeks for the no-show metric. The gain is real but modest.",
            "**Biggest risk to the causal claim:** a Coimbatore-specific shock in the same months, which no pre-launch test can rule out."])
sec_data_note(d)
sec_data(d)
sec_design(d)
sec_ab(d)
sec_did(d)
sec_seg(d)
sec_cuped(d)
sec_wrong(d)
sec_recommend(d)
d.add_page_break()
memo(d)
d.add_page_break()
sec_deliverables(d)
d.save(OUT / "Salon_Project_Report.docx")

# 2) design document
d = new_doc()
title_block(d, "Experiment Design Document", "A/B test of a WhatsApp reminder with one-tap reschedule")
sec_design(d)
d.save(OUT / "Experiment_Design_Document.docx")

# 3) one-page memo
d = new_doc(margin=1.6, base=10)
memo(d)
d.save(OUT / "Decision_Memo.docx")
print("saved", [p.name for p in OUT.iterdir()])
