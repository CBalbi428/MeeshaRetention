# retention_app.py
import io
import re
from datetime import datetime, timezone
from typing import Optional, Tuple

import pandas as pd
import streamlit as st

# ====== CONFIG ======
APP_TITLE = "Meesha Retention Builder"
DATE_TODAY = datetime.now(timezone.utc).astimezone().date()  # local date
DEFAULT_TEMPLATE = (
    "Hey {first_name}, it’s been {human_time} since your last {service} "
    "and we miss you! 💖 Want me to help you snag your next spot or answer any questions?"
)

SERVICE_FLAGS = [
    # (regex, label, threshold_days)
    (r"(tox|botox|xeomin|dysport|jeuveau)", "Tox cadence ~90 days", 90),
    (r"(microneedling|vamp|prf|rf)", "Microneedling/PRF cadence 28–42 days", 42),
    (r"(laser|resurfacing|bbl|moxi|ipl)", "Laser cadence 28–56 days", 56),
    (r"(filler|sculptra|radiesse|bellafill)", "Filler cadence 6–12 months", 180),
]

EXPECTED_COLS = [
    "Location", "Provider", "Resource", "Patient name", "Age", "Gender",
    "Patient's Phone", "Patient's Email", "Creation Date", "Start", "End",
    "Duration", "Type", "Purpose", "Status", "Outcome Status"
]

# ====== PAGE UI ======
st.set_page_config(page_title=APP_TITLE, page_icon="💖", layout="centered")
st.title(APP_TITLE)
st.caption(
    "Upload two EMR exports: **Past 3 Months** and **Upcoming**. "
    "We’ll generate a list of clients with **no future appointment** plus a "
    "pre-written text message for easy outreach."
)

with st.expander("Message template (optional)"):
    tmpl = st.text_area(
        "Edit the message template",
        value=DEFAULT_TEMPLATE,
        height=90,
        help="Fields: {first_name}, {last_name}, {full_name}, {service}, {days_since}, {human_time}, {last_visit_date}",
    )

# ====== HELPERS ======
def load_emr_like(file) -> pd.DataFrame:
    """Load CSV or Excel and force headers to come from Excel row 3 (index 2)."""
    name = getattr(file, "name", "upload")
    # Load with no headers
    if name.lower().endswith((".xlsx", ".xls")):
        raw = pd.read_excel(file, header=None)
    else:
        raw = pd.read_csv(file, header=None)

    header_index = 2  # Excel row 3
    if len(raw) <= header_index:
        raise ValueError("File is too short to contain headers at Excel row 3.")

    headers = raw.iloc[header_index].ffill().tolist()
    data = raw.iloc[header_index + 1 :].copy()
    data.columns = headers

    # Keep only expected columns if present; otherwise keep all
    cols = [c for c in EXPECTED_COLS if c in data.columns]
    if cols:
        data = data[cols].copy()

    # Drop fully empty rows
    data = data.dropna(how="all").reset_index(drop=True)
    return data

def normalize_phone(s: str) -> str:
    if pd.isna(s):
        return ""
    digits = re.sub(r"\D", "", str(s))
    if len(digits) >= 10:
        digits = digits[-10:]  # last 10 digits (US)
        return f"({digits[0:3]}) {digits[3:6]}-{digits[6:10]}"
    return ""

def split_name(full: str) -> Tuple[str, str]:
    if pd.isna(full):
        return ("", "")
    s = str(full).strip()
    if "," in s:  # "Last, First"
        last, first = [x.strip() for x in s.split(",", 1)]
        return (first, last)
    parts = s.split()  # "First Last"
    if len(parts) >= 2:
        return (parts[0], parts[-1])
    return (s, "")

def parse_datetime_maybe(s: str) -> Optional[datetime]:
    if pd.isna(s) or str(s).strip() == "":
        return None
    try:
        return pd.to_datetime(s, errors="coerce").to_pydatetime()
    except Exception:
        return None

def best_service(row: pd.Series) -> str:
    purp = str(row.get("Purpose", "") or "").strip()
    typ = str(row.get("Type", "") or "").strip()
    return purp if purp else typ

def client_key(row: pd.Series) -> str:
    phone = normalize_phone(row.get("Patient's Phone", ""))
    email = str(row.get("Patient's Email", "") or "").strip().lower()
    first, last = split_name(row.get("Patient name", ""))
    if phone:
        return f"p:{re.sub(r'\\D','', phone)}"
    if email:
        return f"e:{email}"
    if first or last:
        return f"n:{(first+' '+last).strip().upper()}"
    return ""

def humanize_days(days: int) -> str:
    if days < 14:
        return f"about {days} days"
    weeks = round(days / 7)
    if days < 60:
        return f"about {weeks} weeks"
    months = round(days / 30)
    return f"about {months} months"

def timing_flag(service: str, days: int) -> str:
    s = (service or "").lower()
    for pattern, label, threshold in SERVICE_FLAGS:
        if re.search(pattern, s):
            return f"{'Overdue' if days >= threshold else 'Due soon'} • {label}"
    return ""

def enrich(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["first_name"], out["last_name"] = zip(*out["Patient name"].apply(split_name))
    out["phone_normalized"] = out["Patient's Phone"].apply(normalize_phone)
    out["start_dt"] = out["Start"].apply(parse_datetime_maybe)
    out["start_date"] = out["start_dt"].apply(lambda d: d.date() if d else pd.NaT)
    out["service_name"] = out.apply(best_service, axis=1)
    out["client_key"] = out.apply(client_key, axis=1)
    return out

def detect_dnc_col(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if str(c).strip().lower() in {
            "opt_out","dnc","do not contact","do_not_contact","do not text","opt out"
        }:
            return c
    return None

# ====== INPUTS ======
st.subheader("1) Upload files")
c1, c2 = st.columns(2)
with c1:
    past_file = st.file_uploader("Past 3 months appointments (CSV or Excel)", type=["csv","xlsx","xls"], key="past")
with c2:
    upcoming_file = st.file_uploader("Upcoming appointments (CSV or Excel)", type=["csv","xlsx","xls"], key="upcoming")

# ====== MAIN ======
if past_file and upcoming_file:
    try:
        past_raw = load_emr_like(past_file)
        upcoming_raw = load_emr_like(upcoming_file)

        # sanity check
        needed = {"Patient name","Patient's Phone","Patient's Email","Start","Purpose","Type"}
        missing_past = needed - set(past_raw.columns)
        missing_up = needed - set(upcoming_raw.columns)
        if missing_past or missing_up:
            st.error(
                f"Missing expected columns. Past missing: {sorted(list(missing_past))} | "
                f"Upcoming missing: {sorted(list(missing_up))}"
            )
        else:
            past = enrich(past_raw)
            upcoming = enrich(upcoming_raw)

            # latest past visit per client
            past = past[past["client_key"] != ""]
            latest_idx = past.groupby("client_key")["start_dt"].idxmax()
            latest = past.loc[latest_idx].copy()

            # build no-upcoming list
            upcoming_clients = set(upcoming.loc[upcoming["client_key"] != "", "client_key"].unique())
            latest["has_upcoming"] = latest["client_key"].isin(upcoming_clients)
            targets = latest[~latest["has_upcoming"]].copy()

            # timing fields
            targets["days_since_last_visit"] = targets["start_date"].apply(
                lambda d: (DATE_TODAY - d).days if pd.notna(d) else None
            )
            targets["human_time"] = targets["days_since_last_visit"].apply(
                lambda n: humanize_days(n) if pd.notna(n) else ""
            )
            targets["timing_flag"] = targets.apply(
                lambda r: timing_flag(
                    r["service_name"],
                    int(r["days_since_last_visit"]) if pd.notna(r["days_since_last_visit"]) else 0
                ),
                axis=1,
            )

            # message
            def build_msg(r, template: str) -> str:
                ctx = {
                    "first_name": r.get("first_name","").strip().split()[0],
                    "last_name": r.get("last_name","").strip(),
                    "full_name": f"{r.get('first_name','').strip()} {r.get('last_name','').strip()}".strip(),
                    "service": r.get("service_name","").strip().lower(),
                    "days_since": int(r["days_since_last_visit"]) if pd.notna(r["days_since_last_visit"]) else "",
                    "human_time": r.get("human_time",""),
                    "last_visit_date": r.get("start_date",""),
                }
                try:
                    return template.format(**ctx)
                except Exception:
                    return DEFAULT_TEMPLATE.format(**ctx)

            targets["suggested_message"] = targets.apply(lambda r: build_msg(r, tmpl), axis=1)

            # optional DNC
            dnc_col_past = detect_dnc_col(past_raw)
            dnc_col_up = detect_dnc_col(upcoming_raw)
            if dnc_col_past or dnc_col_up:
                def dnc_map(df_raw: pd.DataFrame, col: str) -> pd.Series:
                    en = enrich(df_raw)
                    return en.groupby("client_key")[col].agg(lambda x: str(list(x)[-1]) if len(x) else "")
                dmap = pd.Series(dtype=str)
                if dnc_col_past:
                    dmap = dmap.combine_first(dnc_map(past_raw, dnc_col_past))
                if dnc_col_up:
                    dmap = dmap.combine_first(dnc_map(upcoming_raw, dnc_col_up))
                targets["opt_out_flag"] = targets["client_key"].map(dmap).fillna("")

            # final columns
            out = targets.copy()
            out["client_name"] = (out["first_name"] + " " + out["last_name"]).str.strip()
            out["phone"] = out["phone_normalized"]
            out["last_service"] = out["service_name"]
            out["last_visit_date"] = out["start_date"]

            keep_cols = [
                "client_name","phone","Patient's Email","last_visit_date","last_service",
                "days_since_last_visit","human_time","timing_flag","suggested_message",
                "Location","Provider","opt_out_flag"
            ]
            keep_cols = [c for c in keep_cols if c in out.columns]
            out = out[keep_cols].sort_values(by="days_since_last_visit", ascending=False).reset_index(drop=True)

            st.subheader("2) Preview")
            st.dataframe(out.head(50))

            # download buttons
            def to_excel_bytes(df: pd.DataFrame) -> bytes:
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                    df.to_excel(writer, index=False, sheet_name="No Upcoming Appt")
                return buffer.getvalue()

            st.download_button(
                "⬇️ Download Excel (No Upcoming Appt)",
                data=to_excel_bytes(out),
                file_name=f"retention_no_upcoming_{DATE_TODAY.isoformat()}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

            st.download_button(
                "⬇️ Download CSV",
                data=out.to_csv(index=False).encode("utf-8"),
                file_name=f"retention_no_upcoming_{DATE_TODAY.isoformat()}.csv",
                mime="text/csv",
            )

            st.success(f"Built {len(out)} contacts with no upcoming appointment.")

    except Exception as e:
        st.exception(e)
else:
    st.info("Upload both files to continue. Export the same EMR report twice: one filtered to the last 3 months, and one filtered to future dates.")
