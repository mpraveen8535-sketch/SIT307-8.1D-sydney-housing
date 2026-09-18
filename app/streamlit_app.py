from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "model.joblib"

st.set_page_config(page_title="Sydney Housing Price Estimator",
                   page_icon="🏠", layout="wide")


@st.cache_resource
def load_bundle():
    if not MODEL_PATH.exists():
        st.error(
            f"Model file not found at {MODEL_PATH}.\n\n"
            "Run every cell of SIT307_8.1D_Sydney_Housing.ipynb first - the "
            "Part 5 cells write app/model.joblib."
        )
        st.stop()
    return joblib.load(MODEL_PATH)


BUNDLE = load_bundle()
PIPELINE = BUNDLE["pipeline"]
FEATURES = BUNDLE["features"]
TEXT_PATTERNS = BUNDLE["text_patterns"]
LAND_FILL_SQM = BUNDLE["land_fill_sqm"]
SIGMA = BUNDLE["residual_sigma"]
CBD_KM = BUNDLE["cbd_km"]


def engineer(data: pd.DataFrame) -> pd.DataFrame:
    d = data.copy()
    d["sale_date"] = pd.to_datetime(d["sale_date"])

    d["property_age"] = (d["sale_date"].dt.year - d["year_built"]).clip(lower=0)
    d["years_since_2024"] = (d["sale_date"] - pd.Timestamp("2024-01-01")).dt.days / 365.25

    d["area_per_bedroom"] = d["internal_area_sqm"] / d["bedrooms"].clip(lower=1)
    d["bath_per_bed"] = d["bathrooms"] / d["bedrooms"].clip(lower=1)
    d["total_rooms"] = d["bedrooms"] + d["bathrooms"]

    d["has_land"] = d["land_size_sqm"].notna().astype(int)
    d["log_land"] = np.log(d["land_size_sqm"].fillna(LAND_FILL_SQM))
    d["site_coverage"] = (d["internal_area_sqm"] / d["land_size_sqm"]).where(d["has_land"] == 1)

    desc = d["agent_description"].fillna("").str.lower()
    for name, pattern in TEXT_PATTERNS.items():
        d[name] = desc.str.contains(pattern, regex=True).astype(int)
    d["desc_word_count"] = desc.str.split().str.len()

    d["has_pool"] = d["has_pool"].astype(int)
    return d


def predict(rows: pd.DataFrame) -> pd.DataFrame:
    X = engineer(rows)[FEATURES]
    log_pred = PIPELINE.predict(X)
    point = np.exp(log_pred)
    return pd.DataFrame({
        "estimate": point,
        "low_80": point * np.exp(-1.2816 * SIGMA),
        "high_80": point * np.exp(1.2816 * SIGMA),
        "low_95": point * np.exp(-1.96 * SIGMA),
        "high_95": point * np.exp(1.96 * SIGMA),
    }, index=rows.index)


def caveats(row: pd.Series, estimate: float) -> list[str]:
    out = []
    desc = str(row.get("agent_description", "")).lower()

    if estimate > 5_000_000:
        out.append(
            "**Above $5m.** Very few comparable sales exist at this level and "
            "the model under-predicts the top of the market by about 14% on "
            "average. Treat this as a floor rather than a guide."
        )
    if any(k in desc for k in ["waterfront", "harbour view", "jetty", "deep water"]):
        out.append(
            "**Waterfront or harbour outlook mentioned.** This premium is large "
            "and highly variable, and the model only partly captures it from "
            "the description text."
        )
    if "development potential" in desc or "stca" in desc or "da approved" in desc:
        out.append(
            "**Development potential mentioned.** Value then depends on what "
            "the site can actually yield, which is not modelled at all."
        )

    thin = {("Mosman", "Townhouse"), ("Blacktown", "Townhouse"),
            ("Parramatta", "House"), ("Mosman", "Apartment")}
    if (row["suburb"], row["property_type"]) in thin:
        out.append(
            f"**Thin segment.** {row['property_type'].lower()}s are uncommon in "
            f"{row['suburb']} in the training data, so this estimate rests on "
            "few comparable sales."
        )

    area = row.get("internal_area_sqm")
    if pd.notna(area) and area / max(row["bedrooms"], 1) < 30:
        out.append(
            "**Small for its bedroom count.** The model's largest errors are "
            "over-predictions on compact dwellings in expensive suburbs - the "
            "true price is likely below this estimate."
        )
    return out


def money(v) -> str:
    return f"${v:,.0f}"


def money_short(v) -> str:
    if v >= 1_000_000:
        return f"${v/1_000_000:.2f}m"
    return f"${v/1_000:.0f}k"


st.title("🏠 Sydney Housing Price Estimator")
st.caption(
    f"{BUNDLE['model_name']} trained on {BUNDLE['trained_rows']} sales across "
    f"{', '.join(BUNDLE['suburbs'])}. "
    f"Cross-validated error: {BUNDLE['metrics']['MAPE (%)']:.1f}% MAPE "
    f"({money(BUNDLE['metrics']['MAE ($)'])} MAE)."
)

st.warning(
    "**Decision support only.** This is a university project trained on 120 "
    "synthetic sales. It is not a valuation and must not be used to price a "
    "real property.",
    icon="⚠️",
)

with st.sidebar:
    st.header("Property details")

    suburb = st.selectbox("Suburb", BUNDLE["suburbs"])
    property_type = st.selectbox("Property type", BUNDLE["property_types"])

    c1, c2 = st.columns(2)
    bedrooms = c1.number_input("Bedrooms", 1, 8, 3)
    bathrooms = c2.number_input("Bathrooms", 1, 6, 2)
    car_spaces = c1.number_input("Car spaces", 0, 6, 1)
    internal_area = c2.number_input("Internal area (sqm)", 20, 800, 130)

    if property_type == "Apartment":
        land_size = None
        st.caption("Land size does not apply to apartments.")
    else:
        land_size = st.number_input("Land size (sqm)", 50, 3000, 450)

    year_built = st.number_input("Year built", 1880, 2027, 1995)
    condition = st.selectbox("Condition", BUNDLE["conditions"], index=2)
    sale_method = st.selectbox("Sale method", BUNDLE["sale_methods"])

    has_pool = st.checkbox("Swimming pool", value=False)
    station_km = st.slider("Distance to station (km)", 0.1, 5.0, 1.2, 0.1)
    sale_date = st.date_input("Sale date", pd.Timestamp("2026-09-01"))

    description = st.text_area(
        "Agent description (optional)",
        placeholder="Paste the listing text. Mentions of views, renovation, "
                    "or development potential change the estimate.",
        height=110,
    )

    go = st.button("Estimate price", type="primary", use_container_width=True)


single = pd.DataFrame([{
    "suburb": suburb,
    "property_type": property_type,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "car_spaces": car_spaces,
    "land_size_sqm": land_size if land_size else np.nan,
    "internal_area_sqm": internal_area,
    "year_built": year_built,
    "condition": condition,
    "has_pool": has_pool,
    "distance_to_cbd_km": CBD_KM[suburb],
    "distance_to_station_km": station_km,
    "sale_method": sale_method,
    "sale_date": pd.Timestamp(sale_date),
    "days_on_market": np.nan,
    "agent_description": description,
}])

tab_single, tab_batch, tab_about = st.tabs(
    ["Single property", "Batch (CSV upload)", "About the model"])

with tab_single:
    if go:
        res = predict(single).iloc[0]
        row = single.iloc[0]

        st.subheader("Estimated sale price")
        a, b, c = st.columns([1.3, 1, 1])
        a.metric("Point estimate", money(res.estimate))
        b.metric("80% range",
                 f"{money_short(res.low_80)} – {money_short(res.high_80)}")
        c.metric("95% range",
                 f"{money_short(res.low_95)} – {money_short(res.high_95)}")

        st.caption(
            f"Exact 80% range: {money(res.low_80)} to {money(res.high_80)}.  \n"
            f"Exact 95% range: {money(res.low_95)} to {money(res.high_95)}.  \n"
            f"The interval comes from the cross-validated residual spread "
            f"(σ = {SIGMA:.3f} in log space). It reflects how wrong the model "
            f"typically is, not how confident it feels - and it is wide."
        )

        notes = caveats(row, res.estimate)
        if notes:
            st.subheader("Treat this estimate with caution")
            for n in notes:
                st.warning(n, icon="⚠️")
        else:
            st.success(
                "This property sits in a well-represented part of the training "
                "data, so the estimate is as reliable as the model gets.",
                icon="✅",
            )

        with st.expander("What the model read from your inputs"):
            eng = engineer(single)
            flags = {k: int(eng[k].iloc[0]) for k in TEXT_PATTERNS
                     if eng[k].iloc[0] == 1}
            st.write("**Text flags detected:**",
                     ", ".join(flags) if flags else "none")
            st.dataframe(
                eng[["property_age", "years_since_2024", "area_per_bedroom",
                     "bath_per_bed", "has_land", "log_land"]].T.rename(
                    columns={0: "value"}).round(3),
                use_container_width=True,
            )
    else:
        st.info("Enter the property details in the sidebar, then press "
                "**Estimate price**.", icon="👈")

with tab_batch:
    st.write(
        "Upload a CSV with the same columns as `data/sydney_housing.csv`. "
        "`sale_price` is ignored if present, and is used to report accuracy."
    )
    up = st.file_uploader("CSV file", type="csv")

    if up is not None:
        batch = pd.read_csv(up, parse_dates=["sale_date"])
        missing = [c for c in
                   ["suburb", "property_type", "bedrooms", "bathrooms",
                    "car_spaces", "land_size_sqm", "internal_area_sqm",
                    "year_built", "condition", "has_pool",
                    "distance_to_cbd_km", "distance_to_station_km",
                    "sale_method", "sale_date", "days_on_market",
                    "agent_description"]
                   if c not in batch.columns]
        if missing:
            st.error("Missing required columns: " + ", ".join(missing))
        else:
            res = predict(batch)
            out = pd.concat([batch, res], axis=1)

            if "sale_price" in batch.columns:
                ape = 100 * (out.estimate - out.sale_price).abs() / out.sale_price
                out["abs_pct_error"] = ape
                m1, m2, m3 = st.columns(3)
                m1.metric("Properties scored", len(out))
                m2.metric("MAPE", f"{ape.mean():.1f}%")
                m3.metric("Median APE", f"{ape.median():.1f}%")
                st.caption(
                    f"⚠️ If these rows were part of training, this error is "
                    f"optimistic - the model has already seen them. The honest "
                    f"figure is the cross-validated "
                    f"{BUNDLE['metrics']['MAPE (%)']:.1f}% MAPE quoted above."
                )
            else:
                st.metric("Properties scored", len(out))

            show = ["suburb", "property_type", "bedrooms", "internal_area_sqm",
                    "estimate", "low_80", "high_80"]
            if "sale_price" in out.columns:
                show = ["sale_price"] + show + ["abs_pct_error"]
            st.dataframe(out[show].round(0), use_container_width=True, height=380)

            st.download_button(
                "Download predictions as CSV",
                out.to_csv(index=False).encode(),
                file_name="predictions.csv",
                mime="text/csv",
            )

with tab_about:
    st.subheader("How this works")
    st.markdown(f"""
The estimator is the **{BUNDLE['model_name']}** pipeline selected in Part 3 of
the notebook. It predicts **log(sale price)** and exponentiates the result,
because raw Sydney prices span $295k to $12.6m and are strongly right-skewed.

The saved artefact is the *entire* scikit-learn pipeline - imputation, scaling
and one-hot encoding included - so this app cannot drift out of step with how
the model was trained. It does not re-implement any preprocessing.

**Cross-validated performance ({BUNDLE['trained_rows']} properties, 5-fold):**

| Metric | Value |
|---|---|
| MAE | {money(BUNDLE['metrics']['MAE ($)'])} |
| RMSE | {money(BUNDLE['metrics']['RMSE ($)'])} |
| MAPE | {BUNDLE['metrics']['MAPE (%)']:.1f}% |
| Median APE | {BUNDLE['metrics']['Median APE (%)']:.1f}% |
| R² (log price) | {BUNDLE['metrics']['R2 (log)']:.3f} |

**Known limitations**, from Part 4 of the notebook:

- It under-predicts the top of the market by roughly 14%.
- It over-predicts compact dwellings in expensive suburbs.
- It has no information on aspect, position, renovation quality, strata levies
  or school catchment - all significant real price drivers.
- It only knows the three suburbs it was trained on, which is why the suburb
  field is a fixed list rather than free text.
""")
