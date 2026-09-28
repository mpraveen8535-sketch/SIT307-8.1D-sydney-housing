"""Small decision-support prototype trained on the submitted sold listings."""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="Sydney sold-price estimate", page_icon="🏠", layout="wide")


@st.cache_resource
def load_bundle():
    path = ROOT / "app" / "model.joblib"
    if not path.exists():
        st.error("Model missing. Run the full analysis notebook first.")
        st.stop()
    return joblib.load(path)


BUNDLE = load_bundle()
REQUIRED = ["suburb", "property_type_listed", "bedrooms", "bathrooms", "sale_date"]
OPTIONAL = ["car_spaces", "listed_area_sqm"]


def dollars(value):
    return f"${value:,.0f}"


def score(frame):
    predictions = BUNDLE["model"].predict(frame)
    factor = BUNDLE["error_factor_80"]
    result = frame.copy()
    result["predicted_aud"] = predictions.round(0)
    result["historical_low_80_aud"] = (predictions / factor).round(0)
    result["historical_high_80_aud"] = (predictions * factor).round(0)
    return result


def validate(frame):
    missing = [name for name in REQUIRED if name not in frame.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    data = frame.copy()
    for name in OPTIONAL:
        if name not in data:
            data[name] = np.nan
    if data[REQUIRED].isna().any().any():
        raise ValueError("Required fields contain blank values.")
    if not data.suburb.isin(BUNDLE["suburbs"]).all():
        raise ValueError("Suburb must be Blacktown, Parramatta or Mosman.")
    if not data.property_type_listed.isin(BUNDLE["types"]).all():
        raise ValueError("Property type must match a type used in training.")
    for name in ["bedrooms", "bathrooms"]:
        data[name] = pd.to_numeric(data[name], errors="raise")
        if data[name].lt(1).any():
            raise ValueError(name + " must be at least 1.")
    for name in OPTIONAL:
        data[name] = pd.to_numeric(data[name], errors="coerce")
    data["sale_date"] = pd.to_datetime(data["sale_date"], errors="raise")
    return data


def warnings_for(row, estimate):
    messages = []
    key = f"{row.suburb}|{row.property_type_listed}"
    n = BUNDLE["segment_counts"].get(key, 0)
    if n < 5:
        messages.append(f"Only {n} comparable {row.property_type_listed.lower()} sale(s) in {row.suburb} are in this sample.")
    date = pd.Timestamp(row.sale_date)
    if date < pd.Timestamp(BUNDLE["min_date"]) or date > pd.Timestamp(BUNDLE["max_date"]):
        messages.append("The chosen sale date is outside the collection period. Market changes are not tested here.")
    for field, label in [("bedrooms", "Bedrooms"), ("bathrooms", "Bathrooms"), ("car_spaces", "Parking spaces"), ("listed_area_sqm", "Advertised area")]:
        value = getattr(row, field)
        if pd.notna(value):
            low, high = BUNDLE["ranges"][field]
            if value < low or value > high:
                messages.append(f"{label} is outside the observed training range ({low:g}–{high:g}).")
    if pd.isna(row.listed_area_sqm):
        messages.append("No advertised area was supplied; the model filled this from training data.")
    if estimate > 5_000_000:
        messages.append("High-price Mosman sales produced some of the largest validation errors; outlook and condition are not modelled.")
    return messages


st.title("Sydney sold-price estimate")
st.caption("SIT307 8.1D revised prototype · 120 actual sold listings · Blacktown, Parramatta and Mosman")
st.warning("University decision-support exercise only. This is not a formal valuation or price guide.")

with st.expander("Data and model", expanded=False):
    st.write(
        f"The {BUNDLE['model_name']} model was selected by five-fold cross-validation. "
        f"Its out-of-fold mean absolute error was {dollars(BUNDLE['metrics']['mae_aud'])}; "
        f"mean absolute percentage error was {BUNDLE['metrics']['mape_pct']:.1f}%. "
        f"Sales range from {BUNDLE['min_date']} to {BUNDLE['max_date']}. "
        "The 80% band below uses past cross-validation residuals and is only an approximate guide to observed error."
    )

single_tab, batch_tab = st.tabs(["Single property", "Batch CSV"])
with single_tab:
    with st.form("property"):
        c1, c2, c3 = st.columns(3)
        suburb = c1.selectbox("Suburb", BUNDLE["suburbs"])
        property_type = c2.selectbox("Property type", BUNDLE["types"], index=BUNDLE["types"].index("House"))
        sale_date = c3.date_input("Expected sale date", value=pd.Timestamp("2026-09-15"))
        bedrooms = c1.number_input("Bedrooms", min_value=1, max_value=12, value=3)
        bathrooms = c2.number_input("Bathrooms", min_value=1, max_value=10, value=2)
        car_spaces = c3.number_input("Parking spaces", min_value=0, max_value=12, value=1)
        has_area = st.checkbox("Advertised area is available", value=True)
        listed_area = st.number_input("Advertised area (m²)", min_value=1.0, max_value=5000.0, value=500.0, disabled=not has_area)
        st.caption("The area label on listing cards is inconsistent across houses and apartments. Enter the number as advertised; leave it blank if absent.")
        submitted = st.form_submit_button("Estimate sold price", type="primary")
    if submitted:
        input_row = pd.DataFrame([{
            "suburb": suburb, "property_type_listed": property_type,
            "bedrooms": bedrooms, "bathrooms": bathrooms, "car_spaces": car_spaces,
            "listed_area_sqm": listed_area if has_area else np.nan,
            "sale_date": sale_date,
        }])
        result = score(validate(input_row)).iloc[0]
        st.subheader("Estimated sale price")
        a, b, c = st.columns(3)
        a.metric("Point estimate", dollars(result.predicted_aud))
        b.metric("Historical 80% low", dollars(result.historical_low_80_aud))
        c.metric("Historical 80% high", dollars(result.historical_high_80_aud))
        for message in warnings_for(input_row.iloc[0], result.predicted_aud):
            st.info(message)

with batch_tab:
    st.write("Upload a CSV with suburb, property_type_listed, bedrooms, bathrooms and sale_date. car_spaces and listed_area_sqm may be blank or omitted.")
    example = pd.DataFrame([{
        "suburb":"Blacktown", "property_type_listed":"House", "bedrooms":3,
        "bathrooms":2, "car_spaces":1, "listed_area_sqm":550,
        "sale_date":"2026-09-15",
    }])
    st.download_button("Download example CSV", example.to_csv(index=False), "example_input.csv", "text/csv")
    upload = st.file_uploader("Choose a CSV", type="csv")
    if upload is not None:
        try:
            frame = pd.read_csv(upload)
            if len(frame) > 500:
                raise ValueError("Batch size is limited to 500 rows.")
            result = score(validate(frame))
            st.dataframe(result, use_container_width=True)
            st.download_button("Download estimates", result.to_csv(index=False), "housing_estimates.csv", "text/csv")
        except Exception as error:
            st.error(str(error))
