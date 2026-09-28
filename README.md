# SIT307 8.1D - Sydney housing price prediction

This revised project uses 120 publicly shown sold-property records: 40 each from Blacktown, Parramatta and Mosman. It replaces the earlier generated dataset. The PDF report explains the collection, analysis, model comparison, five largest errors and Streamlit app.

## Files

- `data/sold_properties.csv`: property records and a source URL for each row. `listing_url` is populated for 115 records; the other five retain their `results_page_url`.
- `SIT307_8.1D_Sydney_Housing.ipynb`: executed analysis notebook with visible outputs. It regenerates the charts, cross-validation tables and final model.
- `modeling.py`: shared feature builder and model definitions.
- `app/streamlit_app.py` and `app/model.joblib`: web app and fitted model.
- `data/example_input.csv`: a three-row batch example.
- `figures/`: charts, evaluation tables and app screenshots used in the report.
- `SIT307_8.1D_Revised_Report.pdf`: concise report and use instructions.

## Reproduce and run

Use Python 3.11 or newer. From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace SIT307_8.1D_Sydney_Housing.ipynb
streamlit run app/streamlit_app.py
```

Open the local URL printed by Streamlit. The single-property form accepts a suburb, dwelling type, date, rooms, parking and advertised area. The batch tab accepts a CSV with `suburb`, `property_type_listed`, `bedrooms`, `bathrooms` and `sale_date`; `car_spaces` and `listed_area_sqm` are optional. Use `data/example_input.csv` as a template.

The app uses the same saved gradient-boosting pipeline produced by the notebook. Its historical error band is based on five-fold out-of-fold residuals and is approximate, not a formal valuation interval.

The report and screenshot scripts are optional. Run `python build_report.py` after the notebook to regenerate the PDF. To recapture app screenshots, install the Playwright browser with `python -m playwright install chromium`, leave Streamlit running, then run `python capture_app.py` in a second terminal. Set `STREAMLIT_URL` if the app is on a port other than 8501.

## Collection and limits

Records were transcribed from realestate.com.au sold result cards visible on 28 September 2026, with sale dates from 26 February to 27 September 2026. Rows with withheld prices and apparent multi-property sales were excluded. Each CSV row keeps an address, price, date and source page. Several listing cards omit parking or area; blanks remain missing and are imputed inside training folds. The displayed area may be land, building or apartment area, so the column is named `listed_area_sqm` rather than claiming one consistent meaning.

This is a selected sample of public sold results, not a complete or random Sydney transaction register. Its three-suburb scope, uneven dwelling mix and missing amenity, condition and outlook fields limit use beyond a university prototype. For source details and validation discussion, see the report and notebook.
