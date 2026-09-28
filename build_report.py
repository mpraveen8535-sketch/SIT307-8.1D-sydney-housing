"""Build a concise report directly from the executed notebook outputs."""

from pathlib import Path
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parent
FIG = ROOT / 'figures'
REPORT = ROOT / 'SIT307_8.1D_Revised_Report.pdf'
df = pd.read_csv(ROOT / 'data' / 'sold_properties.csv')
cv = pd.read_csv(FIG / 'cv_results.csv').set_index('model')
sub = pd.read_csv(FIG / 'suburb_summary.csv').set_index('suburb')
err = pd.read_csv(FIG / 'error_by_suburb.csv').set_index('suburb')
worst = pd.read_csv(FIG / 'worst_predictions.csv')
imp = pd.read_csv(FIG / 'permutation_checks.csv').set_index('feature')
best = cv.loc['Gradient boosting']

body = ParagraphStyle('body', fontName='Times-Roman', fontSize=10, leading=13.5, spaceAfter=7)
heading = ParagraphStyle('heading', fontName='Times-Bold', fontSize=14, leading=17, spaceBefore=12, spaceAfter=7)
subhead = ParagraphStyle('subhead', fontName='Times-Bold', fontSize=11, leading=14, spaceBefore=9, spaceAfter=4)
small = ParagraphStyle('small', fontName='Times-Roman', fontSize=8.7, leading=11.3, spaceAfter=5)
caption = ParagraphStyle('caption', fontName='Times-Italic', fontSize=8.5, leading=11, alignment=TA_CENTER, spaceBefore=3, spaceAfter=10)
title = ParagraphStyle('title', fontName='Times-Bold', fontSize=17, leading=22, alignment=TA_CENTER, spaceAfter=14)
center = ParagraphStyle('center', fontName='Times-Roman', fontSize=11, leading=15, alignment=TA_CENTER, spaceAfter=10)
mono = ParagraphStyle('mono', fontName='Courier', fontSize=8, leading=11, leftIndent=10, spaceAfter=2)

story = []
def p(text, style=body): story.append(Paragraph(text, style))
def space(h=5): story.append(Spacer(1, h))
def fig(filename, text, width=15.4*cm):
    from PIL import Image as PILImage
    path = FIG / filename
    with PILImage.open(path) as image:
        w, h = image.size
    story.append(KeepTogether([Image(str(path), width=width, height=width*h/w), Paragraph(text, caption)]))
def table(rows, widths, size=8.2):
    style = ParagraphStyle('cell', parent=small, fontSize=size, leading=size*1.22, spaceAfter=0)
    wrapped = [[Paragraph(str(value).replace('&','&amp;'), style) for value in row] for row in rows]
    t = Table(wrapped, colWidths=widths, repeatRows=1, hAlign='LEFT')
    t.setStyle(TableStyle([
        ('FONTNAME',(0,0),(-1,0),'Times-Bold'),
        ('VALIGN',(0,0),(-1,-1),'TOP'),
        ('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4),
        ('LEFTPADDING',(0,0),(-1,-1),4),('RIGHTPADDING',(0,0),(-1,-1),4),
        ('LINEBELOW',(0,0),(-1,0),0.7,colors.black),
        ('LINEBELOW',(0,-1),(-1,-1),0.4,colors.black),
    ]))
    story.append(t); space(8)
def money(v): return f'${v:,.0f}'

p('SIT307 Assessment Task 8.1D', title)
p('Sydney Housing Price Prediction and Decision Support System', center)
p('Revised report after tutor feedback | 28 September 2026', center)
p('Blacktown, Parramatta and Mosman', center)
space(10)
p('<b>Revision summary.</b> I replaced the generated 120-row dataset with 120 actual sold-property records shown on realestate.com.au, then reran every chart, model comparison, error analysis and app estimate. The previous synthetic-data results are not used. The notebook, dataset and app source accompany this report in the source ZIP.')

p('Part 1: Problem and collection', heading)
p('The aim is to estimate sale price from advertised property details as decision support for a buyer or agent. I chose Blacktown for an outer-west market, Parramatta for a dense apartment market, and Mosman for a higher-priced harbour-side market. Their dwelling mix and price range differ enough to test whether one small model generalises across them.')
p('I recorded 40 sold listings per suburb from public result cards between 26 February and 27 September 2026. Each CSV row contains the address, sale price, sale date, dwelling type, bedrooms, bathrooms, parking, advertised area where shown, a results-page URL and, for 115 rows, a direct sold-listing URL. The five without a stable direct URL retain their results page. I excluded price-withheld cards and obvious multi-dwelling sales. No values were generated or estimated.')
table([
    ['Suburb','Sales','Median price','Observed range','Mix in sample'],
    ['Blacktown','40',money(sub.loc['Blacktown','median_price']),f"{money(sub.loc['Blacktown','min_price'])}–{money(sub.loc['Blacktown','max_price'])}",'22 houses; 14 units/apartments; 4 townhouses'],
    ['Parramatta','40',money(sub.loc['Parramatta','median_price']),f"{money(sub.loc['Parramatta','min_price'])}–{money(sub.loc['Parramatta','max_price'])}",'37 units/apartments; 2 townhouses; 1 house'],
    ['Mosman','40',money(sub.loc['Mosman','median_price']),f"{money(sub.loc['Mosman','min_price'])}–{money(sub.loc['Mosman','max_price'])}",'31 units/apartments; 8 houses; 1 townhouse'],
], [2.2*cm,1.1*cm,2.7*cm,4.1*cm,5.0*cm])
p('<b>Quality and bias.</b> Parking is absent on 15 of 120 cards and advertised area on 60. The card’s area can mean different things for houses and apartments, so I retain its neutral label. The sample is drawn from selected result pages, not randomly from all sales; it excludes withheld prices and has only one Parramatta house. Listing text, condition, outlook, strata levies and precise location were not consistently collected. These omissions limit valuation claims.')

p('Part 2: Understanding data and engineering features', heading)
fig('01_price_distribution.png','Figure 1. Sale prices are skewed; models fit log(price) and return estimates in AUD.', 14.0*cm)
fig('02_suburb_type.png','Figure 2. Overall suburb medians reflect dwelling mix. Mosman houses differ greatly from its apartments.', 14.0*cm)
p('<b>Before modelling</b> I expected suburb, dwelling type and bedroom count to be the strongest drivers. I added sale month, bathrooms per bedroom, parking per bedroom, advertised area per bedroom, and an area-missing flag. Numeric missing values are imputed and categories encoded within each cross-validation training fold. Address and source URLs are audit fields, never predictors. These choices largely matched the initial reasoning: out-of-fold shuffling increased MAPE by 42.3 points for suburb, 10.3 for type and 10.0 for bedrooms; area changed it by 0.9 points. These are predictive checks, not causal effects.')
fig('03_monthly_medians.png','Figure 3. Monthly medians are shown for context only. Few sales and changing dwelling mix prevent a reliable growth estimate.', 13.5*cm)

p('Part 3: Model comparison', heading)
p('Before training, I predicted gradient boosting would outperform random forest and Ridge because price effects likely interact with suburb and type. Ridge is a regularised linear baseline; the two tree ensembles capture nonlinear patterns but may overfit this small sample. I used five shuffled folds (seed 307), with the complete preprocessing pipeline fitted inside each fold. The ranking criterion was mean absolute percentage error (MAPE); I also report absolute error in dollars and R².')
table([
    ['Model','Out-of-fold MAE','MAPE','R² (AUD)','Mean train MAPE'],
    *[[name,money(row.mae_aud),f'{row.mape_pct:.1f}%',f'{row.r2_dollar:.3f}',f'{row.mean_train_mape_pct:.1f}%'] for name,row in cv.iterrows()],
], [3.4*cm,3.1*cm,2.0*cm,2.5*cm,3.0*cm])
p(f"Gradient boosting performed best: {money(best.mae_aud)} MAE and {best.mape_pct:.1f}% MAPE. Its mean training MAPE was {best.mean_train_mape_pct:.1f}%, versus {best.mean_test_mape_pct:.1f}% on held-out folds, showing some overfitting. The forest’s gap was also material, while Ridge underfit price differences in the prestige segment. The expected ranking was supported, but a 17.4% average percentage error is too large for a precise price guide.")
fig('04_model_comparison.png','Figure 4. Gradient boosting has the lowest out-of-fold MAPE on this sample.', 11.5*cm)
p('The random-fold result estimates similar-period sales in these three suburbs. It does not test a later market or an unseen suburb. The model is therefore recommended only as a prototype, with explicit warnings and an error band.')

p('Part 4: Five largest prediction failures', heading)
p('I ranked absolute <i>dollar</i> errors from the chosen model’s held-out predictions. Positive error means the model overestimated the sale. All five are Mosman properties; their error reflects thin high-price coverage and features missing from the card fields.')
table([
    ['Property','Actual','Predicted','Error'],
    *[[row.address.replace(', Mosman',''),money(row.sale_price_aud),money(row.predicted_aud),f"{row.error_aud/1e6:+.2f}m"] for _,row in worst.iterrows()],
], [6.1*cm,2.6*cm,2.6*cm,2.0*cm],size=7.7)
p('<b>G05/15-25 Myahgah Road:</b> The model underpriced a $4.5m apartment by $3.37m. Its result card omitted area, while the listing describes a new 213 m² garden residence with a private terrace and premium finishes. A normal two-bedroom label concealed an unusual property.')
p('<b>8A Prince Street:</b> The model overpredicted by $2.69m. It has four bedrooms and three bathrooms but only a 258 m² lot; the model’s other Mosman house examples include much larger prestige sites. This sale illustrates how a small sample lets broad room counts overpower site context.')
p('<b>5 Botanic Road:</b> Underpriced by $2.61m. The card had no area, and the listing describes a position close to Balmoral Beach, a pool and high-end fitout. None of those premiums entered the model.')
p('<b>10 Cyprian Street:</b> Underpriced by $2.58m. The listing describes expansive Middle Harbour views and redevelopment scope; neither is a structured input. A standard 4-bedroom/3-bathroom record cannot capture that outlook.')
p('<b>2 Lennox Street:</b> Underpriced by $1.95m. The listed 659 m² lot helps, but restored character, a pool, solar and other amenities are absent. In this segment, individual qualities matter more than the small set of card features.')
p(f"Mean absolute error by suburb was {money(err.loc['Blacktown','mae_aud'])} in Blacktown, {money(err.loc['Parramatta','mae_aud'])} in Parramatta and {money(err.loc['Mosman','mae_aud'])} in Mosman. Predictions need particular caution for prestige, waterfront, newly built or otherwise unusual stock.")

p('Part 5: App, use and reflection', heading)
p('I saved the final gradient-boosting pipeline to <font face="Courier">app/model.joblib</font> and built a Streamlit page that loads it unchanged. Users enter suburb, type, expected sale date, room counts, parking and advertised area, or upload a CSV. The app reports a point estimate plus an approximate 80% historical error band from cross-validated log residuals. It warns when inputs fall outside training ranges or a suburb/type segment has fewer than five records. The band is not independently calibrated and should not be treated as a valuation interval.')
fig('07_app_result.png','Figure 5. Single-property estimate using a three-bedroom Blacktown house. The app shows the input fields, point estimate and historical error band.', 14.4*cm)
fig('08_app_batch.png','Figure 6. Batch upload scores three example rows and offers a download of estimates.', 14.4*cm)
p('<b>Build and run.</b> From the source ZIP folder, create a Python environment and install the packages in <font face="Courier">requirements.txt</font>. Run the notebook top to bottom to reproduce the figures and model, then start the app:')
for line in ['python -m pip install -r requirements.txt', 'jupyter nbconvert --to notebook --execute --inplace SIT307_8.1D_Sydney_Housing.ipynb', 'streamlit run app/streamlit_app.py']:
    p(line, mono)
p('The app opens at localhost:8501. Select “Single property”, fill the form and choose “Estimate sold price”; for batch use, open “Batch CSV”, download the example file, replace its rows and upload it.')
p('<b>Reflection.</b> The key change was provenance. The earlier synthetic data made model quality look stronger than it was. On actual sales, the top errors reveal how a small and uneven sample misses outlook, condition and location within a suburb. More data, consistent area definitions and a later-date holdout would improve the study. Its present value is as a transparent learning prototype, not a pricing tool. Sold-only data also omits passed-in or withdrawn stock and can inherit market and listing biases.')

p('Appendix: Source and references', heading)
p('The companion archive <font face="Courier">SIT307_8.1D_Revised_Source.zip</font> contains the 120-row CSV, executed notebook, application source, fitted model, figures and run instructions. Source repository: https://github.com/mpraveen8535-sketch/SIT307-8.1D-sydney-housing')
p('The CSV supplies a source URL for every record. Representative sold-listing sources used in the error analysis:', small)
for row in worst.itertuples():
    p(f'{row.address}: {row.listing_url}', small)
p('Scikit-learn developers. User Guide: cross-validation, preprocessing pipelines, Ridge, random forest and gradient boosting. https://scikit-learn.org/stable/user_guide.html', small)
p('Streamlit developers. Documentation. https://docs.streamlit.io/', small)

def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont('Times-Roman',9)
    canvas.drawCentredString(A4[0]/2, 1.2*cm, str(doc.page))
    canvas.restoreState()

doc = SimpleDocTemplate(str(REPORT), pagesize=A4, rightMargin=2.2*cm, leftMargin=2.2*cm, topMargin=1.8*cm, bottomMargin=1.8*cm)
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print(REPORT)
