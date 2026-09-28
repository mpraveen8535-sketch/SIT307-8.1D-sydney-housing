"""Write the resubmission notebook with a visible, executable workflow."""

from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parent
cells = []


def md(body):
    cells.append(nbf.v4.new_markdown_cell(body.strip()))


def code(body):
    cells.append(nbf.v4.new_code_cell(body.strip()))


md("""
# SIT307 8.1D: Sydney housing price decision support

**Revised after tutor feedback, 28 September 2026.** This notebook uses 120 actual sold-property records shown on realestate.com.au: 40 each in Blacktown, Parramatta and Mosman. Each row has a results-page URL and 115 also have a direct sold-listing URL. The earlier synthetic dataset, model and conclusions are not used here.

The code runs top to bottom from the project root. It saves the figures, result tables and `app/model.joblib` used by the Streamlit prototype.
""")

code("""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, median_absolute_error, r2_score
from sklearn.model_selection import KFold
from modeling import make_model

ROOT = Path.cwd()
FIG = ROOT / 'figures'
FIG.mkdir(exist_ok=True)
sns.set_theme(style='whitegrid')
plt.rcParams.update({'figure.dpi': 110, 'savefig.dpi': 160, 'savefig.bbox': 'tight'})
pd.set_option('display.max_colwidth', 90)
pd.set_option('display.float_format', lambda v: f'{v:,.2f}')
""")

md("""
## Part 1: Problem and data collection

I chose Blacktown as a more accessible outer-west market, Parramatta as a dense apartment market and Mosman as a higher-priced harbour-side market. The task is to estimate a sold price from listing fields for decision support. A prediction is not a formal valuation.

I transcribed the sale price, sale date, dwelling type and advertised property fields from publicly visible sold-result cards. The CSV preserves the address and source-page URL for all 120 records, plus a direct sold-listing URL for 115. Five cards did not yield a stable direct URL during verification, so their results page is the audit trail. Collection was a snapshot on 28 September 2026. Price-withheld cards and multi-dwelling sales were excluded. This is a small convenience sample, not a random sample of all Sydney sales.

`listed_area_sqm` is exactly the area shown on the card. For houses it is usually land; for units it can have a different meaning, so I do not call it floor area or treat it as directly comparable across dwelling types. Blank fields remain blank; I did not invent age, condition, station distance or marketing days.
""")

code("""
df = pd.read_csv(ROOT / 'data' / 'sold_properties.csv', parse_dates=['sale_date'])
assert len(df) == 120
assert df['suburb'].value_counts().to_dict() == {'Blacktown': 40, 'Parramatta': 40, 'Mosman': 40}
assert df['address'].is_unique
assert df['sale_price_aud'].gt(0).all()
print(f"{len(df)} sold records, {df['listing_url'].notna().sum()} direct listing URLs")
print('Sale range:', df.sale_date.min().date(), 'to', df.sale_date.max().date())
display(df[['property_id','address','property_type_listed','bedrooms','bathrooms','car_spaces','listed_area_sqm','sale_date','sale_price_aud']].head())
""")

code("""
quality = pd.DataFrame({'missing_count': df.isna().sum(), 'missing_pct': (100 * df.isna().mean()).round(1)})
display(quality.loc[['car_spaces','listed_area_sqm','listing_url']])
display(pd.crosstab(df['suburb'], df['property_type_listed']))
summary = df.groupby('suburb').agg(n=('sale_price_aud','size'), median_price=('sale_price_aud','median'), min_price=('sale_price_aud','min'), max_price=('sale_price_aud','max'))
display(summary)
summary.to_csv(FIG / 'suburb_summary.csv')
quality.to_csv(FIG / 'data_quality.csv')
""")

md("""
## Part 2: Understanding prices and building features

Before fitting models, my three predicted drivers are: **suburb** (different underlying land markets), **dwelling type** (a land-backed house differs from a strata unit), and **bedroom count** (a widely published measure of usable capacity). I expect area to help for houses, but its missing and mixed definition makes it less dependable than those three fields.
""")

code("""
fig, ax = plt.subplots(1, 2, figsize=(10, 3.7))
sns.histplot(df.sale_price_aud / 1e6, bins=22, ax=ax[0], color='#555555')
ax[0].set(xlabel='Sale price (AUD millions)', ylabel='Properties', title='Sale prices')
sns.histplot(np.log(df.sale_price_aud), bins=22, ax=ax[1], color='#888888')
ax[1].set(xlabel='Natural log of sale price', ylabel='Properties', title='Log prices')
fig.tight_layout(); fig.savefig(FIG / '01_price_distribution.png'); plt.show()
print('Raw-price skew:', round(df.sale_price_aud.skew(), 2), '| log-price skew:', round(np.log(df.sale_price_aud).skew(), 2))
""")

code("""
plot_df = df.copy()
plot_df['dwelling_group'] = plot_df.property_type_listed.replace({'Unit':'Apartment','Studio':'Apartment'})
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.boxplot(data=plot_df, x='suburb', y='sale_price_aud', ax=ax[0], color='#cccccc')
ax[0].set_yscale('log'); ax[0].set(ylabel='Sale price (AUD, log scale)', title='By suburb')
sns.boxplot(data=plot_df, x='suburb', y='sale_price_aud', hue='dwelling_group', ax=ax[1])
ax[1].set_yscale('log'); ax[1].set(ylabel='Sale price (AUD, log scale)', title='By suburb and type')
ax[1].legend(title='Type', fontsize=8)
fig.tight_layout(); fig.savefig(FIG / '02_suburb_type.png'); plt.show()
display(plot_df.groupby(['suburb','dwelling_group']).sale_price_aud.agg(['count','median']))
""")

code("""
plot_df['month'] = plot_df.sale_date.dt.to_period('M').astype(str)
monthly = plot_df.groupby(['month','suburb']).sale_price_aud.agg(['count','median']).reset_index()
fig, ax = plt.subplots(figsize=(9, 3.4))
for suburb, group in monthly.groupby('suburb'):
    ax.plot(group.month, group['median']/1e6, marker='o', label=suburb)
ax.set(ylabel='Median sale price (AUD millions)', xlabel='Sale month', title='Observed monthly medians (sale mix changes)')
ax.legend(); ax.tick_params(axis='x', rotation=45)
fig.tight_layout(); fig.savefig(FIG / '03_monthly_medians.png'); plt.show()
display(monthly)
print('Caution: monthly counts are small and the mix of dwelling types changes; these lines are not a market growth estimate.')
""")

code("""
group_median = plot_df.groupby(['suburb','dwelling_group']).sale_price_aud.transform('median')
plot_df['ratio_to_group_median'] = plot_df.sale_price_aud / group_median
display(plot_df.nlargest(5, 'ratio_to_group_median')[['address','dwelling_group','sale_price_aud','ratio_to_group_median']])
print('All valid high-price observations remain in modelling; removing them would hide a difficult part of the task.')
""")

md("""
The shared `FeatureBuilder` in `modeling.py` converts the sale date to month, merges “Unit” with “Apartment”, and adds bathroom/bedroom, parking/bedroom and advertised-area/bedroom ratios. It also flags when area is missing. Median imputation, scaling and one-hot encoding sit *inside* each training fold, avoiding leakage. Address and URLs are kept for audit and error analysis but never used as predictors.

## Part 3: Models and evaluation

I selected Ridge as a simple regularised baseline, random forest for nonlinear interactions, and gradient boosting for a more flexible sequential tree model. **Before fitting**, I expected gradient boosting to perform best, then forest, then Ridge, because dwelling features can have different prices in each suburb. With only 120 rows, the tree models may overfit. I use five shuffled folds with a fixed seed and compare out-of-fold errors in Australian dollars and percentages. Each model fits log(price) and converts predictions back to dollars.
""")

code("""
X = df.drop(columns=['sale_price_aud'])
y = df.sale_price_aud.to_numpy()
cv = KFold(n_splits=5, shuffle=True, random_state=307)
names = ['Ridge', 'Random forest', 'Gradient boosting']
predictions = {}
metrics = []
fold_details = []
for name in names:
    oof = np.empty(len(df))
    train_mapes, test_mapes = [], []
    for fold, (train, test) in enumerate(cv.split(X), start=1):
        fitted = clone(make_model(name)).fit(X.iloc[train], y[train])
        oof[test] = fitted.predict(X.iloc[test])
        train_mape = 100 * mean_absolute_percentage_error(y[train], fitted.predict(X.iloc[train]))
        test_mape = 100 * mean_absolute_percentage_error(y[test], oof[test])
        train_mapes.append(train_mape); test_mapes.append(test_mape)
        fold_details.append({'model':name,'fold':fold,'train_mape_pct':train_mape,'test_mape_pct':test_mape})
    predictions[name] = oof
    metrics.append({'model':name,
        'mae_aud':mean_absolute_error(y,oof),
        'mape_pct':100*mean_absolute_percentage_error(y,oof),
        'median_ape_pct':100*np.median(np.abs(y-oof)/y),
        'r2_dollar':r2_score(y,oof),
        'mean_train_mape_pct':np.mean(train_mapes),
        'mean_test_mape_pct':np.mean(test_mapes)})
results = pd.DataFrame(metrics).set_index('model').sort_values('mape_pct')
display(results.round(2))
results.to_csv(FIG / 'cv_results.csv')
pd.DataFrame(fold_details).to_csv(FIG / 'cv_folds.csv', index=False)
best_name = results.index[0]
print('Best by preselected MAPE criterion:', best_name)
""")

code("""
fig, ax = plt.subplots(figsize=(7, 3.2))
results['mape_pct'].sort_values().plot.barh(ax=ax, color='#777777')
ax.set(xlabel='Out-of-fold mean absolute percentage error (%)', ylabel='', title='Five-fold model comparison')
fig.tight_layout(); fig.savefig(FIG / '04_model_comparison.png'); plt.show()
""")

code("""
best_oof = predictions[best_name]
fig, ax = plt.subplots(figsize=(5.5, 4.8))
for suburb in ['Blacktown','Parramatta','Mosman']:
    mask = df.suburb.eq(suburb).to_numpy()
    ax.scatter(y[mask]/1e6, best_oof[mask]/1e6, label=suburb, alpha=.75)
limit = max(y.max(), best_oof.max())/1e6
ax.plot([0,limit],[0,limit], color='black', linewidth=1)
ax.set(xlabel='Actual (AUD millions)', ylabel='Out-of-fold predicted (AUD millions)', title=f'{best_name}: prediction errors')
ax.legend(); fig.tight_layout(); fig.savefig(FIG / '05_predicted_actual.png'); plt.show()
""")

code("""
diagnostics = []
rng = np.random.default_rng(307)
for train, test in cv.split(X):
    fitted = clone(make_model(best_name)).fit(X.iloc[train], y[train])
    Xt = X.iloc[test].copy()
    base = mean_absolute_percentage_error(y[test], fitted.predict(Xt))
    for column in ['suburb','property_type_listed','bedrooms','listed_area_sqm']:
        shuffled = Xt.copy()
        shuffled[column] = rng.permutation(shuffled[column].to_numpy())
        loss = mean_absolute_percentage_error(y[test], fitted.predict(shuffled))
        diagnostics.append({'feature':column,'mape_increase_points':100*(loss-base)})
importance = pd.DataFrame(diagnostics).groupby('feature').mape_increase_points.agg(['mean','std']).sort_values('mean',ascending=False)
display(importance.round(2))
importance.to_csv(FIG / 'permutation_checks.csv')
print('These are out-of-fold shuffle checks, not causal effects. Correlated features can share importance.')
""")

md("""
## Part 4: Prediction failures

The five records below have the largest **absolute dollar** out-of-fold errors from the selected model. I use held-out predictions for each case, so these are more informative than errors after fitting on the same record. The source link is included for checking the listed features. These cases do not reveal every cause: renovation, outlook, building quality, strata levies and sale conditions are not in the dataset.
""")

code("""
oof_table = df[['property_id','address','suburb','property_type_listed','bedrooms','bathrooms','car_spaces','listed_area_sqm','sale_date','sale_price_aud','listing_url','results_page_url']].copy()
oof_table['predicted_aud'] = best_oof
oof_table['error_aud'] = best_oof - y
oof_table['abs_error_aud'] = abs(oof_table.error_aud)
oof_table['ape_pct'] = 100 * oof_table.abs_error_aud / y
worst = oof_table.nlargest(5,'abs_error_aud')
display(worst[['property_id','address','sale_price_aud','predicted_aud','error_aud','ape_pct','listed_area_sqm']].round(1))
oof_table.to_csv(FIG / 'oof_predictions.csv', index=False)
worst.to_csv(FIG / 'worst_predictions.csv', index=False)
""")

code("""
error_by_suburb = oof_table.groupby('suburb').agg(n=('ape_pct','size'),median_ape_pct=('ape_pct','median'),mean_ape_pct=('ape_pct','mean'),mae_aud=('abs_error_aud','mean'))
display(error_by_suburb.round(2))
error_by_suburb.to_csv(FIG / 'error_by_suburb.csv')
""")

md("""
## Part 5: Deployment and reflection

The Streamlit app loads the same fitted pipeline used above. It accepts one property or a batch CSV. Its estimate includes an empirical 80% error band based on absolute log residuals from cross-validation. This is a rough historical error band, not a formal prediction interval. It warns when a value is outside the observed training range or the property type has few comparables. The app does not extrapolate safely to other suburbs.
""")

code("""
final_model = make_model(best_name).fit(X, y)
log_residual = np.abs(np.log(y) - np.log(best_oof))
error_factor_80 = float(np.exp(np.quantile(log_residual, 0.8)))
bundle = {
    'model': final_model, 'model_name': best_name, 'metrics': results.loc[best_name].to_dict(),
    'error_factor_80': error_factor_80, 'trained_rows':len(df),
    'suburbs':['Blacktown','Parramatta','Mosman'],
    'types':sorted(df.property_type_listed.unique().tolist()),
    'min_date':str(df.sale_date.min().date()), 'max_date':str(df.sale_date.max().date()),
    'ranges':{column:[float(df[column].min()),float(df[column].max())] for column in ['bedrooms','bathrooms','car_spaces','listed_area_sqm']},
    'segment_counts':{f'{a}|{b}':int(n) for (a,b),n in df.groupby(['suburb','property_type_listed']).size().items()},
}
(ROOT/'app').mkdir(exist_ok=True)
joblib.dump(bundle, ROOT/'app'/'model.joblib')
print('Saved app/model.joblib; historical 80% multiplicative factor:', round(error_factor_80,2))
print('App command: streamlit run app/streamlit_app.py')
""")

md("""
The main lesson was that data provenance changes the result: a model can look convincing when trained on invented rows, but it says little about the actual market. This revision uses public sales with row-level source trails. The price range and different dwelling mix make evaluation difficult; percentage and dollar errors tell different stories. The random-fold score describes this small 2026 snapshot, not a future market. With more time, I would collect more sale dates and comparable houses in Parramatta, verify area definitions from each listing, add structured outlook and condition fields, and reserve later sales as a genuine forward-time test. Predictions should be treated cautiously for rare, unusual or luxury properties.
""")

book = nbf.v4.new_notebook(cells=cells, metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
path = ROOT / 'SIT307_8.1D_Sydney_Housing.ipynb'
nbf.write(book, path)
print(path)
