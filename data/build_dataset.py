import numpy as np
import pandas as pd

RNG = np.random.default_rng(307)
N_PER_SUBURB = 40


SUBURBS = {
    "Blacktown": dict(
        postcode=2148, cbd_km=34.0,

        median={"House": 900_000, "Townhouse": 675_000, "Apartment": 520_000},

        type_p=[0.62, 0.23, 0.15],
        land=(520, 0.28), internal=(150, 0.22),
        built=(1978, 16),
        view_p=[0.86, 0.13, 0.01, 0.00, 0.00],
        pool_p=0.10, auction_p=0.35,
    ),
    "Parramatta": dict(
        postcode=2150, cbd_km=23.0,
        median={"House": 1_480_000, "Townhouse": 1_005_000, "Apartment": 650_000},
        type_p=[0.34, 0.14, 0.52],
        land=(450, 0.30), internal=(135, 0.30),
        built=(1992, 22),
        view_p=[0.74, 0.22, 0.03, 0.01, 0.00],
        pool_p=0.08, auction_p=0.55,
    ),
    "Mosman": dict(
        postcode=2088, cbd_km=8.0,
        median={"House": 5_200_000, "Townhouse": 2_700_000, "Apartment": 1_560_000},
        type_p=[0.48, 0.10, 0.42],
        land=(620, 0.34), internal=(210, 0.34),
        built=(1948, 28),
        view_p=[0.34, 0.24, 0.21, 0.16, 0.05],
        pool_p=0.30, auction_p=0.72,
    ),
}

TYPES = ["House", "Townhouse", "Apartment"]


VIEWS = ["None", "District", "Water glimpses", "Harbour views", "Waterfront"]
VIEW_EFFECT = {"None": 0.00, "District": 0.04, "Water glimpses": 0.12,
               "Harbour views": 0.28, "Waterfront": 0.55}

CONDITIONS = ["Needs work", "Original", "Good", "Renovated", "New"]
CONDITION_P = [0.08, 0.24, 0.36, 0.24, 0.08]
CONDITION_EFFECT = {"Needs work": -0.13, "Original": -0.05, "Good": 0.00,
                    "Renovated": 0.06, "New": 0.10}


TYPICAL = {
    "House":     dict(beds=4, baths=2, cars=2),
    "Townhouse": dict(beds=3, baths=2, cars=1),
    "Apartment": dict(beds=2, baths=1, cars=1),
}


def draw_config(ptype):
    if ptype == "House":
        beds = int(RNG.choice([2, 3, 4, 5, 6], p=[0.04, 0.26, 0.40, 0.24, 0.06]))
        baths = int(np.clip(RNG.choice([1, 2, 3, 4], p=[0.18, 0.44, 0.29, 0.09]), 1, beds))
        cars = int(RNG.choice([0, 1, 2, 3, 4], p=[0.03, 0.20, 0.50, 0.20, 0.07]))
    elif ptype == "Townhouse":
        beds = int(RNG.choice([2, 3, 4], p=[0.22, 0.58, 0.20]))
        baths = int(RNG.choice([1, 2, 3], p=[0.20, 0.62, 0.18]))
        cars = int(RNG.choice([1, 2], p=[0.55, 0.45]))
    else:
        beds = int(RNG.choice([1, 2, 3], p=[0.22, 0.55, 0.23]))
        baths = int(RNG.choice([1, 2, 3], p=[0.48, 0.45, 0.07]))
        cars = int(RNG.choice([0, 1, 2], p=[0.15, 0.62, 0.23]))
    return beds, baths, cars


def draw_areas(ptype, cfg, beds):
    land_mu, land_cv = cfg["land"]
    int_mu, int_cv = cfg["internal"]

    if ptype == "House":
        land = land_mu * np.exp(RNG.normal(0, land_cv)) * (0.86 + 0.07 * beds)
        internal = int_mu * np.exp(RNG.normal(0, int_cv)) * (0.70 + 0.10 * beds)
    elif ptype == "Townhouse":
        land = 0.42 * land_mu * np.exp(RNG.normal(0, land_cv))
        internal = 0.78 * int_mu * np.exp(RNG.normal(0, int_cv)) * (0.80 + 0.08 * beds)
    else:
        land = np.nan
        internal = 0.52 * int_mu * np.exp(RNG.normal(0, int_cv)) * (0.72 + 0.16 * beds)

    return (round(land) if land == land else np.nan), round(internal)


DESC_OPEN = {
    "House": [
        "Deceased estate offered to the market for the first time in decades.",
        "Immaculately presented family home in a quiet pocket.",
        "Rare opportunity in one of the area's most tightly held streets.",
        "Sun-drenched residence set on a level block.",
        "Solid brick home with scope to add value over time.",
        "Spacious family home offering flexible living zones.",
    ],
    "Townhouse": [
        "Low-maintenance townhouse in a small, well-kept complex.",
        "Modern three-level townhouse with double garage.",
        "Beautifully updated townhouse ready to move straight into.",
        "Rarely offered townhouse in a boutique group of six.",
    ],
    "Apartment": [
        "Positioned in a boutique security block with lift access.",
        "Bright, generously proportioned apartment with a full-width balcony.",
        "Immaculate apartment in a well-maintained, established building.",
        "Top-floor apartment offering privacy and an open outlook.",
    ],
}

DESC_VIEW = {
    "None": ["A private, leafy outlook from the rear living area."],
    "District": ["Elevated with pleasant district outlook.",
                 "Leafy district views from the main living zone."],
    "Water glimpses": ["Water glimpses from the upper level.",
                       "Enjoy filtered water glimpses through the treetops."],
    "Harbour views": ["Sweeping harbour views across to the city skyline.",
                      "Commanding harbour views from every principal room."],
    "Waterfront": ["An absolute waterfront position with deep water access and jetty.",
                   "Direct waterfront with private jetty and boatshed."],
}

DESC_COND = {
    "Needs work": ["Requires full renovation throughout - bring your builder.",
                   "Original condition and in need of significant work."],
    "Original": ["Retains original features and awaits a cosmetic update."],
    "Good": ["Well maintained and presented in good order throughout."],
    "Renovated": ["Renovated throughout with a designer kitchen and new bathrooms.",
                  "Recently renovated with quality finishes."],
    "New": ["Brand new and never lived in, with full builder's warranty.",
            "Newly completed with premium appliances and finishes."],
}

DESC_DEV = ["Significant development potential (STCA) on this large parcel.",
            "DA approved plans available for a duplex development (STCA)."]

DESC_CLOSE = {
    "Blacktown": [
        "Walk to Westpoint, schools and Blacktown station.",
        "Close to local schools, parklands and the M7 on-ramp.",
        "Convenient to shops, childcare and express city trains.",
    ],
    "Parramatta": [
        "Moments to Parramatta Square, the light rail and the station.",
        "An easy walk to Church Street dining and the river foreshore.",
        "Close to Westfield, the university campus and ferry wharf.",
    ],
    "Mosman": [
        "An easy stroll to Balmoral Beach and the ferry.",
        "Walk to Spit Junction village, cafes and city buses.",
        "Close to quality schools, Mosman village and the harbour foreshore.",
    ],
}


def build():
    rows = []
    for suburb, cfg in SUBURBS.items():
        for _ in range(N_PER_SUBURB):
            ptype = str(RNG.choice(TYPES, p=cfg["type_p"]))
            beds, baths, cars = draw_config(ptype)
            land, internal = draw_areas(ptype, cfg, beds)

            condition = str(RNG.choice(CONDITIONS, p=CONDITION_P))
            view = str(RNG.choice(VIEWS, p=cfg["view_p"]))

            year_built = int(np.clip(RNG.normal(*cfg["built"]), 1895, 2026))
            if condition == "New":
                year_built = int(RNG.integers(2023, 2027))


            dev = bool(land == land and land > 700 and RNG.random() < 0.30)


            month_offset = int(RNG.integers(0, 32))
            sale_date = pd.Timestamp("2024-01-15") + pd.Timedelta(days=30 * month_offset)
            sale_date += pd.Timedelta(days=int(RNG.integers(0, 28)))
            years_since = (sale_date - pd.Timestamp("2024-01-01")).days / 365.25

            has_pool = bool(ptype != "Apartment" and RNG.random() < cfg["pool_p"])
            auction = RNG.random() < cfg["auction_p"]
            station_km = round(float(np.clip(RNG.gamma(2.0, 0.45), 0.15, 4.5)), 2)
            cbd_km = round(cfg["cbd_km"] + float(RNG.normal(0, 0.9)), 1)


            typ = TYPICAL[ptype]
            ref_internal = {"House": 175, "Townhouse": 135, "Apartment": 95}[ptype]
            ref_land = {"House": 540, "Townhouse": 210}.get(ptype, np.nan)

            log_p = np.log(cfg["median"][ptype])
            log_p += 0.110 * (beds - typ["beds"])
            log_p += 0.070 * (baths - typ["baths"])
            log_p += 0.035 * (cars - typ["cars"])
            log_p += 0.30 * np.log(internal / ref_internal)
            if land == land:
                log_p += 0.18 * np.log(land / ref_land)
            log_p += CONDITION_EFFECT[condition]
            log_p += VIEW_EFFECT[view]
            log_p += 0.09 if dev else 0.0
            log_p += 0.045 if has_pool else 0.0
            log_p += 0.055 * years_since
            log_p -= 0.0016 * np.clip(sale_date.year - year_built, 0, 90)
            log_p -= 0.012 * station_km
            log_p += RNG.normal(0, 0.085)

            price = float(np.exp(log_p))

            price = round(price / 5_000) * 5_000

            days_on_market = int(np.clip(RNG.gamma(2.2, 13), 3, 160))

            desc = " ".join([
                str(RNG.choice(DESC_OPEN[ptype])),
                f"{beds} bedrooms, {baths} bathrooms.",
                str(RNG.choice(DESC_VIEW[view])),
                str(RNG.choice(DESC_COND[condition])),
                *( [str(RNG.choice(DESC_DEV))] if dev else [] ),
                str(RNG.choice(DESC_CLOSE[suburb])),
            ])

            rows.append(dict(
                suburb=suburb,
                postcode=cfg["postcode"],
                property_type=ptype,
                bedrooms=beds,
                bathrooms=baths,
                car_spaces=cars,
                land_size_sqm=land,
                internal_area_sqm=internal,
                year_built=year_built,
                condition=condition,
                has_pool=has_pool,
                distance_to_cbd_km=cbd_km,
                distance_to_station_km=station_km,
                sale_method="Auction" if auction else "Private treaty",
                sale_date=sale_date.date().isoformat(),
                days_on_market=days_on_market,
                agent_description=desc,
                sale_price=price,
            ))

    df = pd.DataFrame(rows)


    def blank(col, frac, mask=None):
        idx = df.index if mask is None else df.index[mask]
        n = int(round(frac * len(idx)))
        if n:
            df.loc[RNG.choice(idx, size=n, replace=False), col] = np.nan

    blank("land_size_sqm", 0.08, df["land_size_sqm"].notna())
    blank("internal_area_sqm", 0.12)
    blank("year_built", 0.18)
    blank("car_spaces", 0.06)
    blank("days_on_market", 0.10)

    df = df.sample(frac=1, random_state=307).reset_index(drop=True)
    df.insert(0, "property_id", [f"P{i:03d}" for i in range(1, len(df) + 1)])
    return df


if __name__ == "__main__":
    data = build()
    data.to_csv("sydney_housing.csv", index=False)
    print(f"wrote sydney_housing.csv  rows={len(data)}  cols={data.shape[1]}")
    print(data.groupby("suburb")["sale_price"].agg(["count", "median", "min", "max"]))
