import pandas as pd


def build_eda_context() -> str:
    X_train = pd.read_csv("Data/X_train.csv")
    y_train = pd.read_csv("Data/y_train.csv")
    df = pd.concat([X_train, y_train], axis=1)

    numeric_cols = [
        "bedrooms", "bathrooms", "sqft_living", "sqft_lot",
        "floors", "sqft_above", "sqft_basement", "house_age", "price"
    ]

    stats    = df[numeric_cols].describe()
    corr     = df[numeric_cols].corr()["price"].drop("price").sort_values(ascending=False)
    wf_mean  = df.groupby("waterfront")["price"].mean()
    wf_premium  = wf_mean.get(1, 0) - wf_mean.get(0, 0)
    reno_mean   = df.groupby("has_been_renovated")["price"].mean()
    reno_lift   = reno_mean.get(1, 0) - reno_mean.get(0, 0)
    view_mean   = df.groupby("view")["price"].mean().to_dict()
    cond_mean   = df.groupby("condition")["price"].mean().to_dict()
    top_cities  = (
        df.groupby("city")["price"].median()
        .sort_values(ascending=False).head(5).to_dict()
    )

    context = f"""
=== HOUSE PRICE EDA KNOWLEDGE BASE ===

--- Dataset Overview ---
Total training samples: {len(df)}
Price range: ${stats['price']['min']:,.0f} – ${stats['price']['max']:,.0f}
Average price: ${stats['price']['mean']:,.0f}
Median price: ${stats['price']['50%']:,.0f}
Price std deviation: ${stats['price']['std']:,.0f}

--- Key Feature Statistics ---
Bedrooms: avg {stats['bedrooms']['mean']:.1f}, range {stats['bedrooms']['min']:.0f}–{stats['bedrooms']['max']:.0f}
Bathrooms: avg {stats['bathrooms']['mean']:.1f}, range {stats['bathrooms']['min']:.0f}–{stats['bathrooms']['max']:.0f}
Living Area (sqft): avg {stats['sqft_living']['mean']:,.0f}, range {stats['sqft_living']['min']:,.0f}–{stats['sqft_living']['max']:,.0f}
Lot Area (sqft): avg {stats['sqft_lot']['mean']:,.0f}
House Age: avg {stats['house_age']['mean']:.0f} years, range {stats['house_age']['min']:.0f}–{stats['house_age']['max']:.0f} years

--- Feature Correlations with Price (Pearson) ---
{chr(10).join(f"  {feat}: {val:.3f}" for feat, val in corr.items())}

--- Market Insights ---
Waterfront premium: ${wf_premium:,.0f} above non-waterfront homes
Renovation lift: ${reno_lift:,.0f} above non-renovated homes
~{reno_lift / stats['price']['mean'] * 100:.1f}% price change from renovation

--- View Rating Impact on Price ---
{chr(10).join(f"  View {k}: avg ${v:,.0f}" for k, v in sorted(view_mean.items()))}

--- Condition Rating Impact on Price ---
{chr(10).join(f"  Condition {k}: avg ${v:,.0f}" for k, v in sorted(cond_mean.items()))}

--- Top 5 Cities by Median Price ---
{chr(10).join(f"  {city}: ${price:,.0f}" for city, price in top_cities.items())}

--- Pricing Tiers ---
Budget (bottom 25%): below ${stats['price']['25%']:,.0f}
Mid-range (25–75%): ${stats['price']['25%']:,.0f} – ${stats['price']['75%']:,.0f}
Premium (top 25%): above ${stats['price']['75%']:,.0f}
"""
    return context.strip()


# Cached at module load
EDA_CONTEXT = build_eda_context()
