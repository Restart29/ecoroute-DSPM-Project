"""
EcoRoute AI | Scope 3 Freight Decarbonization Cockpit
-----------------------------------------------------
Fully live version: every KPI, chart, table and memo line is computed at runtime
from data/processed_emissions_v2.csv and the trained artifacts in models/.
Nothing in the dashboard is hardcoded except the published emission factors
(GLEC v3.0 / ISO 14083 defaults) and the user-adjustable scenario assumptions.
"""
import os
import io
import json
from datetime import date

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

try:
    import shap
    SHAP_OK = True
except Exception:  # SHAP is optional; the rest of the app still works
    SHAP_OK = False

# -----------------------------------------------------------------------------
# Constants: emission factors (same ones used to build the training labels)
# -----------------------------------------------------------------------------
EF_ROAD = 0.102      # kg CO2e per tonne-km, diesel HGV (GLEC v3.0 default)
EF_RAIL = 0.030      # kg CO2e per tonne-km, intermodal rail
EF_SEA = 0.015       # kg CO2e per tonne-km, coastal cabotage
HANDLING_KG = 0.15   # fixed hub/terminal handling emission per consignment
CIRCUITY = 1.25      # road circuity multiplier applied to great-circle distance

FEATURES = [
    "price", "freight_value", "weight_tonnes", "volume_cm3", "distance_km",
    "actual_lead_time_days", "product_category_name", "seller_state",
    "customer_state", "hub_cluster_id",
]

CATEGORY_EN = {
    "cama_mesa_banho": "bed_bath_table", "beleza_saude": "health_beauty",
    "esporte_lazer": "sports_leisure", "moveis_decoracao": "furniture_decor",
    "informatica_acessorios": "computers_accessories", "utilidades_domesticas": "housewares",
    "relogios_presentes": "watches_gifts", "telefonia": "telephony",
    "ferramentas_jardim": "garden_tools", "automotivo": "auto", "brinquedos": "toys",
    "cool_stuff": "cool_stuff", "perfumaria": "perfumery", "bebes": "baby",
    "papelaria": "stationery", "eletronicos": "electronics", "pet_shop": "pet_shop",
    "fashion_bolsas_e_acessorios": "fashion_bags_accessories",
    "moveis_escritorio": "office_furniture", "consoles_games": "consoles_games",
    "malas_acessorios": "luggage_accessories",
    "construcao_ferramentas_construcao": "construction_tools",
    "eletrodomesticos": "home_appliances", "eletroportateis": "small_appliances",
    "instrumentos_musicais": "musical_instruments", "eletrodomesticos_2": "home_appliances_2",
}

# -----------------------------------------------------------------------------
# Page config & styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="EcoRoute AI | Scope 3 Decarbonization Cockpit",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { padding-top: 10px; padding-bottom: 10px; font-weight: 600; }
    .live-badge { display:inline-block; padding:2px 10px; border-radius:12px;
                  background:#00CC96; color:#0E1117; font-weight:700; font-size:0.8rem; }
</style>
""", unsafe_allow_html=True)


def _st_version():
    try:
        return tuple(int(x) for x in st.__version__.split(".")[:2])
    except Exception:
        return (1, 0)


NEW_WIDTH_API = _st_version() >= (1, 50)


def show_chart(fig):
    """Render a Plotly figure at full width on both old and new Streamlit versions."""
    if NEW_WIDTH_API:
        st.plotly_chart(fig, width="stretch")
    else:
        st.plotly_chart(fig, use_container_width=True)


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def fmt_cat(c):
    en = CATEGORY_EN.get(c)
    return f"{c} ({en})" if en and en != c else c


# -----------------------------------------------------------------------------
# Data & artifact loading
# -----------------------------------------------------------------------------
def find_data_file():
    for p in ["data/processed_emissions_v2.csv", "processed_emissions_v2.csv"]:
        if os.path.exists(p):
            return p
    return None


@st.cache_data(show_spinner="Loading shipment history...")
def load_data(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=["weight_tonnes", "distance_km", "co2_road_kg"]).copy()
    for c in ["product_category_name", "seller_state", "customer_state"]:
        df[c] = df[c].astype(str)
    df["hub_cluster_id"] = df["hub_cluster_id"].astype(int).astype(str)
    df["tonne_km"] = df["weight_tonnes"] * df["distance_km"]
    df["lane"] = df["seller_state"] + " ➔ " + df["customer_state"]
    return df


@st.cache_resource(show_spinner="Loading trained models...")
def load_all_artifacts():
    model_xgb = joblib.load("models/carbon_model.pkl")
    try:
        model_nn = joblib.load("models/nn_model.pkl")
    except Exception:
        model_nn = None
    try:
        explainer = joblib.load("models/shap_explainer.pkl")
    except Exception:
        explainer = None
    kmeans = joblib.load("models/kmeans_hubs.pkl")
    try:
        with open("models/benchmark_metrics.json", "r") as f:
            metrics = json.load(f)
    except Exception:
        metrics = None
    return model_xgb, model_nn, explainer, kmeans, metrics


data_path = find_data_file()
if data_path is None:
    st.error("Dataset not found. Put processed_emissions_v2.csv in data/ (run src/data.py) and redeploy.")
    st.stop()
df = load_data(data_path)

try:
    xgb_pipeline, nn_pipeline, shap_explainer, kmeans_model, benchmark_metrics = load_all_artifacts()
except Exception as e:
    st.error(f"Missing model artifacts in models/: {e}. Run `python src/build_artifacts.py` and redeploy.")
    st.stop()


@st.cache_data
def state_centroids(_df):
    return _df.groupby("customer_state")[["cust_lat", "cust_lng"]].mean()


@st.cache_data
def lane_table(_df):
    return _df.groupby(["seller_state", "customer_state"]).agg(
        n=("distance_km", "size"),
        distance_km=("distance_km", "median"),
        freight_value=("freight_value", "median"),
        lead_time=("actual_lead_time_days", "median"),
        co2_road=("co2_road_kg", "median"),
    )


@st.cache_data
def state_hub(_df):
    return _df.groupby("customer_state")["hub_cluster_id"].agg(lambda s: s.mode().iloc[0])


CENTROIDS = state_centroids(df)
LANES = lane_table(df)
STATE_HUB = state_hub(df)

# -----------------------------------------------------------------------------
# Sidebar: consignment inputs (auto-filled from real lane history)
# -----------------------------------------------------------------------------
st.sidebar.title("🚚 Consignment Inputs")
st.sidebar.caption("Pick a lane: distance, freight and lead time auto-fill from historical shipments on that lane. "
                   "Every value stays editable.")

origin_opts = df["seller_state"].value_counts().index.tolist()
dest_opts = df["customer_state"].value_counts().index.tolist()
cat_opts = df["product_category_name"].value_counts().index.tolist()

origin_state = st.sidebar.selectbox("Origin Hub (Seller State)", origin_opts,
                                    index=origin_opts.index("SP") if "SP" in origin_opts else 0)
dest_state = st.sidebar.selectbox("Destination Zone (Customer State)", dest_opts,
                                  index=dest_opts.index("RJ") if "RJ" in dest_opts else 0)
category = st.sidebar.selectbox("Commodity Classification", cat_opts, format_func=fmt_cat,
                                index=cat_opts.index("informatica_acessorios") if "informatica_acessorios" in cat_opts else 0)

# Lane benchmark (or great-circle fallback when the lane has no history)
if (origin_state, dest_state) in LANES.index:
    lane = LANES.loc[(origin_state, dest_state)]
    lane_n = int(lane["n"])
    lane_dist, lane_freight, lane_lead = float(lane["distance_km"]), float(lane["freight_value"]), float(lane["lead_time"])
    lane_source = f"median of {lane_n:,} historical shipments"
else:
    lane_n = 0
    if origin_state in CENTROIDS.index and dest_state in CENTROIDS.index:
        o, d = CENTROIDS.loc[origin_state], CENTROIDS.loc[dest_state]
        lane_dist = float(haversine_km(o.cust_lat, o.cust_lng, d.cust_lat, d.cust_lng)) * CIRCUITY
    else:
        lane_dist = float(df["distance_km"].median())
    lane_freight = float(df["freight_value"].median())
    lane_lead = float(df["actual_lead_time_days"].median())
    lane_source = "no history: Haversine × 1.25 estimate"


def clamp(v, lo, hi):
    return float(min(max(v, lo), hi))


lane_key = f"{origin_state}_{dest_state}"
st.sidebar.info(f"Lane {origin_state} ➔ {dest_state}: {lane_source}")

weight_kg = st.sidebar.slider("Cargo Weight (kg)", 0.5, 40.0, 12.5, 0.5)
volume_cm3 = st.sidebar.slider("Package Volume (cm³)", 500, 300000, 18500, 500)
distance_km = st.sidebar.slider("Road Distance (km)", 5.0, 7000.0, clamp(round(lane_dist / 5) * 5, 5, 7000), 5.0,
                                key=f"dist_{lane_key}")
item_price = st.sidebar.number_input("Commercial Invoice Value ($ USD)", 1.0, 5000.0, 145.0, 5.0)
freight_cost = st.sidebar.number_input("Carrier Freight Charge ($ USD)", 0.0, 600.0,
                                       clamp(round(lane_freight, 2), 0, 600), 0.5, key=f"freight_{lane_key}")
lead_time_days = st.sidebar.slider("Service SLA / Lead Time (Days)", 0.5, 60.0,
                                   clamp(round(lane_lead * 2) / 2, 0.5, 60), 0.5, key=f"lead_{lane_key}")

assigned_cluster = str(STATE_HUB.get(dest_state, "0"))
st.sidebar.success(f"Auto-assigned fulfilment hub: **Cluster {assigned_cluster}** (K-Means)")

with st.sidebar.expander("⚙️ Scenario assumptions", expanded=False):
    carbon_price = st.number_input("Carbon price ($ / tCO2e)", 0.0, 500.0, 75.0, 5.0)
    shift_threshold = st.slider("Modal-shift distance threshold (km)", 100, 2000, 700, 50)
    rail_cost_factor = st.slider("Rail cost vs road (×)", 0.50, 1.30, 0.88, 0.01)
    sea_cost_factor = st.slider("Sea cost vs road (×)", 0.40, 1.30, 0.76, 0.01)
    rail_extra_days = st.slider("Rail extra lead time (days)", 0.0, 10.0, 1.5, 0.5)
    sea_extra_days = st.slider("Sea extra lead time (days)", 0.0, 15.0, 3.0, 0.5)

input_df = pd.DataFrame({
    "price": [float(item_price)],
    "freight_value": [float(freight_cost)],
    "weight_tonnes": [float(weight_kg / 1000.0)],
    "volume_cm3": [float(volume_cm3)],
    "distance_km": [float(distance_km)],
    "actual_lead_time_days": [float(lead_time_days)],
    "product_category_name": [category],
    "seller_state": [origin_state],
    "customer_state": [dest_state],
    "hub_cluster_id": [assigned_cluster],
})

# -----------------------------------------------------------------------------
# Core live calculations (shared by all tabs)
# -----------------------------------------------------------------------------
tonne_km = (weight_kg / 1000.0) * distance_km
pred_road = float(xgb_pipeline.predict(input_df[FEATURES])[0])
pred_nn = None
if nn_pipeline is not None:
    try:
        pred_nn = float(nn_pipeline.predict(input_df[FEATURES])[0])
    except Exception:
        pred_nn = None
glec_road = tonne_km * EF_ROAD + HANDLING_KG
pred_rail = tonne_km * EF_RAIL + HANDLING_KG
pred_sea = tonne_km * EF_SEA + HANDLING_KG

reduction_rail = max(pred_road - pred_rail, 0.0)
reduction_sea = max(pred_road - pred_sea, 0.0)
pct_rail = reduction_rail / pred_road * 100 if pred_road > 0 else 0.0
pct_sea = reduction_sea / pred_road * 100 if pred_road > 0 else 0.0
cost_rail = freight_cost * rail_cost_factor
cost_sea = freight_cost * sea_cost_factor


@st.cache_data
def portfolio(_df, threshold, rail_factor, price):
    tot_road_t = _df["co2_road_kg"].sum() / 1000
    eligible = _df["distance_km"] > threshold
    shifted = np.where(eligible, _df["co2_rail_kg"], _df["co2_road_kg"])
    tot_shift_t = shifted.sum() / 1000
    return {
        "n": len(_df),
        "tot_road_t": tot_road_t,
        "tot_shift_t": tot_shift_t,
        "abate_t": tot_road_t - tot_shift_t,
        "abate_pct": (tot_road_t - tot_shift_t) / tot_road_t * 100 if tot_road_t else 0,
        "avg_kg": _df["co2_road_kg"].mean(),
        "intensity": _df["co2_road_kg"].sum() / _df["tonne_km"].sum(),
        "eligible_pct": eligible.mean() * 100,
        "tax_base": tot_road_t * price,
        "tax_saved": (tot_road_t - tot_shift_t) * price,
        "freight_total": _df["freight_value"].sum(),
        "freight_saved": _df.loc[eligible, "freight_value"].sum() * (1 - rail_factor),
    }


P = portfolio(df, shift_threshold, rail_cost_factor, carbon_price)

# -----------------------------------------------------------------------------
# Header & tabs
# -----------------------------------------------------------------------------
st.title("🌱 EcoRoute AI | Scope 3 Freight Decarbonization Platform")
st.markdown(
    f'<span class="live-badge">LIVE</span>&nbsp; Every figure below is computed from <b>{P["n"]:,}</b> real Olist '
    f'shipments and the trained XGBoost / MLP / K-Means models. Change any input on the left to recompute.',
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Executive Summary",
    "🔀 Modal Shift Optimizer",
    "📍 Spatial Hub Network (K-Means)",
    "🔬 Model Rigor & SHAP",
    "📑 Scope 3 Compliance Memo",
    "📦 Batch Scoring",
])

# =============================================================================
# TAB 1: EXECUTIVE SUMMARY (computed from the full dataset)
# =============================================================================
with tab1:
    st.subheader("Enterprise Scope 3 Logistics Footprint")
    st.caption(f"Historical portfolio of {P['n']:,} shipments. Abatement scenario: every shipment above "
               f"{shift_threshold} km moves from road to intermodal rail (threshold adjustable in ⚙️ Scenario assumptions).")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Network Road Emissions", f"{P['tot_road_t']:,.2f} tCO2e",
              delta=f"-{P['abate_pct']:.1f}% with modal shift", delta_color="inverse")
    c2.metric("Effective Carbon Intensity", f"{P['intensity']:.3f} kg/t-km",
              delta=f"{P['intensity'] - EF_ROAD:+.3f} vs GLEC line-haul {EF_ROAD}", delta_color="inverse",
              help="Includes the fixed 0.15 kg handling emission per parcel, which dominates for light parcels.")
    c3.metric(f"Carbon Tax Liability (@ ${carbon_price:.0f}/t)", f"${P['tax_base']:,.2f}",
              delta=f"-${P['tax_saved']:,.2f} avoidable", delta_color="inverse")
    c4.metric("Shipments Eligible for Modal Shift", f"{P['eligible_pct']:.1f}%",
              delta=f"> {shift_threshold} km", delta_color="off")

    st.markdown("##### 📈 Scale to your annual volume")
    annual_volume = st.number_input("Annual shipments to project", 1_000, 50_000_000, 1_000_000, 50_000)
    scale = annual_volume / P["n"]
    p1, p2, p3 = st.columns(3)
    p1.metric("Projected annual road emissions", f"{P['tot_road_t'] * scale:,.0f} tCO2e")
    p2.metric("Projected abatement from modal shift", f"{P['abate_t'] * scale:,.0f} tCO2e",
              delta=f"${P['tax_saved'] * scale:,.0f} carbon tax avoided", delta_color="normal")
    p3.metric("Projected freight savings", f"${P['freight_saved'] * scale:,.0f}",
              delta=f"{(1 - rail_cost_factor) * 100:.0f}% cheaper on shifted lanes", delta_color="normal")

    st.markdown("---")
    c_left, c_right = st.columns([3, 2])
    with c_left:
        top_n = st.slider("Corridors to show", 5, 15, 8)
        inter = df[df["seller_state"] != df["customer_state"]]
        lanes = (inter.groupby("lane")[["co2_road_kg", "co2_rail_kg", "co2_sea_kg"]].sum() / 1000)
        lanes = lanes.sort_values("co2_road_kg", ascending=False).head(top_n).reset_index()
        lanes.columns = ["Corridor", "Diesel Road (tCO2e)", "Intermodal Rail (tCO2e)", "Coastal Sea (tCO2e)"]
        fig_corridor = px.bar(
            lanes, x="Corridor", y=["Diesel Road (tCO2e)", "Intermodal Rail (tCO2e)", "Coastal Sea (tCO2e)"],
            barmode="group", color_discrete_sequence=["#EF553B", "#00CC96", "#636EFA"],
            title=f"Top {top_n} interstate corridors by road emissions: abatement potential per mode",
        )
        fig_corridor.update_layout(legend_title_text="Freight Mode", height=380, yaxis_title="tCO2e")
        show_chart(fig_corridor)

    with c_right:
        st.markdown("#### Capital Allocation & ROI Drivers (live)")
        top10_share = (inter.groupby("lane")["co2_road_kg"].sum().nlargest(10).sum() / df["co2_road_kg"].sum() * 100)
        long_share = df.loc[df["distance_km"] > shift_threshold, "co2_road_kg"].sum() / df["co2_road_kg"].sum() * 100
        st.markdown(f"""
* **Concentration:** the top 10 interstate corridors produce **{top10_share:.1f}%** of all road emissions, so a lane-level programme reaches most of the footprint.
* **Long-haul leverage:** shipments over {shift_threshold} km are **{P['eligible_pct']:.1f}%** of volume but **{long_share:.1f}%** of emissions.
* **Freight savings:** shifting eligible lanes to rail at {rail_cost_factor:.2f}× road cost saves **${P['freight_saved']:,.0f}** on this portfolio (**{P['freight_saved'] / P['freight_total'] * 100:.1f}%** of total freight spend).
* **Carbon-tax hedge:** at ${carbon_price:.0f}/t, the abatement avoids **${P['tax_saved']:,.2f}** here, about **${P['tax_saved'] * scale:,.0f}** at {annual_volume:,} shipments a year.
""")

# =============================================================================
# TAB 2: MODAL SHIFT OPTIMIZER (live model inference)
# =============================================================================
with tab2:
    st.subheader("Consignment Modal Arbitrage")
    st.caption(f"Lane {origin_state} ➔ {dest_state} | {fmt_cat(category)} | {weight_kg:.1f} kg | "
               f"{distance_km:,.0f} km | {tonne_km:.4f} t-km | Hub cluster {assigned_cluster}")

    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.markdown("### 🚛 Diesel Road")
        st.metric("XGBoost Predicted Emission", f"{pred_road:.3f} kg CO2e")
        st.caption(f"Cost: ${freight_cost:.2f} | Lead time: {lead_time_days:.1f} days")
    with col_m2:
        st.markdown("### 🚆 Intermodal Rail / EV")
        st.metric("Optimized Emission", f"{pred_rail:.3f} kg CO2e", delta=f"-{pct_rail:.1f}% CO2e", delta_color="inverse")
        st.caption(f"Cost: ${cost_rail:.2f} | Lead time: {lead_time_days + rail_extra_days:.1f} days")
    with col_m3:
        st.markdown("### 🚢 Coastal Cabotage (Sea)")
        st.metric("Optimized Emission", f"{pred_sea:.3f} kg CO2e", delta=f"-{pct_sea:.1f}% CO2e", delta_color="inverse")
        st.caption(f"Cost: ${cost_sea:.2f} | Lead time: {lead_time_days + sea_extra_days:.1f} days")

    # Model cross-check strip
    x1, x2, x3, x4 = st.columns(4)
    x1.metric("GLEC formula (road)", f"{glec_road:.3f} kg", help="0.102 × t-km + 0.15 kg handling")
    x2.metric("XGBoost vs formula", f"{(pred_road - glec_road):+.3f} kg")
    x3.metric("Neural Net (MLP) prediction", f"{pred_nn:.3f} kg" if pred_nn is not None else "n/a")
    if lane_n:
        x4.metric("Lane historical median", f"{float(LANES.loc[(origin_state, dest_state), 'co2_road']):.3f} kg",
                  help=f"Median road CO2e of {lane_n:,} past shipments on this lane")
    else:
        x4.metric("Lane historical median", "no history")

    st.markdown("---")
    chart_col, notes_col = st.columns([3, 2])
    with chart_col:
        modes = ["Diesel Road", "Intermodal Rail / EV", "Coastal Cabotage (Sea)"]
        fig_tradeoff = go.Figure()
        fig_tradeoff.add_trace(go.Bar(x=modes, y=[pred_road, pred_rail, pred_sea], name="Emissions (kg CO2e)",
                                      marker_color=["#EF553B", "#00CC96", "#636EFA"], yaxis="y1"))
        fig_tradeoff.add_trace(go.Scatter(x=modes, y=[freight_cost, cost_rail, cost_sea], name="Freight Cost ($)",
                                          marker_color="#FFA15A", yaxis="y2", mode="lines+markers", line=dict(width=3)))
        fig_tradeoff.update_layout(
            title="Carbon footprint vs freight cost by mode",
            yaxis=dict(title="Emissions (kg CO2e)"),
            yaxis2=dict(title="Freight Expense ($)", overlaying="y", side="right"),
            legend=dict(x=0.01, y=0.99), height=360,
        )
        show_chart(fig_tradeoff)

    with notes_col:
        st.markdown("#### Executive Recommendation Engine")
        if distance_km > shift_threshold and reduction_rail > 0:
            st.success(
                f"**Modal shift approved: Intermodal Rail.**\n\n"
                f"At {distance_km:,.0f} km (above the {shift_threshold} km threshold), rail cuts this shipment's emissions by "
                f"**{reduction_rail:.3f} kg CO2e ({pct_rail:.1f}%)** and saves **${freight_cost - cost_rail:.2f}** "
                f"in line-haul cost, for **+{rail_extra_days:.1f} days** of lead time."
            )
            if dest_state in {"BA", "PE", "CE", "RN", "PB", "AL", "SE", "MA", "PA", "AM", "AP", "ES", "RJ", "SC", "RS", "PR", "SP", "PI"}:
                st.info(f"🚢 Coastal destination: sea cabotage would cut **{pct_sea:.1f}%** if the customer accepts "
                        f"**+{sea_extra_days:.1f} days**.")
        else:
            st.warning(
                f"**Maintain direct road route.**\n\n"
                f"Below the {shift_threshold} km threshold ({distance_km:,.0f} km), transfer overhead and the extra "
                f"{rail_extra_days:.1f} days outweigh the {pct_rail:.1f}% carbon saving. Recommend road with EV last-mile delivery."
            )

# =============================================================================
# TAB 3: SPATIAL HUB NETWORK (K-Means)
# =============================================================================
with tab3:
    st.subheader("Spatial Network Optimization (K-Means Clustering)")
    centroids = np.asarray(kmeans_model.cluster_centers_)[:, :2]
    k = len(centroids)
    st.caption(f"{k} consolidation hubs learned from customer coordinates. The yellow hub serves the current consignment.")

    @st.cache_data
    def hub_stats(_df, cents):
        lat, lng = _df["cust_lat"].to_numpy(), _df["cust_lng"].to_numpy()
        dmat = np.stack([haversine_km(lat, lng, c[0], c[1]) for c in cents], axis=1) * CIRCUITY
        nearest = dmat.argmin(axis=1)
        to_hub = dmat[np.arange(len(_df)), nearest]
        out = pd.DataFrame({"hub": nearest, "to_hub_km": to_hub, "orig_km": _df["distance_km"].to_numpy(),
                            "co2": _df["co2_road_kg"].to_numpy()})
        table = out.groupby("hub").agg(Customers=("hub", "size"), Avg_last_leg_km=("to_hub_km", "mean"),
                                       Avg_current_km=("orig_km", "mean"), Road_tCO2e=("co2", "sum"))
        table["Road_tCO2e"] /= 1000
        reduction = (1 - out["to_hub_km"].mean() / out["orig_km"].mean()) * 100
        return table, reduction, out["to_hub_km"].mean(), out["orig_km"].mean()

    table, dist_reduction, avg_hub, avg_orig = hub_stats(df, centroids)

    sample_n = st.slider("Customer points on map", 200, 5000, 1000, 100)
    geo = df.sample(min(sample_n, len(df)), random_state=7)
    customer_df = pd.DataFrame({"lat": geo["cust_lat"], "lon": geo["cust_lng"], "color": "#636EFA", "size": 6000})
    hub_colors = ["#FFD700" if str(i) == assigned_cluster else "#EF553B" for i in range(k)]
    hub_df = pd.DataFrame({"lat": centroids[:, 0], "lon": centroids[:, 1], "color": hub_colors, "size": 60000})
    st.map(pd.concat([customer_df, hub_df], ignore_index=True), latitude="lat", longitude="lon",
           color="color", size="size", zoom=3.4)

    m1, m2, m3 = st.columns(3)
    m1.metric("Avg current seller→customer road km", f"{avg_orig:,.0f} km")
    m2.metric("Avg hub→customer last-leg km", f"{avg_hub:,.0f} km")
    m3.metric("Last-leg distance reduction", f"{dist_reduction:.1f}%",
              help="If inventory is pre-positioned at the nearest K-Means hub, the customer-facing leg shrinks by this much.")

    show = table.reset_index().rename(columns={"hub": "Hub", "Avg_last_leg_km": "Avg last-leg km",
                                               "Avg_current_km": "Avg current km", "Road_tCO2e": "Road tCO2e"})
    show.insert(1, "Lat", centroids[show["Hub"], 0].round(2))
    show.insert(2, "Lon", centroids[show["Hub"], 1].round(2))
    st.dataframe(show.round(2), hide_index=True)

# =============================================================================
# TAB 4: MODEL RIGOR & SHAP
# =============================================================================
with tab4:
    st.subheader("Model Diagnostics & Architecture Benchmarking")
    col_bench, col_shap = st.columns([1, 1])

    with col_bench:
        st.markdown("#### Dual-architecture benchmark (from training run)")
        if benchmark_metrics:
            def g(m, kname):
                try:
                    return benchmark_metrics[m][kname]
                except Exception:
                    return float("nan")
            bench_df = pd.DataFrame({
                "Evaluation Metric": ["R² Score", "RMSE (kg CO2e)", "MAE (kg CO2e)", "5-Fold CV R² (Mean)", "5-Fold CV R² (Std)"],
                "XGBoost Regressor": [f"{g('XGBoost', k):.4f}" for k in ["R2", "RMSE", "MAE", "CV_R2_Mean"]] + [f"± {g('XGBoost', 'CV_R2_Std'):.4f}"],
                "Neural Network (MLP)": [f"{g('Neural_Network', k):.4f}" for k in ["R2", "RMSE", "MAE", "CV_R2_Mean"]] + [f"± {g('Neural_Network', 'CV_R2_Std'):.4f}"],
            })
            st.table(bench_df)
        else:
            st.info("benchmark_metrics.json not found; showing the live check only.")

        st.markdown("#### Live re-scoring on historical shipments")
        n_check = st.select_slider("Shipments to re-score now", [500, 1000, 2000, 5000], value=2000)

        @st.cache_data
        def live_check(_df, n):
            s = _df.sample(min(n, len(_df)), random_state=42)
            y = s["co2_road_kg"].to_numpy()
            res = {}
            preds = {"XGBoost": xgb_pipeline.predict(s[FEATURES])}
            if nn_pipeline is not None:
                try:
                    preds["Neural Net"] = nn_pipeline.predict(s[FEATURES])
                except Exception:
                    pass
            for name, p in preds.items():
                p = np.asarray(p, dtype=float)
                ss_res = ((y - p) ** 2).sum()
                ss_tot = ((y - y.mean()) ** 2).sum()
                res[name] = {"R²": 1 - ss_res / ss_tot, "RMSE": np.sqrt(((y - p) ** 2).mean()), "MAE": np.abs(y - p).mean()}
            return pd.DataFrame(res).T, y, preds["XGBoost"]

        live_tbl, y_true, y_xgb = live_check(df, n_check)
        st.dataframe(live_tbl.round(4))
        st.caption("Re-scored live from the CSV. The sample can include training rows, so read it as a sanity check; "
                   "the holdout figures are in the table above.")
        fig_pa = px.scatter(x=y_true, y=y_xgb, opacity=0.4, labels={"x": "Actual road CO2e (kg)", "y": "XGBoost predicted (kg)"},
                            title="Predicted vs actual")
        mx = float(max(np.max(y_true), np.max(y_xgb)))
        fig_pa.add_shape(type="line", x0=0, y0=0, x1=mx, y1=mx, line=dict(color="#EF553B", dash="dash"))
        fig_pa.update_layout(height=320)
        show_chart(fig_pa)

    with col_shap:
        st.markdown("#### Local SHAP attribution (current consignment)")
        st.caption("Which inputs pushed this shipment's predicted road emissions up or down.")
        if not SHAP_OK or shap_explainer is None:
            st.info("SHAP not available in this deployment.")
        else:
            try:
                prep = xgb_pipeline.named_steps["prep"]
                transformed_input = prep.transform(input_df[FEATURES])
                if hasattr(transformed_input, "toarray"):
                    transformed_input = transformed_input.toarray()
                shap_vals = shap_explainer(transformed_input)
                clean_cols = [c.replace("num__", "").replace("cat__", "") for c in prep.get_feature_names_out()]
                plt.figure(figsize=(6, 4))
                shap.waterfall_plot(
                    shap.Explanation(values=shap_vals[0].values, base_values=shap_vals[0].base_values,
                                     data=transformed_input[0], feature_names=clean_cols),
                    max_display=8, show=False,
                )
                st.pyplot(plt.gcf())
                plt.close("all")
            except Exception as e:
                st.warning(f"SHAP rendering notice: {e}")

        st.markdown("""
**Pipeline design:**
1. **Features:** `StandardScaler` on continuous variables; `OneHotEncoder(handle_unknown="ignore")` on category, states and hub.
2. **Validation:** 5-fold shuffled K-Fold cross-validation plus a holdout test set.
3. **Regularisation:** tree depth capped at 6 to balance bias and variance.
""")

# =============================================================================
# TAB 5: COMPLIANCE MEMO (live)
# =============================================================================
with tab5:
    st.subheader("Scope 3 Category 4 Executive Audit Statement")
    st.caption("GHG Protocol-aligned logistics emissions disclosure, generated from the current inputs.")
    rail_tax = reduction_rail / 1000 * carbon_price
    memo_text = f"""
========================================================================================
SCOPE 3 CATEGORY 4 (UPSTREAM TRANSPORTATION & DISTRIBUTION) EMISSIONS MEMORANDUM
========================================================================================
DATE GENERATED: {date.today().isoformat()}
STANDARDS APPLIED: ISO 14083:2023 | GLEC Framework v3.0 (default emission factors)
STATUS: AUDIT-READY DRAFT, SUBJECT TO THIRD-PARTY VERIFICATION
----------------------------------------------------------------------------------------

1. CONSIGNMENT SPECIFICATIONS
   • Corridor:                 {origin_state} ➔ {dest_state}  ({lane_source})
   • Product classification:   {fmt_cat(category)}
   • Net cargo weight:         {weight_kg:.2f} kg ({weight_kg / 1000.0:.4f} t)
   • Road distance used:       {distance_km:,.1f} km
   • Transport activity:       {tonne_km:.4f} tonne-km
   • Assigned fulfilment hub:  K-Means cluster {assigned_cluster}

2. EMISSIONS BY MODE
   • Road (Diesel HGV), XGBoost:  {pred_road:.4f} kg CO2e
     (GLEC formula cross-check:   {glec_road:.4f} kg CO2e)
   • Intermodal Rail / EV:        {pred_rail:.4f} kg CO2e  (abatement -{pct_rail:.1f}%)
   • Coastal Cabotage (Sea):      {pred_sea:.4f} kg CO2e  (abatement -{pct_sea:.1f}%)

3. METHODOLOGY
   Road emissions are estimated by a gradient-boosted tree model trained on {P['n']:,}
   historical shipments labelled with GLEC default factors ({EF_ROAD} kg CO2e/t-km road,
   {EF_RAIL} rail, {EF_SEA} sea, plus {HANDLING_KG} kg handling per consignment).

4. FINANCIAL IMPACT (RAIL SCENARIO)
   • Line-haul cost change:       -${freight_cost - cost_rail:.2f} USD
   • Carbon cost avoided (@ ${carbon_price:.0f}/t): ${rail_tax:.4f} USD per shipment
   • At {annual_volume:,} similar shipments/yr: ${rail_tax * annual_volume:,.0f} USD

5. PORTFOLIO CONTEXT ({P['n']:,} shipments)
   • Total road emissions:        {P['tot_road_t']:,.2f} tCO2e
   • Abatement if > {shift_threshold} km moves to rail: {P['abate_t']:,.2f} tCO2e (-{P['abate_pct']:.1f}%)
========================================================================================
"""
    st.text_area("Audit Memo Output", value=memo_text, height=420)
    st.download_button("📥 Download Audit Memo (.txt)", data=memo_text,
                       file_name=f"Scope3_Memo_{origin_state}_{dest_state}_{date.today().isoformat()}.txt",
                       mime="text/plain")

# =============================================================================
# TAB 6: BATCH SCORING
# =============================================================================
with tab6:
    st.subheader("Batch Scoring: score a whole shipment file")
    st.caption("Upload a CSV with the 10 model input columns, or score a random sample of historical shipments.")

    template = df[FEATURES].sample(10, random_state=1)
    st.download_button("📄 Download CSV template (10 sample rows)", data=template.to_csv(index=False),
                       file_name="ecoroute_batch_template.csv", mime="text/csv")

    src = st.radio("Source", ["Random historical sample", "Upload my CSV"], horizontal=True)
    batch = None
    if src == "Upload my CSV":
        up = st.file_uploader("Shipments CSV", type=["csv"])
        if up is not None:
            batch = pd.read_csv(up)
            missing = [c for c in FEATURES if c not in batch.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
                batch = None
    else:
        n_batch = st.slider("Sample size", 50, 5000, 500, 50)
        seed = st.number_input("Random seed", 0, 9999, 2026)
        batch = df[FEATURES].sample(min(n_batch, len(df)), random_state=int(seed)).reset_index(drop=True)

    if batch is not None and len(batch):
        b = batch.copy()
        for c in ["product_category_name", "seller_state", "customer_state"]:
            b[c] = b[c].astype(str)
        b["hub_cluster_id"] = b["hub_cluster_id"].astype(str).str.replace(r"\.0$", "", regex=True)
        num_cols = ["price", "freight_value", "weight_tonnes", "volume_cm3", "distance_km", "actual_lead_time_days"]
        b[num_cols] = b[num_cols].apply(pd.to_numeric, errors="coerce")
        b = b.dropna(subset=num_cols).reset_index(drop=True)
        tkm = b["weight_tonnes"] * b["distance_km"]
        b["pred_road_kg"] = np.asarray(xgb_pipeline.predict(b[FEATURES]), dtype=float)
        b["rail_kg"] = tkm * EF_RAIL + HANDLING_KG
        b["sea_kg"] = tkm * EF_SEA + HANDLING_KG
        b["recommended_mode"] = np.where((b["distance_km"] > shift_threshold) & (b["pred_road_kg"] > b["rail_kg"]),
                                         "Intermodal Rail", "Road")
        b["optimised_kg"] = np.where(b["recommended_mode"] == "Road", b["pred_road_kg"], b["rail_kg"])
        b["co2_saved_kg"] = b["pred_road_kg"] - b["optimised_kg"]
        b["freight_saved_usd"] = np.where(b["recommended_mode"] == "Road", 0.0, b["freight_value"] * (1 - rail_cost_factor))

        tot_base = b["pred_road_kg"].sum()
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Shipments scored", f"{len(b):,}")
        k2.metric("Baseline road CO2e", f"{tot_base:,.1f} kg")
        k3.metric("CO2e saved", f"{b['co2_saved_kg'].sum():,.1f} kg",
                  delta=f"-{b['co2_saved_kg'].sum() / tot_base * 100:.1f}%" if tot_base else None, delta_color="inverse")
        k4.metric("Freight saved", f"${b['freight_saved_usd'].sum():,.2f}",
                  delta=f"{(b['recommended_mode'] != 'Road').mean() * 100:.0f}% shifted to rail", delta_color="off")

        fig_mix = px.histogram(b, x="distance_km", color="recommended_mode", nbins=40,
                               color_discrete_map={"Road": "#EF553B", "Intermodal Rail": "#00CC96"},
                               title="Recommended mode by distance")
        fig_mix.add_vline(x=shift_threshold, line_dash="dash", annotation_text=f"{shift_threshold} km threshold")
        fig_mix.update_layout(height=320)
        show_chart(fig_mix)

        st.dataframe(b.round(4).head(200), hide_index=True)
        st.download_button("📥 Download scored shipments (.csv)", data=b.to_csv(index=False),
                           file_name="ecoroute_scored_shipments.csv", mime="text/csv")
