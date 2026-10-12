import pandas as pd
import numpy as np
import os
import joblib
from sklearn.cluster import KMeans

def haversine_np(lon1, lat1, lon2, lat2):
    """Calculate the great circle distance between two points on the earth in km."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km

def run_advanced_pipeline(data_dir="data", output_path="data/processed_emissions_v2.csv", model_dir="models"):
    print("Loading raw Olist datasets...")
    # Load core data
    orders = pd.read_csv(os.path.join(data_dir, "olist_orders_dataset.csv"))
    items = pd.read_csv(os.path.join(data_dir, "olist_order_items_dataset.csv"))
    products = pd.read_csv(os.path.join(data_dir, "olist_products_dataset.csv"))
    customers = pd.read_csv(os.path.join(data_dir, "olist_customers_dataset.csv"))
    sellers = pd.read_csv(os.path.join(data_dir, "olist_sellers_dataset.csv"))
    geo = pd.read_csv(os.path.join(data_dir, "olist_geolocation_dataset.csv"))

    # Deduplicate coordinates
    print("Aggregating geospatial data...")
    geo_agg = geo.groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]].mean().reset_index()

    # Join coordinates
    sellers = sellers.merge(geo_agg, left_on="seller_zip_code_prefix", right_on="geolocation_zip_code_prefix", how="inner").rename(columns={"geolocation_lat": "seller_lat", "geolocation_lng": "seller_lng"})
    customers = customers.merge(geo_agg, left_on="customer_zip_code_prefix", right_on="geolocation_zip_code_prefix", how="inner").rename(columns={"geolocation_lat": "cust_lat", "geolocation_lng": "cust_lng"})

    # Filter and merge relational tables
    print("Executing relational joins...")
    merged = orders[orders["order_status"] == "delivered"].merge(items, on="order_id", how="inner")
    merged = merged.merge(products, on="product_id", how="inner")
    merged = merged.merge(sellers[["seller_id", "seller_state", "seller_lat", "seller_lng"]], on="seller_id", how="inner")
    merged = merged.merge(customers[["customer_id", "customer_state", "cust_lat", "cust_lng"]], on="customer_id", how="inner")

    # Clean missing values and downsample for performance
    merged = merged.dropna(subset=["product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"])
    merged = merged.sample(n=35000, random_state=42).reset_index(drop=True)

    # --- K-MEANS SPATIAL CLUSTERING ---
    print("Training K-Means Spatial Clustering for Green Hubs...")
    # We cluster customer locations to find where regional distribution hubs should be placed
    coords = merged[["cust_lat", "cust_lng"]]
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    merged["hub_cluster_id"] = kmeans.fit_predict(coords)
    
    # Save the K-Means model for the UI Map
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(kmeans, os.path.join(model_dir, "kmeans_hubs.pkl"))

    # --- ADVANCED LOGISTICS METRICS ---
    print("Computing Haversine transit distances and multi-modal emissions...")
    merged["distance_km"] = haversine_np(merged["seller_lng"], merged["seller_lat"], merged["cust_lng"], merged["cust_lat"])
    merged["distance_km"] = merged["distance_km"] * 1.25 # Road circuity factor
    merged = merged[merged["distance_km"] > 5.0]

    merged["weight_tonnes"] = (merged["product_weight_g"] / 1000.0) / 1000.0
    merged["volume_cm3"] = merged["product_length_cm"] * merged["product_height_cm"] * merged["product_width_cm"]
    merged["tonne_km"] = merged["weight_tonnes"] * merged["distance_km"]

    # GLEC Multi-Modal Emission Target Calculations
    merged["co2_road_kg"] = np.round((merged["tonne_km"] * 0.102) + 0.15, 3)
    merged["co2_rail_kg"] = np.round((merged["tonne_km"] * 0.030) + 0.15, 3)
    merged["co2_sea_kg"] = np.round((merged["tonne_km"] * 0.015) + 0.15, 3)

    # Time calculation
    merged["order_purchase_timestamp"] = pd.to_datetime(merged["order_purchase_timestamp"])
    merged["order_delivered_customer_date"] = pd.to_datetime(merged["order_delivered_customer_date"])
    merged["actual_lead_time_days"] = np.round((merged["order_delivered_customer_date"] - merged["order_purchase_timestamp"]).dt.total_seconds() / 86400.0, 1)

    # Final feature selection
    feature_cols = [
        "price", "freight_value", "product_category_name", 
        "weight_tonnes", "volume_cm3", "distance_km", 
        "seller_state", "customer_state", "actual_lead_time_days",
        "cust_lat", "cust_lng", "hub_cluster_id", 
        "co2_road_kg", "co2_rail_kg", "co2_sea_kg"
    ]
    
    final_df = merged[feature_cols].dropna().reset_index(drop=True)
    final_df.to_csv(output_path, index=False)
    print(f"\nSuccess! V2 Dataset saved to: {output_path}")
    print(f"K-Means Spatial model saved to: {os.path.join(model_dir, 'kmeans_hubs.pkl')}")

if __name__ == "__main__":
    run_advanced_pipeline()