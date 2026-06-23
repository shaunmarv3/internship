"""
Folium map builders for all dashboard views.
Each function returns a folium.Map that Streamlit renders with st.components.
"""

import folium
import pandas as pd
import geopandas as gpd
from shapely.geometry import mapping
import json


RISK_COLORS = {"HIGH": "#d32f2f", "MEDIUM": "#f57c00", "LOW": "#388e3c"}


def dark_fleet_map(vessel_df: pd.DataFrame, center: list = [0.0, 0.0], zoom: int = 4) -> folium.Map:
    """
    Module 1 output: normal vessels (blue), dark vessels (red), AIS anomalies (orange).
    """
    m = folium.Map(location=center, zoom_start=zoom, tiles="CartoDB dark_matter")

    for _, row in vessel_df.iterrows():
        if not pd.notna(row.get("lat")) or not pd.notna(row.get("lon")):
            continue
        if row.get("dark_vessel", False):
            color, icon_name, label = "red", "eye-slash", "DARK"
        else:
            color, icon_name, label = "blue", "ship", "AIS"

        popup_html = f"""
        <b>{label} Vessel</b><br>
        MMSI: {row.get('matched_mmsi', 'N/A')}<br>
        Type: {row.get('vessel_type', 'unknown')}<br>
        Risk: <span style='color:{RISK_COLORS.get(row.get('risk_label','LOW'), 'green')}'>{row.get('risk_label', 'LOW')}</span>
        """
        folium.Marker(
            location=[row["lat"], row["lon"]],
            popup=folium.Popup(popup_html, max_width=220),
            icon=folium.Icon(color=color, icon="ship", prefix="fa"),
        ).add_to(m)

    return m


def illegal_fishing_map(vessel_df: pd.DataFrame, mpa_gdf: gpd.GeoDataFrame = None,
                         center: list = [0.0, 0.0], zoom: int = 4) -> folium.Map:
    """Module 2: legal (green), suspected illegal (red), zone overlays."""
    m = folium.Map(location=center, zoom_start=zoom, tiles="CartoDB positron")

    if mpa_gdf is not None:
        folium.GeoJson(
            mpa_gdf.to_crs("EPSG:4326").__geo_interface__,
            style_function=lambda f: {"fillColor": "#1565c0", "color": "#1565c0",
                                       "weight": 1, "fillOpacity": 0.15},
            tooltip="MPA",
        ).add_to(m)

    for _, row in vessel_df.iterrows():
        if not pd.notna(row.get("lat")) or not pd.notna(row.get("lon")):
            continue
        illegal = row.get("suspected_illegal_fishing", False)
        color = "red" if illegal else "green"
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=5,
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=f"MMSI: {row.get('matched_mmsi','?')} | Illegal: {illegal}",
        ).add_to(m)

    return m


def oil_spill_map(spill_gdf: gpd.GeoDataFrame = None,
                   vessel_df: pd.DataFrame = None,
                   center: list = [36.0, 14.0], zoom: int = 6) -> folium.Map:
    """Module 3: spill polygons (severity-coloured) + candidate vessels."""
    m = folium.Map(location=center, zoom_start=zoom, tiles="CartoDB dark_matter")

    if spill_gdf is not None and not spill_gdf.empty:
        for _, row in spill_gdf.iterrows():
            severity = row.get("severity", 0.5)
            fill_color = f"#ff{int((1-severity)*200):02x}00"
            folium.GeoJson(
                mapping(row.geometry),
                style_function=lambda f, fc=fill_color: {
                    "fillColor": fc, "color": "#ff6600",
                    "weight": 2, "fillOpacity": 0.5,
                },
                tooltip=f"Oil spill | Severity: {severity:.2f}",
            ).add_to(m)

    if vessel_df is not None:
        for _, v in vessel_df[vessel_df.get("dark_vessel", False) == True].iterrows():
            if pd.notna(v.get("lat")) and pd.notna(v.get("lon")):
                folium.Marker(
                    location=[v["lat"], v["lon"]],
                    icon=folium.Icon(color="red", icon="exclamation-triangle", prefix="fa"),
                    popup="Candidate responsible vessel (dark)",
                ).add_to(m)

    return m


def drift_forecast_map(
    drift_points: pd.DataFrame,   # columns: lon, lat, hour (0..N)
    spill_origin: tuple = None,
    center: list = [36.0, 14.0],
    zoom: int = 6,
) -> folium.Map:
    """Module 4: animated-ish drift trajectory as polyline + time markers."""
    m = folium.Map(location=center, zoom_start=zoom, tiles="CartoDB dark_matter")

    if drift_points.empty:
        return m

    if spill_origin:
        folium.Marker(
            location=[spill_origin[1], spill_origin[0]],
            icon=folium.Icon(color="orange", icon="fire", prefix="fa"),
            popup="Spill origin",
        ).add_to(m)

    coords = [[r["lat"], r["lon"]] for _, r in drift_points.iterrows()]
    folium.PolyLine(coords, color="#ff6600", weight=3, tooltip="Predicted drift path").add_to(m)

    # mark every 6-hour interval
    for _, row in drift_points[drift_points["hour"] % 6 == 0].iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=5,
            color="#ffcc00",
            fill=True,
            tooltip=f"T+{int(row['hour'])}h",
        ).add_to(m)

    return m


def risk_dashboard_map(risk_df: pd.DataFrame, center: list = [0.0, 0.0], zoom: int = 4) -> folium.Map:
    """Module 6: all vessels colour-coded by risk_label (HIGH/MEDIUM/LOW)."""
    m = folium.Map(location=center, zoom_start=zoom, tiles="CartoDB dark_matter")

    for _, row in risk_df.iterrows():
        if not pd.notna(row.get("lat")) or not pd.notna(row.get("lon")):
            continue
        color = RISK_COLORS.get(row.get("risk_label", "LOW"), "#388e3c")
        popup_html = f"""
        <b>Risk: {row.get('risk_label')}</b><br>
        Security: {row.get('security_risk', 0):.0f}/100<br>
        Environmental: {row.get('env_risk', 0):.0f}/100<br>
        Dark: {row.get('dark_vessel', False)}<br>
        MPA: {row.get('in_mpa', False)}
        """
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=8,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            popup=folium.Popup(popup_html, max_width=200),
        ).add_to(m)

    # legend
    legend = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;background:rgba(0,0,0,0.7);
                padding:10px;border-radius:5px;color:white;font-size:13px;">
    <b>Risk Level</b><br>
    <span style="color:#d32f2f">&#9679;</span> HIGH<br>
    <span style="color:#f57c00">&#9679;</span> MEDIUM<br>
    <span style="color:#388e3c">&#9679;</span> LOW
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend))
    return m
