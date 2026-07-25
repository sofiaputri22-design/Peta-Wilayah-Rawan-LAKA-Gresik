import streamlit as st
import pandas as pd
import numpy as np
import folium
import osmnx as ox
import geopandas as gpd
from streamlit_folium import st_folium
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from collections import Counter
from shapely.ops import unary_union

st.set_page_config(page_title='Road Safety Intelligence', layout='wide')

# Header Identik dengan Colab
st.markdown('''
<div style="background: linear-gradient(135deg, #00c6ff, #0072ff, #2ecc71); padding: 25px; border-radius: 10px; color: white; text-align: center; margin-bottom: 20px;">
    <h1 style="margin: 0;">Road Safety Intelligence</h1>
    <p style="margin: 5px 0 0 0;">Dashboard Identifikasi Wilayah Rawan Laka Lantas Kabupaten Gresik</p>
</div>
''', unsafe_allow_html=True)

@st.cache_data
def load_and_process_data():
    df = pd.read_csv('Dataset Clustering 5.csv', sep=';', decimal=',', encoding='latin1')
    df = df.loc[:, ~df.columns.str.contains('Unnamed', case=False)]
    df.columns = df.columns.str.lower()
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str).str.strip().str.lower()

    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longititude'] = pd.to_numeric(df['longititude'], errors='coerce')
    df = df.dropna(subset=['latitude', 'longititude'])

    # MODELING & VARIABEL IDENTIK DENGAN COLAB
    feat = ['latitude', 'longititude', 'jumlah korban meninggal dunia', 'jumlah korban luka berat', 'jumlah korban luka ringan']
    X = df[feat].copy()
    X_scaled = MinMaxScaler().fit_transform(X)

    # K-Means (n=3, rs=42)
    kmeans = KMeans(n_clusters=3, random_state=42).fit(X_scaled)
    df['cluster'] = kmeans.labels_

    cluster_means = df.groupby('cluster')['jumlah korban meninggal dunia'].mean().sort_values()
    urutan = cluster_means.index.tolist()
    mapping = {urutan[0]: 'rendah', urutan[1]: 'sedang', urutan[2]: 'tinggi'}
    df['kategori'] = df['cluster'].map(mapping)
    return df.reset_index().rename(columns={'index': 'original_row_id'})

@st.cache_resource
def get_osm_network():
    road_types = ['trunk', 'primary', 'secondary', 'tertiary']
    cf = '["highway"~"' + '|'.join(road_types) + '"]'
    graph = ox.graph_from_place('Kabupaten Gresik, Indonesia', network_type='drive', custom_filter=cf)
    _, edges = ox.graph_to_gdfs(graph)
    edges = edges.reset_index()

    def clean_osm_name(name_val):
        if isinstance(name_val, list): return str(name_val[0]).upper()
        if pd.isna(name_val) or str(name_val).lower() in ['nan', 'none', 'unnamed']: return None
        return str(name_val).upper()

    edges['clean_name'] = edges['name'].apply(clean_osm_name)
    return edges

try:
    df = load_and_process_data()
    edges_final = get_osm_network()

    # Spatial Join with logic for clean names and original_row_id
    gdf_acc = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['longititude'], df['latitude']), crs='EPSG:4326').to_crs(epsg=32749)
    roads_proj = edges_final.to_crs(epsg=32749)
    joined = gpd.sjoin_nearest(gdf_acc, roads_proj, distance_col='dist')
    joined = joined.sort_values('dist').drop_duplicates(subset=['original_row_id'])

    def finalize_road_name(row):
        osm_n = edges_final.loc[row['index_right'], 'clean_name']
        return osm_n if osm_n else str(row['lokasi jalan']).upper()

    joined['final_road_name'] = joined.apply(finalize_road_name, axis=1)
    joined['kecamatan_upper'] = joined['kecamatan'].str.upper()

    # Agregasi untuk Peta (road_summary logic)
    segment_stats = joined.groupby(['index_right', 'final_road_name', 'kecamatan_upper']).agg({
        'kategori': list, 
        'original_row_id': 'count'
    }).reset_index()

    def get_dominant(cat_list):
        counts = Counter(cat_list)
        return counts.most_common(1)[0][0]

    segment_stats['dominant_kategori'] = segment_stats['kategori'].apply(get_dominant)

    # road_summary grouped by finalized names for neat objects
    road_summary = segment_stats.groupby(['final_road_name', 'kecamatan_upper', 'dominant_kategori']).agg({
        'index_right': lambda x: list(set(x))
    }).reset_index()

    st.sidebar.header('Filter Dashboard')
    selected_kec = st.sidebar.selectbox('🏙️ Pilih Kecamatan', ['SEMUA KECAMATAN'] + sorted(road_summary['kecamatan_upper'].unique()))

    filtered_df = road_summary.copy()
    if selected_kec != 'SEMUA KECAMATAN':
        filtered_df = filtered_df[filtered_df['kecamatan_upper'] == selected_kec]

    m = folium.Map(location=[df['latitude'].mean(), df['longititude'].mean()], zoom_start=11)
    palette = {'tinggi': '#D32F2F', 'sedang': '#FBC02D', 'rendah': '#388E3C'}

    for _, row in filtered_df.iterrows():
        segment_ids = row['index_right']
        # Union all segments with the same name into one object
        combined_geom = edges_final.loc[edges_final.index.isin(segment_ids)].geometry.unary_union
        color = palette.get(row['dominant_kategori'])
        risk_label = row['dominant_kategori'].upper()

        popup_html = f"""<div style='font-family: Arial; width: 220px; border: 1px solid {color}; border-radius: 8px; overflow:hidden;'><div style='background-color:{color}; color:white; padding:8px; text-align:center;'><b>RAWAN {risk_label}</b></div><div style='padding:10px; background: white;'><b>JALAN:</b> {row['final_road_name']}<br><b>KECAMATAN:</b> {row['kecamatan_upper']}</div></div>"""

        folium.GeoJson(
            combined_geom,
            style_function=lambda x, c=color: {'color': c, 'weight': 7, 'opacity': 0.8},
            tooltip=f"{row['final_road_name']} ({risk_label})",
            popup=folium.Popup(popup_html, max_width=300)
        ).add_to(m)

    st_folium(m, width=1200, height=600)

    st.markdown('''
    <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; border: 1px solid #ddd; display: flex; justify-content: center; gap: 30px; font-weight: bold;">
        <span style="color:#D32F2F">🔴 Rawan Tinggi</span>
        <span style="color:#FBC02D">🟡 Rawan Sedang</span>
        <span style="color:#388E3C">🟢 Rawan Rendah</span>
    </div>
    ''', unsafe_allow_html=True)

except Exception as e:
    st.error(f'Terjadi Kesalahan: {e}')