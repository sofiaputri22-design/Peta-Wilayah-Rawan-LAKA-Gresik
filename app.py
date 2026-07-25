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

st.set_page_config(page_title='Road Safety Intelligence', layout='wide')

# Custom Header
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

    feat = ['latitude', 'longititude', 'jumlah korban meninggal dunia', 'jumlah korban luka berat', 'jumlah korban luka ringan']
    X = df[feat].copy()
    X_scaled = MinMaxScaler().fit_transform(X)
    kmeans = KMeans(n_clusters=3, random_state=42).fit(X_scaled)
    df['cluster'] = kmeans.labels_

    idx_sorted = df.groupby('cluster')['jumlah korban meninggal dunia'].mean().sort_values().index.tolist()
    mapping = {idx_sorted[0]: 'rendah', idx_sorted[1]: 'sedang', idx_sorted[2]: 'tinggi'}
    df['kategori'] = df['cluster'].map(mapping)
    return df

@st.cache_resource
def get_osm_network():
    cf = '["highway"~"trunk|primary|secondary|tertiary"]'
    graph = ox.graph_from_place('Kabupaten Gresik, Indonesia', network_type='drive', custom_filter=cf)
    _, edges = ox.graph_to_gdfs(graph)
    return edges.reset_index()

try:
    df = load_and_process_data()
    edges_final = get_osm_network()

    edges_final['clean_name'] = edges_final['name'].apply(lambda x: str(x[0]).upper() if isinstance(x, list) else (str(x).upper() if pd.notna(x) else 'JALAN TANPA NAMA'))

    gdf_acc = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['longititude'], df['latitude']), crs='EPSG:4326').to_crs(epsg=32749)
    roads_proj = edges_final.to_crs(epsg=32749)
    joined = gpd.sjoin_nearest(gdf_acc, roads_proj, distance_col='dist').drop_duplicates(subset=['latitude', 'longititude'])

    segment_stats = joined.groupby(['index_right', 'clean_name', 'kecamatan']).agg({'kategori': list}).reset_index()
    segment_stats['dominant_kategori'] = segment_stats['kategori'].apply(lambda x: Counter(x).most_common(1)[0][0])

    st.sidebar.header('Filter Dashboard')
    selected_kec = st.sidebar.selectbox('Pilih Kecamatan', ['SEMUA KECAMATAN'] + sorted(segment_stats['kecamatan'].str.upper().unique()))

    filtered_stats = segment_stats.copy()
    if selected_kec != 'SEMUA KECAMATAN':
        filtered_stats = filtered_stats[filtered_stats['kecamatan'].str.upper() == selected_kec]

    m = folium.Map(location=[df['latitude'].mean(), df['longititude'].mean()], zoom_start=11)
    palette = {'tinggi': '#D32F2F', 'sedang': '#FBC02D', 'rendah': '#388E3C'}

    for _, row in filtered_stats.iterrows():
        geom = edges_final.loc[edges_final.index == row['index_right']].geometry.iloc[0]
        color = palette.get(row['dominant_kategori'])
        
        popup_text = f"""<div style='font-family: Arial; width: 200px;'><div style='background-color:{color}; color:white; padding:5px; border-radius:3px; text-align:center;'><b>{row['dominant_kategori'].upper()}</b></div><div style='padding:5px;'><b>Jalan:</b> {row['clean_name']}<br><b>Kecamatan:</b> {row['kecamatan'].upper()}</div></div>"""
        
        folium.GeoJson(
            geom,
            style_function=lambda x, c=color: {'color': c, 'weight': 6, 'opacity': 0.8},
            tooltip=f"{row['clean_name']} ({row['dominant_kategori'].upper()})",
            popup=folium.Popup(popup_text, max_width=300)
        ).add_to(m)

    st_folium(m, width=1200, height=600)

    st.markdown('''
    <div style="display: flex; justify-content: center; gap: 20px;">
        <span style="color:#D32F2F">&#9679; Tinggi</span> <span style="color:#FBC02D">&#9679; Sedang</span> <span style="color:#388E3C">&#9679; Rendah</span>
    </div>
    ''', unsafe_allow_html=True)

except Exception as e:
    st.error(f'Error loading dashboard: {e}')