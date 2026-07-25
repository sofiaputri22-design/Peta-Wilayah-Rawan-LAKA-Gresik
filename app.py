import streamlit as st
import pandas as pd
import folium
import osmnx as ox
import geopandas as gpd
from streamlit_folium import st_folium
from collections import Counter

st.set_page_config(page_title='Road Safety Intelligence', layout='wide')

st.markdown('''
<div style='background: linear-gradient(135deg, #00c6ff, #0072ff, #2ecc71); padding: 35px; border-radius: 12px 12px 0 0; color: white; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.3);'>
    <h1 style='margin: 0; font-family: "Segoe UI"; font-weight: 800; font-size: 2.5em; text-transform: uppercase; letter-spacing: 4px;'>
        ⛡️ Road Safety Intelligence
    </h1>
    <p style='margin: 10px 0 0 0; font-size: 1.2em; opacity: 0.95;'>
        Dashboard Identifikasi Wilayah Rawan Laka Lantas Kabupaten Gresik
    </p>
</div>
''', unsafe_allow_html=True)

@st.cache_data
def load_data():
    df_load = pd.read_csv('Dataset_Clustering_Final.csv', sep=';', decimal=',')
    df_load['kecamatan_upper'] = df_load['kecamatan'].str.upper()
    return df_load.reset_index().rename(columns={'index': 'original_row_id'})

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
    df_app = load_data()
    edges_final = get_osm_network()
    gdf_acc = gpd.GeoDataFrame(df_app, geometry=gpd.points_from_xy(df_app['longititude'], df_app['latitude']), crs='EPSG:4326').to_crs(epsg=32749)
    roads_proj = edges_final.to_crs(epsg=32749)
    joined = gpd.sjoin_nearest(gdf_acc, roads_proj, distance_col='dist').sort_values('dist').drop_duplicates(subset=['original_row_id'])

    def finalize_road_name(row):
        osm_n = edges_final.loc[row['index_right'], 'clean_name']
        return osm_n if osm_n and str(osm_n).lower() != 'nan' else str(row['lokasi jalan']).upper()
    joined['final_road_name'] = joined.apply(finalize_road_name, axis=1)

    def get_majority_color(row):
        cat_list = row['kategori']
        if any(x in str(row['final_road_name']).upper() for x in ['WAHIDIN', 'SUDIROHUSODO']):
            return 'sedang'
        if not cat_list: return 'rendah'
        counts = Counter(cat_list)
        return counts.most_common(1)[0][0]

    segment_stats = joined.groupby(['index_right', 'final_road_name', 'kecamatan_upper']).agg({'kategori': list}).reset_index()
    segment_stats['dominant_kategori'] = segment_stats.apply(get_majority_color, axis=1)
    road_summary = segment_stats.groupby(['final_road_name', 'kecamatan_upper', 'dominant_kategori']).agg({'index_right': lambda x: list(set(x))}).reset_index()

    st.sidebar.header('Filter Dashboard')
    selected_kec = st.sidebar.selectbox('🏙️ Kecamatan:', ['SEMUA KECAMATAN'] + sorted(road_summary['kecamatan_upper'].unique()))
    filtered = road_summary if selected_kec == 'SEMUA KECAMATAN' else road_summary[road_summary['kecamatan_upper'] == selected_kec]

    m = folium.Map(location=[df_app['latitude'].mean(), df_app['longititude'].mean()], zoom_start=11)
    palette = {'tinggi': '#D32F2F', 'sedang': '#FBC02D', 'rendah': '#388E3C'}

    for _, row in filtered.iterrows():
        combined_geom = edges_final.loc[edges_final.index.isin(row['index_right'])].geometry.union_all()
        color = palette.get(row['dominant_kategori'], '#95a5a6')
        risk_label = row['dominant_kategori'].upper()
        pop_html = f"<div style='font-family: Arial; width: 220px; border: 1px solid {color}; border-radius: 8px; overflow:hidden;'><div style='background-color:{color}; color:white; padding:8px; text-align:center;'><b>RAWAN {risk_label}</b></div><div style='padding:10px; background: white;'><b>JALAN:</b> {row['final_road_name']}<br><b>KECAMATAN:</b> {row['kecamatan_upper']}</div></div>"
        folium.GeoJson(combined_geom, style_function=lambda x, c=color: {'color': c, 'weight': 7, 'opacity': 0.8}, tooltip=f\"{row['final_road_name']} ({risk_label})\", popup=folium.Popup(pop_html, max_width=300)).add_to(m)

    st_folium(m, width=1200, height=600)

    legend_html = f\"\"\"<div style='background-color: #f8f9fa; padding: 15px; border-radius: 0 0 12px 12px; border: 1px solid #ddd; display: flex; justify-content: center; gap: 30px; font-family: Arial; font-size: 14px;'>
        <div style='display: flex; align-items: center; gap: 8px;'><div style='width: 30px; height: 12px; background-color: {palette['tinggi']}; border-radius: 2px;'></div> <b>Rawan Tinggi</b></div>
        <div style='display: flex; align-items: center; gap: 8px;'><div style='width: 30px; height: 12px; background-color: {palette['sedang']}; border-radius: 2px;'></div> <b>Rawan Sedang</b></div>
        <div style='display: flex; align-items: center; gap: 8px;'><div style='width: 30px; height: 12px; background-color: {palette['rendah']}; border-radius: 2px;'></div> <b>Rawan Rendah</b></div>
    </div>\"\"\"
    st.markdown(legend_html, unsafe_allow_html=True)

except Exception as e:
    st.error(f'Error: {e}')
