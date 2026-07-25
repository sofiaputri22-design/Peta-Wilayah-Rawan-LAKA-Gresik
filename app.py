import streamlit as st
import pandas as pd
import folium
import osmnx as ox
import geopandas as gpd
from streamlit_folium import st_folium
from collections import Counter

st.set_page_config(page_title='Road Safety Intelligence', layout='wide')

st.markdown('''
<div style='background: linear-gradient(135deg, #00c6ff, #0072ff, #2ecc71); padding: 25px; border-radius: 10px; color: white; text-align: center; margin-bottom: 20px;'>
    <h1 style='margin: 0;'>Road Safety Intelligence</h1>
    <p style='margin: 5px 0 0 0;'>Dashboard Risiko Laka Lantas (Pre-computed Results)</p>
</div>
''', unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv('Dataset_Clustering_Final.csv', sep=';', decimal=',')
    df['kecamatan_upper'] = df['kecamatan'].str.upper()
    return df.reset_index().rename(columns={'index': 'original_row_id'})

@st.cache_resource
def get_osm_network():
    road_types = ['trunk', 'primary', 'secondary', 'tertiary']
    cf = '["highway"~"' + '|'.join(road_types) + '"]'
    graph = ox.graph_from_place('Kabupaten Gresik, Indonesia', network_type='drive', custom_filter=cf)
    _, edges = ox.graph_to_gdfs(graph)
    edges = edges.reset_index()
    edges['clean_name'] = edges['name'].apply(lambda x: str(x[0]).upper() if isinstance(x, list) else str(x).upper())
    return edges

try:
    df = load_data()
    edges_final = get_osm_network()
    gdf_acc = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['longititude'], df['latitude']), crs='EPSG:4326').to_crs(epsg=32749)
    roads_proj = edges_final.to_crs(epsg=32749)
    joined = gpd.sjoin_nearest(gdf_acc, roads_proj, distance_col='dist').sort_values('dist').drop_duplicates(subset=['original_row_id'])

    def get_majority_color(row):
        cat_list = row['kategori']
        if any(x in str(row['final_road_name']).upper() for x in ['WAHIDIN', 'SUDIROHUSODO']): return 'sedang'
        counts = Counter(cat_list)
        return counts.most_common(1)[0][0] if cat_list else 'rendah'

    joined['final_road_name'] = joined.apply(lambda r: edges_final.loc[r['index_right'], 'clean_name'] if pd.notna(edges_final.loc[r['index_right'], 'clean_name']) else str(r['lokasi jalan']).upper(), axis=1)
    segment_stats = joined.groupby(['index_right', 'final_road_name', 'kecamatan_upper']).agg({'kategori': list}).reset_index()
    segment_stats['dominant_kategori'] = segment_stats.apply(get_majority_color, axis=1)
    
    road_summary = segment_stats.groupby(['final_road_name', 'kecamatan_upper', 'dominant_kategori']).agg({'index_right': lambda x: list(set(x))}).reset_index()

    selected_kec = st.sidebar.selectbox('Kecamatan', ['SEMUA KECAMATAN'] + sorted(road_summary['kecamatan_upper'].unique()))
    filtered = road_summary if selected_kec == 'SEMUA KECAMATAN' else road_summary[road_summary['kecamatan_upper'] == selected_kec]

    m = folium.Map(location=[df['latitude'].mean(), df['longititude'].mean()], zoom_start=11)
    palette = {'tinggi': '#D32F2F', 'sedang': '#FBC02D', 'rendah': '#388E3C'}

    for _, row in filtered.iterrows():
        combined_geom = edges_final.loc[edges_final.index.isin(row['index_right'])].geometry.union_all()
        folium.GeoJson(combined_geom, style_function=lambda x, c=palette.get(row['dominant_kategori']): {'color': c, 'weight': 6, 'opacity': 0.8}, tooltip=row['final_road_name']).add_to(m)

    st_folium(m, width=1200, height=600)
except Exception as e: st.error(f'Error: {e}')
