# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 19:33:49 2025

@author: Julian D. Sanchez
"""

import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import numpy as np
import os

# Configuración para matplotlib
plt.style.use('default')

# --------------------
# Funciones auxiliares optimizadas
# --------------------
@st.cache_data
def crear_mapa_conflictos_simple(lat_list, lon_list, datos_list, filtros_key):
    """Crea el mapa de conflictos con datos simplificados para mejor cache"""
    
    if not lat_list:  # Si no hay datos
        return folium.Map(location=[0, 0], zoom_start=2, tiles='CartoDB positron'), 0, 0
    
    # Calcular centro
    centro_lat = sum(lat_list) / len(lat_list)
    centro_lon = sum(lon_list) / len(lon_list)
    
    # Crear mapa base
    m = folium.Map(
        location=[centro_lat, centro_lon],
        zoom_start=2,
        tiles='CartoDB positron'
    )
    
    # Colores por tipo de violencia
    colores_tipos = {1: 'red', 2: 'blue', 3: 'green'}
    tipos_violencia = {
        1: "🏛️ Violencia Estatal",
        2: "⚔️ Violencia No Estatal", 
        3: "🎯 Violencia Unidireccional"
    }
    
    # Añadir puntos al mapa
    for i, datos in enumerate(datos_list):
        color = colores_tipos.get(datos['tipo'], 'gray')
        tamaño = min(max(datos['muertes'] / 10, 3), 15)
        
        popup_text = f"""
        <b>Conflicto:</b> {datos['conflicto']}<br>
        <b>País:</b> {datos['pais']}<br>
        <b>Año:</b> {datos['año']}<br>
        <b>Tipo:</b> {tipos_violencia.get(datos['tipo'], 'Desconocido')}<br>
        <b>Muertes:</b> {datos['muertes']}<br>
        <b>Región:</b> {datos['region']}
        """
        
        folium.CircleMarker(
            location=[lat_list[i], lon_list[i]],
            radius=tamaño,
            popup=folium.Popup(popup_text, max_width=300),
            color=color,
            fill=True,
            fillOpacity=0.7,
            weight=1
        ).add_to(m)
    
    # Añadir leyenda
    


    legend_html = '''
    <div style="position: fixed; top: 10px; right: 10px; z-index: 9999;
                background-color: rgba(255,255,255,0.98); padding: 15px;
                border: 2px solid #ccc; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.15);
                font-family: Arial, sans-serif; min-width: 180px;">
        <p style="margin: 0 0 10px 0; font-weight: bold; font-size: 14px; color: #333;">Tipos de Violencia:</p>
        <p style="margin: 8px 0; font-size: 13px; color: #333;"><i style="color:red; font-size: 1.3em;">●</i>&nbsp; Estatal</p>
        <p style="margin: 8px 0; font-size: 13px; color: #333;"><i style="color:blue; font-size: 1.3em;">●</i>&nbsp; No Estatal</p>
        <p style="margin: 8px 0; font-size: 13px; color: #333;"><i style="color:green; font-size: 1.3em;">●</i>&nbsp; Unidireccional</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m, len(datos_list), len(datos_list)

def crear_mapa_coropletico_simple(world_data, datos_muertes):
    """Crea el mapa coroplético de muertes"""
    # Convertir a WGS84 para Folium
    world_wgs84 = world_data.to_crs('EPSG:4326')
    
    # Preparar datos
    world_joined = world_wgs84[['name', 'geometry']].rename(columns={'name': 'country'})
    world_joined = world_joined.merge(datos_muertes[['country', 'total_deaths']], on='country', how='left')
    world_joined['total_deaths'] = world_joined['total_deaths'].fillna(0)
    
    # Aplicar escala logarítmica
    world_joined['deaths_log'] = np.log1p(world_joined['total_deaths'])  # log(1 + x)
    
    # Crear mapa base
    m_choro = folium.Map(location=[10, 0], zoom_start=2, tiles="CartoDB positron")
    
    # Choropleth con datos logarítmicos 
    folium.Choropleth(
        geo_data=world_joined.to_json(),
        name='choropleth',
        data=world_joined,
        columns=['country', 'deaths_log'],
        key_on='feature.properties.country',
        fill_color='YlOrRd',
        fill_opacity=0.8,
        line_opacity=0.4,
        nan_fill_color="lightgray",
        nan_fill_opacity=0.4,
        legend_name='Muertes totales por país',
        highlight=True
    ).add_to(m_choro)
    
    # Añadir popups interactivos
    tooltip = folium.features.GeoJsonTooltip(
        fields=['country', 'total_deaths'],
        aliases=['País:', 'Muertes:'],
        localize=True
    )
    
    # Layer transparente para tooltips
    folium.GeoJson(
        world_joined,
        style_function=lambda x: {
            'fillColor': '#ffffff',
            'color': '#000000',
            'fillOpacity': 0,
            'weight': 0.1
        },
        tooltip=tooltip
    ).add_to(m_choro)
    
    return m_choro

def crear_metrica_card(label, value):
    """Función helper para crear tarjetas métricas consistentes"""
    return f'''
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
    </div>
    '''

def mostrar_metricas_en_columnas(metricas_data):
    """Función helper para mostrar métricas en columnas"""
    cols = st.columns(len(metricas_data), gap="large")
    for i, (label, value) in enumerate(metricas_data):
        with cols[i]:
            st.markdown(crear_metrica_card(label, value), unsafe_allow_html=True)

# --------------------
# Configuración de página
# --------------------
st.set_page_config(
    page_title="Análisis de Conflictos Históricos",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------
# CSS personalizado 
# --------------------
st.markdown("""
<style>
    .main { padding-top: 2rem; }
    
    .main-title {
        font-size: 3.5rem !important;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .subtitle {
        font-size: 1.4rem;
        text-align: center;
        color: #4a5568;
        margin-bottom: 3rem;
        line-height: 1.6;
    }
    
    .project-description {
        font-size: 2rem;
        color: #2d3748;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    .description-text {
        font-size: 1.2rem;
        color: #4a5568;
        line-height: 1.7;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 2rem 1rem;
        text-align: center;
        color: white;
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        margin-bottom: 1rem;
        border: none;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.4);
    }
    
    .metric-label {
        font-size: 1rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
        opacity: 0.9;
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
    }
    
    .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }
</style>
""", unsafe_allow_html=True)

# --------------------
# Definir rutas
# --------------------
BASE_DIR = "."  # Directorio actual
DATA_DIR = BASE_DIR  # Los archivos estarán en la raíz
SHAPE_PATH = os.path.join(DATA_DIR, "naturalearth_lowres.zip")

# --------------------
# Carga de datos con caché
# --------------------
@st.cache_data
def load_data():
    url = "https://drive.google.com/uc?id=1u8CdHeCpwL64LuYYevz0QYvrasLqGyxr"
    df = pd.read_csv(url, parse_dates=['date_start', 'date_end'], low_memory=False)
    df['total_deaths'] = df['best'].clip(lower=0).fillna(0)
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.longitude, df.latitude),
        crs='EPSG:4326'
    )
    return gdf.to_crs(epsg=3395)

@st.cache_data
def load_world():
    world = gpd.read_file(SHAPE_PATH)
    world = world[world.name != 'Antarctica'].to_crs(epsg=3395)
    return world

# --------------------
# Constantes globales
# --------------------
TIPOS_VIOLENCIA = {
    1: "🏛️ Violencia Estatal",
    2: "⚔️ Violencia No Estatal", 
    3: "🎯 Violencia Unidireccional"
}

# --------------------
# Carga inicial
# --------------------
gdf = load_data()
world = load_world()

# --------------------
# Menú lateral
# --------------------
st.sidebar.markdown("## 🧭 Navegación")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Seleccione una página:",
    ["🏠 Introducción", "⚔️ Conflictos", "💀 Muertes", "📊 Análisis Colombia"],
    index=0
)

# --------------------
# Sección: Introducción
# --------------------
if page == "🏠 Introducción":
    # Título principal
    st.markdown('<h1 class="main-title">Análisis de Conflictos Históricos</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="main-title" style="font-size: 2rem; margin-top: -1rem;">(1989 - 2023)</h2>', unsafe_allow_html=True)
    
    # Subtítulo
    st.markdown(
        '<p class="subtitle">' +
        'Análisis interactivo basado en datos del <strong>Uppsala Conflict Data Program – ' +
        'Georeferenced Event Dataset (UCDP GED)</strong>, con información georreferenciada ' +
        'de conflictos armados a nivel mundial.' +
        '</p>',
        unsafe_allow_html=True
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Descripción del proyecto
    st.markdown(
        '<h3 class="project-description">🗂️ Detalle del Proyecto</h3>' +
        '<p class="description-text">' +
        'Esta página interactiva permite explorar patrones espaciales y temporales de conflictos armados, ' +
        'visualizar mapas, realizar análisis por décadas, examinar tipos de violencia y ' +
        'descubrir tendencias históricas en los datos de conflictos globales.' +
        '</p>',
        unsafe_allow_html=True
    )
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown('<h3 class="project-description">📋 Datos Utilizados</h3>', unsafe_allow_html=True)
    
    # Cálculo de métricas
    metricas_introduccion = [
        ("📋 Registros Totales", f"{len(gdf):,}"),
        ("🔢 Variables", f"{len(gdf.columns) - 1:,}"),
        ("⚔️ Conflictos Únicos", f"{gdf['conflict_name'].nunique():,}"),
        ("💀 Muertes Totales", f"{int(gdf['total_deaths'].sum()):,}"),
        ("🌎 Regiones", f"{gdf['region'].nunique():,}"),
        ("🏳️ Países", f"{gdf['country'].nunique():,}")
    ]
    
    # Primera fila de métricas
    mostrar_metricas_en_columnas(metricas_introduccion[:3])
    st.markdown("<br>", unsafe_allow_html=True)
    # Segunda fila de métricas
    mostrar_metricas_en_columnas(metricas_introduccion[3:])
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Información adicional
    st.markdown(
        '<div style="background-color: #f7fafc; padding: 2rem; border-radius: 10px; border-left: 5px solid #667eea;">' +
        '<h4 style="color: #2d3748; margin-bottom: 1rem;">💡 Cómo usar esta página:</h4>' +
        '<ul style="color: #4a5568; font-size: 1.1rem; line-height: 1.8;">' +
        '<li><strong>Conflictos:</strong> Explora conflictos en un mapa interactivo por tipo de violencia. </li>' +
        '<li><strong>Muertes:</strong> Muestra la cantidad de muertes por país y año. </li>' +
        '<li><strong>Análisis Colombia:</strong> Análisis detallado de conflictos en Colombia.</li>' +
        '</ul>' +
        '</div>',
        unsafe_allow_html=True
    )

elif page == "⚔️ Conflictos":
    # Título principal
    st.markdown('<h1 class="main-title">Análisis Global de Conflictos</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Exploración interactiva de patrones de conflictos armados por región, tipo y período temporal</p>', unsafe_allow_html=True)
    
    # Filtros en sidebar
    st.sidebar.markdown("## 🔧 Filtros de Análisis")
    st.sidebar.markdown("---")
    
    # Filtro por período temporal
    años_disponibles = sorted(gdf['year'].unique())
    año_min, año_max = st.sidebar.select_slider(
        "📅 Período temporal:",
        options=años_disponibles,
        value=(años_disponibles[0], años_disponibles[-1]),
        format_func=lambda x: str(x)
    )
    
    # Filtro por tipo de violencia
    tipos_seleccionados = st.sidebar.multiselect(
        "🔍 Tipo de violencia:",
        options=list(TIPOS_VIOLENCIA.keys()),
        default=list(TIPOS_VIOLENCIA.keys()),
        format_func=lambda x: TIPOS_VIOLENCIA[x]
    )
    
    # Filtro por región
    regiones_disponibles = sorted(gdf['region'].dropna().unique())
    regiones_seleccionadas = st.sidebar.multiselect(
        "🌍 Regiones:",
        options=regiones_disponibles,
        default=regiones_disponibles
    )
    
    # Aplicar filtros
    gdf_filtrado = gdf[
        (gdf['year'] >= año_min) & 
        (gdf['year'] <= año_max) &
        (gdf['type_of_violence'].isin(tipos_seleccionados)) &
        (gdf['region'].isin(regiones_seleccionadas))
    ].copy()
    
    # Key única basada en los filtros
    filtros_key = f"{año_min}_{año_max}_{'-'.join(map(str, tipos_seleccionados))}_{'-'.join(regiones_seleccionadas)}"
    
    # Verificar si hay datos después del filtrado
    if len(gdf_filtrado) == 0:
        st.warning("⚠️ No hay datos disponibles para los filtros seleccionados. Por favor, ajusta los criterios de búsqueda.")
        st.stop()
    
    # Métricas filtradas
    metricas_conflictos = [
        ("📊 Eventos Totales", f"{len(gdf_filtrado):,}"),
        ("⚔️ Conflictos Únicos", f"{gdf_filtrado['conflict_name'].nunique():,}"),
        ("🏳️ Países Afectados", f"{gdf_filtrado['country'].nunique():,}"),
        ("💀 Muertes Totales", f"{int(gdf_filtrado['total_deaths'].sum()):,}")
    ]
    
    mostrar_metricas_en_columnas(metricas_conflictos)
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Sección del mapa
    st.markdown('<h3 class="project-description">🗺️ Mapa Interactivo de Conflictos</h3>', unsafe_allow_html=True)
    
    # Control de cache del mapa
    if 'filtros_anteriores' not in st.session_state:
        st.session_state.filtros_anteriores = filtros_key
        st.session_state.mapa_cache = None
    
    filtros_cambiaron = st.session_state.filtros_anteriores != filtros_key
    
    # Solo recrear el mapa si los filtros cambiaron
    if filtros_cambiaron or st.session_state.mapa_cache is None:
        try:
            # Limitar datos para mejor rendimiento
            max_points = 1000
            if len(gdf_filtrado) > max_points:
                gdf_muestra = gdf_filtrado.groupby('type_of_violence').apply(
                    lambda x: x.sample(min(len(x), max_points // len(tipos_seleccionados)), random_state=42)
                ).reset_index(drop=True)
            else:
                gdf_muestra = gdf_filtrado.copy()
            
            # Convertir a WGS84 para Folium
            gdf_wgs84 = gdf_muestra.to_crs('EPSG:4326')
            
            # Preparar datos simplificados
            lat_list = gdf_wgs84.geometry.y.tolist()
            lon_list = gdf_wgs84.geometry.x.tolist()
            datos_list = []
            
            for _, row in gdf_wgs84.iterrows():
                datos_list.append({
                    'conflicto': row['conflict_name'],
                    'pais': row['country'],
                    'año': int(row['year']),
                    'tipo': row['type_of_violence'],
                    'muertes': int(row['total_deaths']),
                    'region': row['region']
                })
            
            # Crear mapa usando función cacheada
            mapa, _, _ = crear_mapa_conflictos_simple(lat_list, lon_list, datos_list, filtros_key)
            
            # Guardar en session_state
            st.session_state.mapa_cache = mapa
            st.session_state.filtros_anteriores = filtros_key
            st.session_state.puntos_en_mapa = len(gdf_muestra)
            
        except Exception as e:
            st.error(f"Error al cargar el mapa: {e}")
            st.session_state.mapa_cache = folium.Map(location=[0, 0], zoom_start=2, tiles='CartoDB positron')
            st.session_state.puntos_en_mapa = 0
    
    # Calcular altura dinámica del mapa
    
    if hasattr(st.session_state, 'puntos_en_mapa'):
        if st.session_state.puntos_en_mapa < 100:
            altura_mapa = 350
        elif st.session_state.puntos_en_mapa < 500:
            altura_mapa = 400
        else:
            altura_mapa = 450
    else:
        altura_mapa = 400
    
    # Mostrar el mapa desde el cache
    if st.session_state.mapa_cache is not None:
        with st.container():
            st_folium(
                st.session_state.mapa_cache, 
                width=None,
                height=altura_mapa,
                returned_objects=[],
                key=f"static_map_{filtros_key}"
            )
        

        st.markdown("""
        <style>
        iframe[src*="streamlit"][height="1621"] {
            height: 450px !important;
            max-height: 450px !important;
        }
        div[data-testid="stElementContainer"] iframe[src*="streamlit"] {
            height: 450px !important;
        }
        div[data-testid="stElementContainer"]:has(iframe[src*="streamlit"]) {
            height: 450px !important;
            overflow: hidden !important;
            margin-bottom: 10px !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
    else:
        st.error("No se pudo cargar el mapa")
    
    # Separador visual
    st.markdown('<div style="margin: 10px 0; border-bottom: 1px solid #e0e0e0; opacity: 0.5;"></div>', unsafe_allow_html=True)
    
    # Sección de análisis complementario
    st.markdown('<h3 class="project-description">📈 Análisis Complementario</h3>', unsafe_allow_html=True)
    
    # Crear tres columnas para las gráficas
    col_graf1, col_graf2, col_graf3 = st.columns(3, gap="large")
    
    with col_graf1:
        # Gráfica 1: Eventos por tipo de violencia
        st.markdown("**Distribución por Tipo de Violencia**")
        eventos_por_tipo = gdf_filtrado['type_of_violence'].value_counts()
        eventos_por_tipo.index = eventos_por_tipo.index.map(lambda x: TIPOS_VIOLENCIA.get(x, f'Tipo {x}'))
        
        fig_pie = plt.figure(figsize=(8, 6))
        colores = ['#ff6b6b', '#4ecdc4', '#45b7d1']
        plt.pie(eventos_por_tipo.values, labels=eventos_por_tipo.index, autopct='%1.1f%%', 
                colors=colores, startangle=90)
        plt.title('Distribución de Eventos por Tipo de Violencia')
        st.pyplot(fig_pie)
        plt.close()
    
    with col_graf2:
        # Gráfica 2: Top 10 países con más eventos
        st.markdown("**Top 10 Países con Más Eventos**")
        eventos_por_pais = gdf_filtrado['country'].value_counts().head(10)
        
        fig_bar = plt.figure(figsize=(8, 6))
        eventos_por_pais.plot(kind='barh', color='#667eea')
        plt.title('Top 10 Países con Más Eventos')
        plt.xlabel('Número de Eventos')
        plt.tight_layout()
        st.pyplot(fig_bar)
        plt.close()
    
    with col_graf3:
        # Gráfica 3: Evolución temporal
        st.markdown("**Evolución Temporal de Conflictos**")
        eventos_por_año = gdf_filtrado.groupby('year').size()
        
        fig_line = plt.figure(figsize=(8, 6))
        eventos_por_año.plot(kind='line', color='#764ba2', linewidth=2, marker='o')
        plt.title('Evolución Anual de Eventos')
        plt.xlabel('Año')
        plt.ylabel('Número de Eventos')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_line)
        plt.close()
    
    # Tabla resumen al final
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<h3 class="project-description">📋 Resumen Detallado</h3>', unsafe_allow_html=True)
    
    # Crear resumen por país
    resumen_pais = gdf_filtrado.groupby('country').agg({
        'id': 'count',
        'total_deaths': 'sum',
        'conflict_name': 'nunique',
        'region': 'first'
    }).round(0).reset_index()
    
    resumen_pais.columns = ['País', 'Eventos', 'Muertes Totales', 'Conflictos Únicos', 'Región']
    resumen_pais = resumen_pais.sort_values('Eventos', ascending=False).head(15)
    
    # Formatear números
    resumen_pais['Eventos'] = resumen_pais['Eventos'].apply(lambda x: f"{int(x):,}")
    resumen_pais['Muertes Totales'] = resumen_pais['Muertes Totales'].apply(lambda x: f"{int(x):,}")
    resumen_pais['Conflictos Únicos'] = resumen_pais['Conflictos Únicos'].apply(lambda x: f"{int(x):,}")
    
    st.dataframe(resumen_pais, use_container_width=True, hide_index=True)

elif page == "💀 Muertes":
    # Título principal
    st.markdown('<h1 class="main-title">Análisis Global de Muertes en Conflictos</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Visualización de la intensidad y distribución de víctimas mortales por región y período temporal</p>', unsafe_allow_html=True)
    
    # Filtros en sidebar
    st.sidebar.markdown("## 🔧 Filtros de Análisis")
    st.sidebar.markdown("---")
    
    # Filtro por período temporal
    años_disponibles = sorted(gdf['year'].unique())
    año_min_muertes, año_max_muertes = st.sidebar.select_slider(
        "📅 Período temporal:",
        options=años_disponibles,
        value=(años_disponibles[0], años_disponibles[-1]),
        format_func=lambda x: str(x),
        key="slider_muertes"
    )
    
    # Filtro por tipo de violencia
    tipos_seleccionados_muertes = st.sidebar.multiselect(
        "🔍 Tipo de violencia:",
        options=list(TIPOS_VIOLENCIA.keys()),
        default=list(TIPOS_VIOLENCIA.keys()),
        format_func=lambda x: TIPOS_VIOLENCIA[x],
        key="multi_tipos_muertes"
    )
    
    # Filtro por región
    regiones_disponibles = sorted(gdf['region'].dropna().unique())
    regiones_seleccionadas_muertes = st.sidebar.multiselect(
        "🌍 Regiones:",
        options=regiones_disponibles,
        default=regiones_disponibles,
        key="multi_regiones_muertes"
    )
    
    # Aplicar filtros
    gdf_filtrado_muertes = gdf[
        (gdf['year'] >= año_min_muertes) & 
        (gdf['year'] <= año_max_muertes) &
        (gdf['type_of_violence'].isin(tipos_seleccionados_muertes)) &
        (gdf['region'].isin(regiones_seleccionadas_muertes))
    ].copy()
    
    # Crear key única para cache
    filtros_key_muertes = f"muertes_{año_min_muertes}_{año_max_muertes}_{'-'.join(map(str, tipos_seleccionados_muertes))}_{'-'.join(regiones_seleccionadas_muertes)}"
    
    # Verificar si hay datos después del filtrado
    if len(gdf_filtrado_muertes) == 0:
        st.warning("⚠️ No hay datos disponibles para los filtros seleccionados. Por favor, ajusta los criterios de búsqueda.")
        st.stop()
    
    # Agregar datos por país para el mapa coroplético
    datos_por_pais = gdf_filtrado_muertes.groupby('country').agg({
        'total_deaths': 'sum',
        'id': 'count',
        'conflict_name': 'nunique'
    }).reset_index()
    datos_por_pais.columns = ['country', 'total_deaths', 'eventos', 'conflictos']
    
    # Sección del mapa coroplético
    st.markdown('<h3 class="project-description">🗺️ Mapa de Muertes por País</h3>', unsafe_allow_html=True)
    
    # Métricas globales
    metricas_muertes = [
        ("💀 Muertes Totales", f"{int(gdf_filtrado_muertes['total_deaths'].sum()):,}"),
        ("🏳️ Países Afectados", f"{len(datos_por_pais):,}"),
        ("📊 Promedio por País", f"{int(datos_por_pais['total_deaths'].mean()) if len(datos_por_pais) > 0 else 0:,}"),
        ("⚠️ Máximo por País", f"{int(datos_por_pais['total_deaths'].max()) if len(datos_por_pais) > 0 else 0:,}")
    ]
    
    mostrar_metricas_en_columnas(metricas_muertes)
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Crear mapa coroplético
    try:
        # Verificar cache del mapa de muertes
        if 'mapa_muertes_cache' not in st.session_state:
            st.session_state.mapa_muertes_cache = None
            st.session_state.filtros_muertes_anteriores = None
        
        filtros_muertes_cambiaron = st.session_state.filtros_muertes_anteriores != filtros_key_muertes
        
        if filtros_muertes_cambiaron or st.session_state.mapa_muertes_cache is None:
            mapa_coropletico = crear_mapa_coropletico_simple(world, datos_por_pais)
            st.session_state.mapa_muertes_cache = mapa_coropletico
            st.session_state.filtros_muertes_anteriores = filtros_key_muertes
        
        # Mostrar mapa con altura fija
        with st.container():
            st_folium(
                st.session_state.mapa_muertes_cache,
                width=None,
                height=500,
                returned_objects=[],
                key=f"choropleth_map_{filtros_key_muertes}"
            )
            
    except Exception as e:
        st.error(f"Error al cargar el mapa coroplético: {e}")
        st.info("💡 Intenta ajustar los filtros para obtener más datos.")
    
    # Separador visual
    st.markdown('<div style="margin: 20px 0; border-bottom: 2px solid #e0e0e0; opacity: 0.7;"></div>', unsafe_allow_html=True)
    
    # Sección de análisis complementario
    st.markdown('<h3 class="project-description">📈 Análisis Complementario de Muertes</h3>', unsafe_allow_html=True)
    
    # Crear tres columnas para las gráficas
    col_graf1, col_graf2, col_graf3 = st.columns(3, gap="large")
    
    with col_graf1:
        # Gráfica 1: Top 10 países con más muertes
        st.markdown("**Top 10 Países con Más Muertes**")
        top_paises_muertes = datos_por_pais.nlargest(10, 'total_deaths')
        
        fig_bar_muertes = plt.figure(figsize=(8, 6))
        plt.barh(top_paises_muertes['country'], top_paises_muertes['total_deaths'], color='#e74c3c')
        plt.title('Top 10 Países con Más Muertes')
        plt.xlabel('Número de Muertes')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig_bar_muertes)
        plt.close()
    
    with col_graf2:
        # Gráfica 2: Distribución de muertes por tipo de violencia
        st.markdown("**Muertes por Tipo de Violencia**")
        muertes_por_tipo = gdf_filtrado_muertes.groupby('type_of_violence')['total_deaths'].sum()
        muertes_por_tipo.index = muertes_por_tipo.index.map(lambda x: TIPOS_VIOLENCIA.get(x, f'Tipo {x}'))
        
        fig_pie_muertes = plt.figure(figsize=(8, 6))
        colores_muertes = ['#e74c3c', '#3498db', '#2ecc71']
        plt.pie(muertes_por_tipo.values, labels=muertes_por_tipo.index, autopct='%1.1f%%', 
                colors=colores_muertes, startangle=90)
        plt.title('Distribución de Muertes por Tipo de Violencia')
        st.pyplot(fig_pie_muertes)
        plt.close()
    
    with col_graf3:
        # Gráfica 3: Evolución temporal de muertes
        st.markdown("**Evolución Temporal de Muertes**")
        muertes_por_año = gdf_filtrado_muertes.groupby('year')['total_deaths'].sum()
        
        fig_line_muertes = plt.figure(figsize=(8, 6))
        muertes_por_año.plot(kind='line', color='#e74c3c', linewidth=2, marker='o')
        plt.title('Evolución Anual de Muertes')
        plt.xlabel('Año')
        plt.ylabel('Número de Muertes')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_line_muertes)
        plt.close()
    
    # Añadir una segunda fila de gráficas
    st.markdown("<br>", unsafe_allow_html=True)
    col_graf4, col_graf5 = st.columns(2, gap="large")
    
    with col_graf4:
        # Gráfica 4: Muertes por región
        st.markdown("**Muertes Totales por Región**")
        muertes_por_region = gdf_filtrado_muertes.groupby('region')['total_deaths'].sum().sort_values(ascending=False)
        
        fig_region = plt.figure(figsize=(10, 6))
        muertes_por_region.plot(kind='bar', color='#9b59b6', alpha=0.8)
        plt.title('Muertes Totales por Región')
        plt.xlabel('Región')
        plt.ylabel('Número de Muertes')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig_region)
        plt.close()
    
    with col_graf5:
        # Gráfica 5: Relación eventos vs muertes
        st.markdown("**Relación Eventos vs Muertes por País**")
        
        fig_scatter = plt.figure(figsize=(10, 6))
        plt.scatter(datos_por_pais['eventos'], datos_por_pais['total_deaths'], 
                   alpha=0.6, color='#f39c12', s=50)
        plt.xlabel('Número de Eventos')
        plt.ylabel('Número de Muertes')
        plt.title('Relación entre Eventos y Muertes')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_scatter)
        plt.close()
    
    # Tabla resumen detallada
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown('<h3 class="project-description">📋 Ranking Detallado por País</h3>', unsafe_allow_html=True)
    
    # Preparar tabla con métricas adicionales
    tabla_resumen = datos_por_pais.copy()
    tabla_resumen['promedio_muertes_evento'] = (tabla_resumen['total_deaths'] / tabla_resumen['eventos']).round(1)
    tabla_resumen = tabla_resumen.sort_values('total_deaths', ascending=False).head(20)
    
    # Añadir información de región
    regiones_por_pais = gdf_filtrado_muertes.groupby('country')['region'].first().to_dict()
    tabla_resumen['region'] = tabla_resumen['country'].map(regiones_por_pais)
    
    # Reordenar columnas
    tabla_resumen = tabla_resumen[['country', 'region', 'total_deaths', 'eventos', 'conflictos', 'promedio_muertes_evento']]
    tabla_resumen.columns = ['País', 'Región', 'Muertes Totales', 'Eventos', 'Conflictos', 'Muertes/Evento']
    
    # Formatear números
    tabla_resumen['Muertes Totales'] = tabla_resumen['Muertes Totales'].apply(lambda x: f"{int(x):,}")
    tabla_resumen['Eventos'] = tabla_resumen['Eventos'].apply(lambda x: f"{int(x):,}")
    tabla_resumen['Conflictos'] = tabla_resumen['Conflictos'].apply(lambda x: f"{int(x):,}")
    
    st.dataframe(tabla_resumen, use_container_width=True, hide_index=True)

elif page == "📊 Análisis Colombia":
    # Título principal
    st.markdown('<h1 class="main-title">Análisis Específico de Colombia</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Exploración detallada del conflicto armado colombiano: actores, territorios y evolución temporal</p>', unsafe_allow_html=True)
    
    # Filtrar datos solo de Colombia
    colombia_data = gdf[gdf['country'] == 'Colombia'].copy()
    
    if len(colombia_data) == 0:
        st.error("❌ No se encontraron datos para Colombia en el dataset.")
        st.info("💡 Verifica que el nombre del país en los datos sea exactamente 'Colombia'.")
        st.stop()
    
    # Filtros específicos para Colombia en sidebar
    st.sidebar.markdown("## 🇨🇴 Filtros Colombia")
    st.sidebar.markdown("---")
    
    # Filtro por período temporal
    años_colombia = sorted(colombia_data['year'].unique())
    año_min_col, año_max_col = st.sidebar.select_slider(
        "📅 Período temporal:",
        options=años_colombia,
        value=(años_colombia[0], años_colombia[-1]),
        format_func=lambda x: str(x),
        key="slider_colombia"
    )
    
    # Filtro por tipo de violencia
    tipos_seleccionados_col = st.sidebar.multiselect(
        "🔍 Tipo de violencia:",
        options=list(TIPOS_VIOLENCIA.keys()),
        default=list(TIPOS_VIOLENCIA.keys()),
        format_func=lambda x: TIPOS_VIOLENCIA[x],
        key="multi_tipos_colombia"
    )
    
    # Filtro por actores del conflicto (si existe información)
    actores_seleccionados = []
    if 'side_a' in colombia_data.columns:
        actores_disponibles = sorted(colombia_data['side_a'].dropna().unique())
        if len(actores_disponibles) > 0:
            actores_seleccionados = st.sidebar.multiselect(
                "👥 Actores del conflicto:",
                options=actores_disponibles,
                default=actores_disponibles[:10] if len(actores_disponibles) > 10 else actores_disponibles,
                key="multi_actores_colombia"
            )
    
    # Aplicar filtros
    colombia_filtrado = colombia_data[
        (colombia_data['year'] >= año_min_col) & 
        (colombia_data['year'] <= año_max_col) &
        (colombia_data['type_of_violence'].isin(tipos_seleccionados_col))
    ].copy()
    
    # Aplicar filtro de actores si existe
    if actores_seleccionados and 'side_a' in colombia_data.columns:
        colombia_filtrado = colombia_filtrado[colombia_filtrado['side_a'].isin(actores_seleccionados)]
    
    # Verificar si hay datos después del filtrado
    if len(colombia_filtrado) == 0:
        st.warning("⚠️ No hay datos disponibles para los filtros seleccionados en Colombia.")
        st.stop()
    
    # Métricas específicas de Colombia
    st.markdown('<h3 class="project-description">📊 Panorama General del Conflicto</h3>', unsafe_allow_html=True)
    
    años_conflicto = año_max_col - año_min_col + 1
    total_eventos_col = len(colombia_filtrado)
    promedio_anual = int(total_eventos_col / años_conflicto) if años_conflicto > 0 else 0
    
    metricas_colombia = [
        ("📊 Eventos en Colombia", f"{total_eventos_col:,}"),
        ("💀 Muertes Totales", f"{int(colombia_filtrado['total_deaths'].sum()):,}"),
        ("📅 Años Analizados", f"{años_conflicto}"),
        ("📈 Eventos/Año", f"{promedio_anual:,}")
    ]
    
    mostrar_metricas_en_columnas(metricas_colombia)
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Contexto histórico
    st.markdown('<h3 class="project-description">🏛️ Contexto Histórico</h3>', unsafe_allow_html=True)
    
    # Crear información contextual basada en el período seleccionado
    contexto_html = '''
    <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 5px solid #28a745;">
        <h4 style="color: #155724; margin-bottom: 15px;">📚 Períodos Presidenciales en el Rango Seleccionado:</h4>
        <ul style="color: #155724; font-size: 1.1rem; line-height: 1.8;">
    '''
    
    if año_min_col <= 1998 <= año_max_col:
        contexto_html += '<li><strong>Andrés Pastrana (1998-2002):</strong> Proceso de paz con las FARC, zona de distensión</li>'
    if año_min_col <= 2002 <= año_max_col:
        contexto_html += '<li><strong>Álvaro Uribe (2002-2010):</strong> Política de Seguridad Democrática, ofensiva militar</li>'
    if año_min_col <= 2010 <= año_max_col:
        contexto_html += '<li><strong>Juan Manuel Santos (2010-2018):</strong> Proceso de paz, Acuerdo Final (2016)</li>'
    if año_min_col <= 2018 <= año_max_col:
        contexto_html += '<li><strong>Iván Duque (2018-2022):</strong> Implementación del Acuerdo de Paz</li>'
    if año_min_col <= 2022 <= año_max_col:
        contexto_html += '<li><strong>Gustavo Petro (2022-presente):</strong> "Paz Total", diálogos múltiples</li>'
    
    contexto_html += '</ul></div>'
    
    st.markdown(contexto_html, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Análisis temporal con hitos históricos
    st.markdown('<h3 class="project-description">📈 Evolución Temporal del Conflicto</h3>', unsafe_allow_html=True)
    
    # Crear gráfica temporal con eventos por año
    eventos_por_año_col = colombia_filtrado.groupby('year').agg({
        'id': 'count',
        'total_deaths': 'sum'
    }).reset_index()
    eventos_por_año_col.columns = ['year', 'eventos', 'muertes']
    
    col_temp1, col_temp2 = st.columns(2, gap="large")
    
    with col_temp1:
        st.markdown("**Evolución Anual de Eventos**")
        fig_eventos = plt.figure(figsize=(10, 6))
        plt.plot(eventos_por_año_col['year'], eventos_por_año_col['eventos'], 
                color='#e74c3c', linewidth=2, marker='o', markersize=4)
        
        # Añadir líneas verticales para eventos importantes
        if 2016 in eventos_por_año_col['year'].values:
            plt.axvline(x=2016, color='green', linestyle='--', alpha=0.7, label='Acuerdo de Paz')
        if 2002 in eventos_por_año_col['year'].values:
            plt.axvline(x=2002, color='blue', linestyle='--', alpha=0.7, label='Gobierno Uribe')
        
        plt.title('Eventos de Conflicto por Año en Colombia')
        plt.xlabel('Año')
        plt.ylabel('Número de Eventos')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        st.pyplot(fig_eventos)
        plt.close()
    
    with col_temp2:
        st.markdown("**Evolución Anual de Muertes**")
        fig_muertes = plt.figure(figsize=(10, 6))
        plt.plot(eventos_por_año_col['year'], eventos_por_año_col['muertes'], 
                color='#c0392b', linewidth=2, marker='s', markersize=4)
        
        # Añadir líneas verticales para eventos importantes
        if 2016 in eventos_por_año_col['year'].values:
            plt.axvline(x=2016, color='green', linestyle='--', alpha=0.7, label='Acuerdo de Paz')
        if 2002 in eventos_por_año_col['year'].values:
            plt.axvline(x=2002, color='blue', linestyle='--', alpha=0.7, label='Gobierno Uribe')
        
        plt.title('Muertes por Año en Colombia')
        plt.xlabel('Año')
        plt.ylabel('Número de Muertes')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        st.pyplot(fig_muertes)
        plt.close()
    
    # Separador
    st.markdown('<div style="margin: 15px 0; border-bottom: 1px solid #e0e0e0; opacity: 0.5;"></div>', unsafe_allow_html=True)
    
    # Análisis por tipo de violencia y actores
    st.markdown('<h3 class="project-description">⚔️ Análisis por Actores y Tipos de Violencia</h3>', unsafe_allow_html=True)
    
    col_act1, col_act2, col_act3 = st.columns(3, gap="large")
    
    with col_act1:
        # Gráfica por tipo de violencia
        st.markdown("**Distribución por Tipo de Violencia**")
        eventos_por_tipo_col = colombia_filtrado['type_of_violence'].value_counts()
        eventos_por_tipo_col.index = eventos_por_tipo_col.index.map(lambda x: TIPOS_VIOLENCIA.get(x, f'Tipo {x}'))
        
        fig_pie_col = plt.figure(figsize=(8, 6))
        colores_col = ['#e74c3c', '#3498db', '#2ecc71']
        plt.pie(eventos_por_tipo_col.values, labels=eventos_por_tipo_col.index, autopct='%1.1f%%', 
                colors=colores_col, startangle=90)
        plt.title('Tipos de Violencia en Colombia')
        st.pyplot(fig_pie_col)
        plt.close()
    
    with col_act2:
        # Top actores si existe la información
        if 'side_a' in colombia_filtrado.columns and len(actores_seleccionados) > 0:
            st.markdown("**Top 10 Actores del Conflicto**")
            actores_eventos = colombia_filtrado['side_a'].value_counts().head(10)
            
            fig_actores = plt.figure(figsize=(8, 6))
            actores_eventos.plot(kind='barh', color='#f39c12')
            plt.title('Eventos por Actor')
            plt.xlabel('Número de Eventos')
            plt.tight_layout()
            st.pyplot(fig_actores)
            plt.close()
        else:
            st.markdown("**Intensidad por Año (Heatmap)**")
            intensidad_anual = eventos_por_año_col.set_index('year')['eventos']
            
            fig_heat = plt.figure(figsize=(8, 6))
            plt.bar(intensidad_anual.index, intensidad_anual.values, color='#e67e22', alpha=0.7)
            plt.title('Intensidad del Conflicto por Año')
            plt.xlabel('Año')
            plt.ylabel('Eventos')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig_heat)
            plt.close()
    
    with col_act3:
        # Análisis antes/después del Acuerdo de Paz
        st.markdown("**Análisis Proceso de Paz (2016)**")
        if 2016 in colombia_filtrado['year'].values:
            antes_2016 = colombia_filtrado[colombia_filtrado['year'] < 2016]
            despues_2016 = colombia_filtrado[colombia_filtrado['year'] >= 2016]
            
            años_antes = len(antes_2016['year'].unique())
            años_despues = len(despues_2016['year'].unique())
            
            promedio_antes = len(antes_2016) / años_antes if años_antes > 0 else 0
            promedio_despues = len(despues_2016) / años_despues if años_despues > 0 else 0
            
            fig_paz = plt.figure(figsize=(8, 6))
            categorias = ['Antes 2016', 'Después 2016']
            promedios = [promedio_antes, promedio_despues]
            colores_paz = ['#e74c3c', '#27ae60']
            
            plt.bar(categorias, promedios, color=colores_paz, alpha=0.8)
            plt.title('Eventos Promedio por Año\n(Antes vs Después del Acuerdo)')
            plt.ylabel('Eventos Promedio/Año')
            
            # Añadir valores en las barras
            for i, v in enumerate(promedios):
                plt.text(i, v + 0.1, f'{v:.1f}', ha='center', fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig_paz)
            plt.close()
        else:
            st.info("📅 El año 2016 no está en el rango seleccionado para mostrar el análisis del Acuerdo de Paz")
    
    # Separador
    st.markdown('<div style="margin: 15px 0; border-bottom: 1px solid #e0e0e0; opacity: 0.5;"></div>', unsafe_allow_html=True)
    
    # Análisis geográfico por regiones/departamentos
    st.markdown('<h3 class="project-description">🗺️ Análisis Geográfico</h3>', unsafe_allow_html=True)
    
    # Si existe información de ubicación más específica
    if 'adm_1' in colombia_filtrado.columns:
        # Análisis por departamento/región administrativa
        eventos_por_region = colombia_filtrado['adm_1'].value_counts().head(15)
        
        col_geo1, col_geo2 = st.columns(2, gap="large")
        
        with col_geo1:
            st.markdown("**Top 15 Departamentos/Regiones Afectadas**")
            fig_dept = plt.figure(figsize=(10, 8))
            eventos_por_region.plot(kind='barh', color='#9b59b6')
            plt.title('Eventos por Departamento/Región')
            plt.xlabel('Número de Eventos')
            plt.tight_layout()
            st.pyplot(fig_dept)
            plt.close()
        
        with col_geo2:
            # Crear tabla resumen por departamento
            st.markdown("**Resumen por Departamento (Top 10)**")
            resumen_dept = colombia_filtrado.groupby('adm_1').agg({
                'id': 'count',
                'total_deaths': 'sum'
            }).round(0).reset_index().head(10)
            resumen_dept.columns = ['Departamento', 'Eventos', 'Muertes']
            resumen_dept = resumen_dept.sort_values('Eventos', ascending=False)
            
            # Formatear números
            resumen_dept['Eventos'] = resumen_dept['Eventos'].apply(lambda x: f"{int(x):,}")
            resumen_dept['Muertes'] = resumen_dept['Muertes'].apply(lambda x: f"{int(x):,}")
            
            st.dataframe(resumen_dept, use_container_width=True, hide_index=True)
    else:
        # Análisis usando coordenadas para inferir regiones aproximadas
        st.info("📍 Análisis geográfico basado en coordenadas disponibles")
        
        col_coord1, col_coord2 = st.columns(2, gap="large")
        
        with col_coord1:
            # Distribución geográfica básica
            st.markdown("**Distribución de Eventos en el Territorio**")
            fig_geo = plt.figure(figsize=(10, 6))
            plt.scatter(colombia_filtrado['longitude'], colombia_filtrado['latitude'], 
                       c=colombia_filtrado['total_deaths'], cmap='Reds', alpha=0.6, s=30)
            plt.colorbar(label='Muertes')
            plt.title('Distribución Geográfica de Eventos')
            plt.xlabel('Longitud')
            plt.ylabel('Latitud')
            plt.tight_layout()
            st.pyplot(fig_geo)
            plt.close()
        
        with col_coord2:
            # Estadísticas geográficas básicas
            st.markdown("**Estadísticas Geográficas**")
            lat_centro = colombia_filtrado['latitude'].mean()
            lon_centro = colombia_filtrado['longitude'].mean()
            
            stats_geo = pd.DataFrame({
                'Métrica': ['Latitud Centro', 'Longitud Centro', 'Eventos Norte', 'Eventos Sur'],
                'Valor': [
                    f"{lat_centro:.3f}°",
                    f"{lon_centro:.3f}°",
                    f"{len(colombia_filtrado[colombia_filtrado['latitude'] > lat_centro]):,}",
                    f"{len(colombia_filtrado[colombia_filtrado['latitude'] <= lat_centro]):,}"
                ]
            })
            
            st.dataframe(stats_geo, use_container_width=True, hide_index=True)
    
    # Tabla resumen final
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<h3 class="project-description">📋 Resumen Ejecutivo del Período</h3>', unsafe_allow_html=True)
    
    # Crear resumen ejecutivo
    summary_data = {
        'Indicador': [
            'Total de Eventos',
            'Total de Muertes', 
            'Eventos por Año (Promedio)',
            'Muertes por Evento (Promedio)',
            'Año más Violento',
            'Tipo de Violencia Predominante'
        ],
        'Valor': [
            f"{len(colombia_filtrado):,}",
            f"{int(colombia_filtrado['total_deaths'].sum()):,}",
            f"{len(colombia_filtrado) / años_conflicto:.1f}" if años_conflicto > 0 else "N/A",
            f"{colombia_filtrado['total_deaths'].mean():.1f}",
            f"{eventos_por_año_col.loc[eventos_por_año_col['eventos'].idxmax(), 'year']}" if not eventos_por_año_col.empty else "N/A",
            f"{eventos_por_tipo_col.index[0]}" if not eventos_por_tipo_col.empty else "N/A"
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    # Conclusiones contextual
    st.markdown("""
    <div style="background-color: #fff3cd; padding: 20px; border-radius: 10px; border-left: 5px solid #ffc107; margin-top: 20px;">
        <h4 style="color: #856404; margin-bottom: 15px;">📝 Puntos Clave del Análisis:</h4>
        <ul style="color: #856404; font-size: 1.1rem; line-height: 1.8;">
            <li><strong>Intensidad del Conflicto:</strong> El período analizado muestra la complejidad del conflicto armado colombiano</li>
            <li><strong>Impacto Territorial:</strong> El conflicto ha afectado múltiples regiones del país de manera diferenciada</li>
            <li><strong>Evolución Temporal:</strong> Se observan variaciones significativas según los períodos presidenciales y políticas implementadas</li>
            <li><strong>Proceso de Paz:</strong> Los datos reflejan el impacto de las negociaciones y acuerdos de paz en la dinámica del conflicto</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)