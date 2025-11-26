import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime, date
import os
import re
import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.colors as mcolors
import io
import base64
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from datetime import datetime, timedelta
import numpy as np

host = "172.16.17.24"
database = "oncologicos"
user = "oncologicos_lectura_cti"
password = "CtI_F3r5_2025$."
port = "5432"

engine = create_engine(
    f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
)

mov_med = pd.read_sql("SELECT * FROM movimiento_medicamentos;", engine)
ajuste = pd.read_sql("SELECT * FROM ajuste_inventario_motivos;", engine)
id_movimientos = pd.read_sql("SELECT * FROM tipo_movimientos;", engine)
salida = pd.read_sql("SELECT * FROM tipo_salidas;", engine)
clues = pd.read_sql("SELECT * FROM public.clues;", engine)
medicamentos_df = pd.read_sql("SELECT * FROM public.medicamentos;", engine)
movimientos = pd.read_sql("SELECT * FROM public.movimientos;", engine)
existencia = pd.read_sql("SELECT * FROM public.existencia_medicamentos;", engine)

movimientos = movimientos.rename(columns={"identificador": "clues"})
movimientos["clues"] = (
    movimientos["clues"]
    .str.replace(r"^[A-Za-z]-", "", regex=True)
    .str.replace(r"-\d+$", "", regex=True)
)

clues_ren = clues[["id", "clues", "nombre"]].rename(
    columns={"clues": "clues_2", "nombre": "unidad"}
)

movimientos = movimientos.merge(clues_ren, left_on="clues_id", right_on="id", how="left")
movimientos = movimientos.drop(columns=["clues_id", "id"], errors="ignore")


diferencias = movimientos[movimientos["clues"] != movimientos["clues_2"]]

mov_med = mov_med.rename(columns={"created_at": "fecha_movimiento"})
tipomov_ren = id_movimientos[["id", "nombre"]].rename(columns={"nombre": "movimiento"})

movimientos = movimientos.merge(tipomov_ren, left_on="tipo_movimiento_id", right_on="id", how="left")
movimientos = movimientos.drop(columns=["tipo_movimiento_id", "id"], errors="ignore")
movimientos = movimientos.drop(columns=["clues_2", "tipo_salida_texto"], errors="ignore")

# Agregar tipo de salida
tiposal_ren = salida[["id", "nombre"]].rename(columns={"nombre": "salida"})
movimientos = movimientos.merge(tiposal_ren, left_on="tipo_salida_id", right_on="id", how="left")
movimientos = movimientos.drop(columns=["tipo_salida_id", "id"], errors="ignore")

# Agregar tipo inventario
tipoin_ren = ajuste[["id", "nombre"]].rename(columns={"nombre": "inventario"})
movimientos = movimientos.merge(tipoin_ren, left_on="ajuste_inventario_motivo_id", right_on="id", how="left")
movimientos = movimientos.drop(
    columns=[
        "ajuste_inventario_motivo_id", "id", "consecutivo", "created_at",
        "updated_at", "ajuste_inventario_motivo_otro", "created_user", "updated_user"
    ],
    errors="ignore"
)

movimientos = movimientos.rename(columns={"salida": "tipo_salida"})

# Flags Entrada / Salida / Ajuste
movimientos["Entrada"] = (movimientos["movimiento"] == "Entrada").astype(int)
movimientos["Salida"] = (movimientos["movimiento"] == "Salida").astype(int)
movimientos["Ajuste"] = (movimientos["movimiento"] == "Ajuste").astype(int)

# ============================
# Integración con movimiento_medicamentos
# ============================
mov_med = mov_med.drop(columns=["updated_at", "esencial"], errors="ignore")
mov_med = mov_med.rename(columns={"id": "id_mov"})

mov_med = mov_med.merge(medicamentos_df, left_on="medicamento_id", right_on="id", how="left")
mov_med = mov_med.drop(columns=["medicamento_id", "created_at", "id_y", "id"], errors="ignore")

mov_med = mov_med.merge(movimientos, left_on="movimiento_id", right_index=True, how="left")
mov_med = mov_med.drop(columns=["id_mov", "movimiento_id"], errors="ignore")

# ============================
# Reporte del día
# ============================
hoy = pd.Timestamp("today").normalize()
mov_med["fecha_movimiento"] = pd.to_datetime(mov_med["fecha_movimiento"]).dt.normalize()

mov_hoy = mov_med[mov_med["fecha_movimiento"] == hoy]

conteo_mov = (
    mov_hoy.groupby("clues", as_index=False)
    .agg(claves=("movimiento", "count"))
)

unidades_df = clues[["clues", "nombre"]].rename(columns={"nombre": "unidad"})
todas_clues = mov_med[["clues"]].drop_duplicates().merge(unidades_df, on="clues", how="left")

# ============================
# Reemplazos de nombres
# ============================
reemplazos = {
        "HOSPITAL GENERAL DE ENSENADA": "HGE_Ensenada",
        "HOSPITAL GENERAL TIJUANA": "HG_Tijuana",
        "UNEME DE ONCOLOGÍA": "UNEME_Oncologia",
        "HOSPITAL GENERAL DE MEXICALI": "HG_Mexicali",
        "B. HOSPITAL GENERAL CON ESPECIALIDADES JUAN MARÍA DE SALVATIERRA": "HG_Salvatierra",
        "CENTRO ESTATAL DE ONCOLOGÍA \"DR. RUBÉN CARDOZA MACÍAS\"": "CEO_Cardoza",
        "CEO: CENTRO ESTATAL DE ONCOLOGÍA DE CAMPECHE": "CEO_Campeche",
        "HOSPITAL PEDIÁTRICO MOCTEZUMA": "HP_Moctezuma",
        "INSTITUTO ESTATAL DE CANCEROLOGÍA LIC. CARLOS DE LA MADRID VIRGEN": "IEC_LaMadrid",
        "HOSPITAL DE ESPECIALIDADES PEDIÁTRICAS": "HEP",
        "HOSPITAL CHIAPAS NOS UNE DR. JESUS GILBERTO GOMEZ MAZA": "H_ChiapasUnidos",
        "HOSPITAL GENERAL TAPACHULA": "HG_Tapachula",
        "HOSPITAL REGIONAL DE ALTA ESPECIALIDAD CIUDAD SALUD": "HRAE_CiudadSalud",
        "HOSPITAL REGIONAL DE ALTA ESPECIALIDAD IXTAPALUCA": "HRAE_Ixtapaluca",
        "INSTITUTO ESTATAL DE CANCEROLOGÍA DR ARTURO BELTRÁN ORTEGA": "IEC_Beltran",
        "HOSPITAL REGIONAL DE ALTA ESPECIALIDAD DEL BAJÍO": "HRAE_Bajio",
        "HOSPITAL GENERAL PACHUCA": "HG_Pachuca",
        "HOSPITAL GENERAL TULA": "HG_Tula",
        "CENTRO ESTATAL DE ATENCION ONCOLOGICA": "CEAO",
        "HOSPITAL INFANTIL DE MORELIA \"EVA SAMANO DE LÓPEZ MATEOS\"": "HI_EvaSamano",
        "HG DE CUERNAVACA DR. JOSE G. PARRES": "HG_Parres",
        "HOSPITAL DEL NIÑO MORELENSE": "HN_Morelense",
        "CENTRO ESTATAL DE CANCEROLOGÍA": "CEC",
        "CENTRO DE ONCOLOGÍA Y RADIOTERAPIA DE OAXACA, S.S.O.": "COR_Oaxaca",
        "HE DE LA NIÑEZ OAXAQUEÑA": "HE_NinezOaxaquena",
        "HOSPITAL REGIONAL DE ALTA ESPECIALIDAD DE OAXACA": "HRAE_Oaxaca",
        "HOSPITAL DE LA NIÑEZ POBLANA": "HN_Poblana",
        "UNIDAD DE ONCOLOGÍA": "U_Oncologia",
        "HOSPITAL DE ONCOLOGÍA IMSS-BIENESTAR CHETUMAL": "HO_Chetumal",
        "HOSPITAL GENERAL DE CANCÚN DR. JESÚS KUMATE RODRÍGUEZ": "HG_Kumate",
        "HOSPITAL REGIONAL DE ALTA ESPECIALIDAD DR.IGNACIO MORONES PRIETO": "HRAE_Morones",
        "HOSPITAL GENERAL DE CULIACÁN": "HG_Culiacan",
        "INSTITUTO SINALOENSE DE CANCEROLOGÍA": "ISC",
        "HOSPITAL PEDIÁTRICO DE SINALOA": "HP_Sinaloa",
        "CENTRO ESTATAL DE ONCOLOGÍA DR. ERNESTO RIVERA CLAISSE": "CEO_Rivera",
        "HOSPITAL GENERAL DEL ESTADO DE SONORA": "HGE_Sonora",
        "HOSPITAL INFANTIL DEL ESTADO DE SONORA": "HI_Sonora",
        "HOSPITAL REGIONAL DE ALTA ESPECIALIDAD DR. JUAN GRAHAM CASASUS": "HRAE_Graham",
        "HOSPITAL REGIONAL DE ALTA ESPECIALIDAD DEL NIÑO \"DR. RODOLFO NIETO PADRÓN\"": "HRAE_Nieto",
        "CENTRO ONCOLÓGICO DE TAMAULIPAS": "COTamaulipas",
        "HOSPITAL GENERAL DR. NORBERTO TREVIÑO ZAPATA": "HG_Trevino",
        "HOSPITAL GENERAL DE MATAMOROS": "HG_Matamoros",
        "CENTRO ONCOLÓGICO NUEVO LAREDO DR. RODOLFO TORRE CANTÚ": "CONL_Torre",
        "HOSPITAL GENERAL DE TAMPICO DR. CARLOS CANSECO": "HG_Canseco",
        "HOSPITAL INFANTIL DE TAMAULIPAS": "HI_Tamaulipas",
        "HOSPITAL REGIONAL DE ALTA ESPECIALIDAD EN CD VICTORIA BICENTENARIO 2010": "HRAE_Bicentenario",
        "HOSPITAL GENERAL MATERNO INFANTIL DE REYNOSA": "HGMIR_Reynosa",
        "HOSPITAL INFANTIL DE TLAXCALA": "HI_Tlaxcala",
        "HOSPITAL REGIONAL POZA RICA DE HIDALGO": "HR_PozaRica",
        "HOSPITAL REGIONAL RÍO BLANCO": "HR_RioBlano",
        "HOSPITAL REGIONAL DE COATZACOALCOS DR.VALENTIN GÓMEZ FARIAS": "HR_GomezFarias",
        "HOSPITAL DE ALTA ESPECIALIDAD DE VERACRUZ": "HAE_Veracruz",
        "CENTRO ESTATAL DE CANCEROLOGÍA DR. MIGUEL DORANTES MESA": "CEC_Dorantes",
        "HOSPITAL REGIONAL DE ALTA ESPECIALIDAD DE LA PENÍNSULA DE YUCATÁN": "HRAE_Peninsula",
        "HOSPITAL GENERAL ZACATECAS LUZ GONZÁLEZ COSIO": "HG_Zacatecas",
        "HOSPITAL DE LA NIÑO Y LA MUJER DR. ALBERTO LOPEZ HERMOSA": "HNM_Alberto_Lopez"
    }

todas_clues['unidad'] = todas_clues['unidad'].replace(reemplazos)

tabla_hoy = todas_clues.merge(conteo_mov, on="clues", how="left").fillna({"claves": 0})
tabla_hoy["claves"] = tabla_hoy["claves"].astype(int)
tabla_hoy = tabla_hoy[["clues", "unidad", "claves"]].sort_values(by="claves")

reportaron = (tabla_hoy["claves"] > 0).sum()
no_reportaron = (tabla_hoy["claves"] == 0).sum()
print(f"Reportaron: {reportaron}, No reportaron: {no_reportaron}")

# ============================
# Histórico
# ============================
mov_med["fecha_movimiento"] = pd.to_datetime(mov_med["fecha_movimiento"], errors="coerce").dt.normalize()

df_agg = (
    mov_med.groupby(["clues", "fecha_movimiento"], as_index=False)
    .agg(claves=("movimiento", "count"))
)

df_agg = df_agg.merge(unidades_df, on="clues", how="left")
df_agg["unidad"] = df_agg["unidad"].replace(reemplazos)
df_agg["fecha"] = df_agg["fecha_movimiento"].dt.strftime("%d/%m/%Y")

historico_df = df_agg.pivot_table(
    index=["clues", "unidad"],
    columns="fecha",
    values="claves",
    fill_value=0
).reset_index()

columnas_fechas = sorted(
    [c for c in historico_df.columns if c not in ["clues", "unidad"]],
    key=lambda x: pd.to_datetime(x, format="%d/%m/%Y")
)

historico_df = historico_df[["clues", "unidad"] + columnas_fechas]

# ============================
# FILTRAR SOLO DÍAS HÁBILES EN EL HISTÓRICO
# ============================
print("Filtrando días hábiles (excluyendo fines de semana)...")

# Función para identificar si una fecha es día hábil
def es_dia_habil(fecha_str):
    """
    Retorna True si la fecha es día hábil (lunes a viernes)
    """
    try:
        fecha_dt = pd.to_datetime(fecha_str, format='%d/%m/%Y')
        return fecha_dt.weekday() < 5  # 0-4 = lunes a viernes
    except:
        return False

# Identificar columnas que son días hábiles
columnas_habiles = [col for col in columnas_fechas if es_dia_habil(col)]
columnas_fin_semana = [col for col in columnas_fechas if not es_dia_habil(col)]

print(f"Días hábiles encontrados: {len(columnas_habiles)}")
print(f"Días fin de semana excluidos: {len(columnas_fin_semana)}")

# ACTUALIZAR TODOS LOS DATAFRAMES PARA USAR SOLO DÍAS HÁBILES
columnas_fechas = columnas_habiles  # Reemplazar con solo días hábiles

# Actualizar historico_df para usar solo días hábiles
historico_df = historico_df[['clues', 'unidad'] + columnas_fechas]

# Actualizar df_plot para el heatmap (solo días hábiles)
df_plot = historico_df[columnas_fechas].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
max_val = int(df_plot.values.max())

# Actualizar y_labels para el heatmap
y_labels = historico_df['unidad'].astype(str).tolist()
x_labels = columnas_fechas

# ACTUALIZAR DATOS PARA GRÁFICAS DE MOVIMIENTOS
# 1. Gráfica de CLUES con movimientos (solo días hábiles)
historico_bin = historico_df.copy()
historico_bin[columnas_fechas] = (historico_bin[columnas_fechas] > 0).astype(int)
conteo_diario = historico_bin[columnas_fechas].sum().reset_index()
conteo_diario.columns = ["Fecha", "CLUES con movimientos"]
conteo_diario["Fecha_dt"] = pd.to_datetime(conteo_diario["Fecha"], format="%d/%m/%Y")

# 2. Gráfica de total de movimientos (solo días hábiles)
conteo_total = historico_df[columnas_fechas].sum().reset_index()
conteo_total.columns = ["Fecha", "Total de movimientos"]
conteo_total["Total de movimientos"] = conteo_total["Total de movimientos"].astype(int)
conteo_total["Fecha_dt"] = pd.to_datetime(conteo_total["Fecha"], format="%d/%m/%Y")

# Obtener la fecha de hoy en formato dd/mm/yyyy
fecha_hoy = datetime.now().strftime('%d/%m/%Y')

# Verificar si la fecha de hoy ya está en las columnas del DataFrame histórico
if fecha_hoy not in historico_df.columns:
    # Agregar la columna de hoy con valores 0 al histórico
    historico_df[fecha_hoy] = 0
    # Actualizar la lista de columnas de fechas para incluir hoy
    if fecha_hoy not in columnas_fechas:
        columnas_fechas.append(fecha_hoy)

# SOLUCIÓN: Filtrar solo las columnas que existen en el DataFrame histórico
columnas_existentes = [col for col in columnas_fechas if col in historico_df.columns]
df_numerico = historico_df[columnas_existentes].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
max_val = int(df_numerico.values.max())

df_plot = df_numerico.copy()
rows = df_plot.shape[0]
cols = df_plot.shape[1]

# PREPARAR DATOS PARA EXCEL (AHORA SOLO DÍAS HÁBILES)
print("Preparando datos para exportación Excel...")

# Crear DataFrames para exportación
historico_resumen = historico_df.copy()
historico_resumen['Total_Movimientos'] = historico_resumen[columnas_fechas].sum(axis=1)
historico_resumen['Dias_Con_Movimiento'] = (historico_resumen[columnas_fechas] > 0).sum(axis=1)
historico_resumen['Dias_Sin_Movimiento'] = (historico_resumen[columnas_fechas] == 0).sum(axis=1)

# Calcular porcentaje basado en días hábiles (se mantiene para referencia)
historico_resumen['Porcentaje_Dias_Activos'] = (
    historico_resumen['Dias_Con_Movimiento'] / len(columnas_fechas) * 100
).round(2)

historico_resumen = historico_resumen.sort_values('Dias_Con_Movimiento', ascending=False)

# DataFrame para tabla del día (sin cambios)
tabla_hoy_excel = tabla_hoy.copy()

# DataFrame para tabla histórica - BASADO EN DÍAS ABSOLUTOS CON NUEVOS RANGOS
tabla_historica_excel = historico_resumen[['clues', 'unidad', 'Total_Movimientos', 
                                          'Dias_Con_Movimiento', 'Dias_Sin_Movimiento', 
                                          'Porcentaje_Dias_Activos']].copy()

# NUEVA LÓGICA: Clasificación basada en días absolutos con rangos 0-20, 21-40, 41-61
tabla_historica_excel['Nivel_Actividad'] = tabla_historica_excel['Dias_Con_Movimiento'].apply(
    lambda x: 'Alto' if x > 40 else 'Medio' if x > 20 else 'Bajo'
)

# DataFrame para datos del heatmap (movimientos diarios por unidad - solo días hábiles)
heatmap_data_excel = historico_df[['clues', 'unidad'] + columnas_fechas].copy()

# GENERAR ARCHIVOS EXCEL
import io
import base64

def dataframe_to_excel_bytes(df, sheet_name='Datos'):
    """Convierte un DataFrame a bytes de archivo Excel en base64"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        # Ajustar automáticamente el ancho de las columnas
        worksheet = writer.sheets[sheet_name]
        for idx, col in enumerate(df.columns):
            max_len = max(df[col].astype(str).str.len().max(), len(col)) + 2
            worksheet.set_column(idx, idx, min(max_len, 50))
            
    excel_bytes = output.getvalue()
    return base64.b64encode(excel_bytes).decode()

# Generar archivos Excel en base64
excel_tabla_hoy_b64 = dataframe_to_excel_bytes(tabla_hoy_excel, 'Movimientos_Dia')
excel_tabla_historica_b64 = dataframe_to_excel_bytes(tabla_historica_excel, 'Resumen_Historico')
excel_heatmap_b64 = dataframe_to_excel_bytes(heatmap_data_excel, 'Movimientos_Detallados')

# ============================
# CONFIGURACIÓN PLOTLY
# ============================
pio.templates.default = "plotly_white"

COLOR_PRIMARIO = '#611232'  
COLOR_SECUNDARIO = '#AE8640'  
COLOR_FONDO = '#FFF8E7'  
COLOR_BORDE = '#7A1737'  
COLOR_TEXTO = '#000000'  
COLOR_TITULO = '#AE8640'

TAMANO_GRAFICA = (20, 18)

def plotly_to_html(fig, width='100%', height='600px'):
    return pio.to_html(fig, include_plotlyjs='cdn', div_id=None, 
                      auto_play=False, config={'responsive': True})

print("Generando gráficas para HTML...")

print("Generando gráfica de dona interactiva...")
fig_dona = go.Figure()
fig_dona.add_trace(go.Pie(
    values=[reportaron, no_reportaron],
    labels=[f'Reportaron ({reportaron})', f'No reportaron ({no_reportaron})'],
    hole=0.6,
    marker_colors=["#10312B", "#9D2449"],
    textinfo='percent',
    textposition='inside',
    textfont_size=16
))
fig_dona.update_layout(
    title_text="",
    title_x=0.5,
    showlegend=True,
    height=700
)
dona_html = plotly_to_html(fig_dona)

print("Generando gráfica de movimientos interactiva...")

fig_mov = go.Figure()
fig_mov.add_trace(go.Bar(
    x=conteo_diario['Fecha'],
    y=conteo_diario['CLUES con movimientos'],
    marker_color="#10312B",
    opacity=0.7,
    text=conteo_diario['CLUES con movimientos'],
    textposition='auto',
    textfont=dict(size=12)
))

layout_config = {
    'title_text': "",
    'title_x': 0.5,
    'xaxis_title': "",
    'yaxis_title': "",
    'xaxis_tickangle': -45,
    'height': 700,
    'xaxis': dict(
        rangeslider=dict(visible=True, thickness=0.05),
        type="category"
    )
}

fig_mov.update_layout(**layout_config)
movimientos_html = plotly_to_html(fig_mov)

print("Generando gráfica de total movimientos interactiva...")

fig_total = go.Figure()
fig_total.add_trace(go.Bar(
    x=conteo_total['Fecha'],
    y=conteo_total['Total de movimientos'],
    marker_color="#10312B",
    opacity=0.7,
    text=conteo_total['Total de movimientos'],
    textposition='auto',
    textfont=dict(size=12)
))

layout_config_total = {
    'title_text': "",
    'title_x': 0.5,
    'xaxis_title': "",
    'yaxis_title': "",
    'xaxis_tickangle': -45,
    'height': 700,
    'xaxis': dict(
        rangeslider=dict(visible=True, thickness=0.05),
        type="category"
    )
}

fig_total.update_layout(**layout_config_total)
total_movimientos_html = plotly_to_html(fig_total)

print("Generando heatmap interactivo...")

z_data = df_plot.values
x_labels = df_plot.columns.tolist()

max_val = z_data.max()

colorscale = [
    [0, "#9D2449"],
    [0.5/max_val, "#9D2449"],
    [0.5/max_val, "#DDC9A3"],
    [10/max_val, "#DDC9A3"],
    [10/max_val, "#235B4A"],
    [50/max_val, "#235B4A"],
    [50/max_val, "#10312B"],
    [1, "#10312B"]
]

hover_text = []
for i, unidad in enumerate(y_labels):
    hover_row = []
    for j, fecha in enumerate(x_labels):
        valor = z_data[i, j]
        if fecha == fecha_hoy:
            hover_row.append(f"<b>UNIDAD: {unidad}<br>FECHA: {fecha} (HOY)<br>MOVIMIENTOS: {valor}</b>")
        else:
            hover_row.append(f"UNIDAD: {unidad}<br>FECHA: {fecha}<br>MOVIMIENTOS: {valor}")
    hover_text.append(hover_row)

fig_heatmap = go.Figure(data=go.Heatmap(
    z=z_data,
    x=x_labels,
    y=y_labels,
    colorscale=colorscale,
    hoverinfo='text',
    text=hover_text,
    showscale=True,
    colorbar=dict(
        title="Movimientos",
        tickvals=[0, 0.5, 10, 50, max_val],
        ticktext=["0", "0.5", "10", "50", f"{max_val}+"]
    )
))

hoy_index = x_labels.index(fecha_hoy) if fecha_hoy in x_labels else -1

# CONFIGURACIÓN DEL HEATMAP SIN ZOOM
fig_heatmap.update_layout(
    title=f"",
    title_x=0.5,
    xaxis_title="",
    yaxis_title="",
    xaxis=dict(
        tickangle=-45,
        tickfont=dict(size=15),
        type="category",
        # DESACTIVAR ZOOM Y DESPLAZAMIENTO
        rangeslider=dict(visible=False),
        fixedrange=True  # Evitar zoom
    ),
    yaxis=dict(
        tickfont=dict(size=12),
        autorange="reversed",
        fixedrange=True  # Evitar zoom
    ),
    height=max(800, len(y_labels) * 25),
    width=max(1200, len(x_labels) * 30),
    margin=dict(l=200, r=50, t=80, b=150),
    # DESACTIVAR INTERACTIVIDAD DE ZOOM
    dragmode=False
)

if hoy_index >= 0:
    fig_heatmap.add_vline(
        x=hoy_index,
        line_width=3,
        line_dash="dash",
        line_color="blue",
        opacity=0.7
    )

    fig_heatmap.update_xaxes(
        ticktext=[f"<b>{label}</b>" if label == fecha_hoy else label for label in x_labels],
        tickvals=list(range(len(x_labels)))
    )

heatmap_html = plotly_to_html(fig_heatmap)

# ============================
# GENERAR HTML
# ============================
html_content = f"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Reporte de Movimientos de Medicamentos - IMSS Bienestar</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;600;700&display=swap');
        
        body {{
            font-family: 'Montserrat', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background-color: {COLOR_FONDO};
            color: {COLOR_TEXTO};
        }}
        .header {{
            background-color: {COLOR_PRIMARIO};
            padding: 15px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        }}
        .header-logos {{
            display: flex;
            align-items: center;
            gap: 20px;
        }}
        .logo {{
            height: 50px;
        }}
        .logo-imss {{
            height: 70px;
        }}
        .container {{
            max-width: 1800px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 15px rgba(0,0,0,0.1);
            margin-top: 20px;
            margin-bottom: 20px;
            border: 1px solid {COLOR_BORDE};
        }}
        h1, h2, h3 {{
            color: {COLOR_TITULO};
            text-align: center;
            font-weight: 700;
        }}
        h1 {{
            border-bottom: 2px solid {COLOR_SECUNDARIO};
            padding-bottom: 10px;
            margin-bottom: 25px;
        }}
        .dashboard {{
            display: flex;
            flex-direction: column;
            gap: 30px;
            margin-bottom: 30px;
        }}
        .chart-row {{
            display: flex;
            justify-content: center;
            width: 100%;
        }}
        .chart-container {{
            background-color: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 1px 8px rgba(0,0,0,0.1);
            text-align: center;
            border: 1px solid {COLOR_BORDE};
            margin: 10px;
            width: 100%;
        }}
        .chart-single {{
            width: 98%;
            max-width: none;
        }}
        .plotly-chart {{
            width: 100%;
            border: 1px solid #ddd;
            border-radius: 4px;
        }}
        .heatmap-container {{
            width: 100%;
            overflow-x: auto;
        }}
        .stats-container {{
            display: flex;
            justify-content: space-around;
            margin: 20px 0;
        }}
        .stat-card {{
            background: linear-gradient(135deg, {COLOR_PRIMARIO}, {COLOR_BORDE});
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            width: 45%;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            transition: transform 0.3s ease;
        }}
        .stat-card:hover {{
            transform: translateY(-5px);
        }}
        .stat-card h3 {{
            margin-top: 0;
            font-weight: 600;
        }}
        .stat-value {{
            font-size: 2.5em;
            font-weight: bold;
        }}
        .table-container {{
            margin: 30px 0;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 1px 8px rgba(0,0,0,0.1);
            border: 1px solid {COLOR_BORDE};
            position: relative;
        }}
        .download-btn {{
            position: absolute;
            top: 20px;
            right: 20px;
            background-color: {COLOR_SECUNDARIO};
            color: white;
            border: none;
            padding: 10px 15px;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
            transition: background-color 0.3s ease;
            display: flex;
            align-items: center;
            gap: 5px;
        }}
        .download-btn:hover {{
            background-color: {COLOR_PRIMARIO};
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 14px;
        }}
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: {COLOR_PRIMARIO};
            color: white;
            font-weight: bold;
            position: sticky;
            top: 0;
        }}
        tr:hover {{
            background-color: #f8f9fa;
        }}
        .reportaron {{
            background-color: #e8f5e8;
        }}
        .no-reportaron {{
            background-color: #ffeaea;
        }}
        .alta-actividad {{
            background-color: #e8f5e8;
        }}
        .media-actividad {{
            background-color: #fff8e1;
        }}
        .baja-actividad {{
            background-color: #ffeaea;
        }}
        .scrollable-table {{
            max-height: 600px;
            overflow-y: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        .update-info {{
            text-align: center;
            margin-top: 20px;
            font-style: italic;
            color: #666;
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid {COLOR_SECUNDARIO};
        }}
        .resumen-tabla {{
            background: linear-gradient(135deg, {COLOR_FONDO}, #fff);
            padding: 15px;
            border-radius: 5px;
            margin: 15px 0;
            text-align: center;
            border: 1px solid {COLOR_BORDE};
            font-weight: 600;
        }}
        .tabla-section {{
            margin-bottom: 40px;
        }}
        .badge {{
            display: inline-block;
            padding: 5px 12px;
            border-radius: 15px;
            font-size: 12px;
            font-weight: bold;
            color: white;
        }}
        .badge-alta {{
            background-color: #28a745;
        }}
        .badge-media {{
            background-color: #ffc107;
            color: black;
        }}
        .badge-baja {{
            background-color: #dc3545;
        }}
        .footer {{
            background-color: {COLOR_PRIMARIO};
            color: white;
            text-align: center;
            padding: 20px;
            margin-top: 30px;
            border-radius: 8px;
        }}
        .zoom-info {{
            background-color: #e7f3ff;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
            font-size: 12px;
            text-align: center;
            border-left: 4px solid #007bff;
        }}
        .heatmap-instructions {{
            background-color: #fff3cd;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
            font-size: 12px;
            text-align: center;
            border-left: 4px solid #ffc107;
        }}
        @media (max-width: 768px) {{
            .container {{
                max-width: 95%;
                padding: 10px;
            }}
            .chart-row {{
                flex-direction: column;
                align-items: center;
            }}
            .chart-single {{
                width: 98%;
                margin: 10px 0;
            }}
            .stats-container {{
                flex-direction: column;
                align-items: center;
            }}
            .stat-card {{
                width: 90%;
                margin: 10px 0;
            }}
            .download-btn {{
                position: relative;
                top: auto;
                right: auto;
                margin-bottom: 15px;
                width: 100%;
            }}
        }}
    </style>
</head>
<body>
    <!-- ENCABEZADO CON LOGOS -->
    <div class="header">
        <div class="header-logos">
            <img src="https://framework-gb.cdn.gob.mx/landing/img/logoheader.svg" 
                 alt="Gobierno de México" class="logo">
            <img src="https://imssbienestar.gob.mx/assets/img/imb_b.svg" 
                 alt="IMSS Bienestar" class="logo-imss">
        </div>
    </div>

    <div class="container">
        <h1>Reporte de Movimientos de Medicamentos</h1>
        <div class="update-info">
            Última actualización: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}
        </div>
        
        <div class="zoom-info">
            <strong> Controles interactivos:</strong> Las gráficas de series de tiempo muestran todos los datos disponibles y tienen barras de desplazamiento para navegar por el tiempo.
        </div>
        
        <div class="stats-container">
            <div class="stat-card">
                <h3> Reportados al día </h3>
                <div class="stat-value">{reportaron}</div>
            </div>
            <div class="stat-card">
                <h3>No Reportados al día </h3>
                <div class="stat-value">{no_reportaron}</div>
            </div>
        </div>

        <!-- 1. GRÁFICA DE DONA - PRIMERA VISUALIZACIÓN -->
        <div class="dashboard">
            <div class="chart-row">
                <div class="chart-container chart-single">
                    <h2>Reporte del Día</h2>
                    {dona_html}
                </div>
            </div>
        </div>

        <!-- 2. TABLA HISTÓRICA - SEGUNDA VISUALIZACIÓN -->
        <div class="tabla-section">
            <div class="table-container">
                <h2>Reporte Histórico - Resumen por Unidad</h2>
                
                <!-- Botón de descarga para tabla histórica -->
                <button class="download-btn" onclick="downloadExcel('tabla_historica')">
                    Descargar Excel
                </button>
                
                <div class="resumen-tabla">
                    <strong>Período analizado:</strong> {len(columnas_fechas)} días hábiles (excluyendo fines de semana) | 
                    <strong>Desde:</strong> {columnas_fechas[0] if columnas_fechas else 'N/A'} | 
                    <strong>Hasta:</strong> {columnas_fechas[-1] if columnas_fechas else 'N/A'}<br>
                    <strong>Criterio basado en días absolutos de actividad</strong>
                </div>
                
                <div class="scrollable-table">
                    <table>
                        <thead>
                            <tr>
                                <th>CLUES</th>
                                <th>Unidad</th>
                                <th>Total Movimientos</th>
                                <th>Días con Movimiento</th>
                                <th>Días sin Movimiento</th>
                                <th>% Días Activos</th>
                                <th>Nivel de Actividad</th>
                            </tr>
                        </thead>
                        <tbody>
"""

# NUEVA LÓGICA BASADA EN DÍAS ABSOLUTOS CON RANGOS 0-20, 21-40, 41-61
for _, fila in historico_resumen.iterrows():
    dias_con_movimiento = fila['Dias_Con_Movimiento']
    
    # Clasificación basada en días absolutos con nuevos rangos
    if dias_con_movimiento > 40:  # 41+ días -> Alto
        nivel_actividad = "Alto"
        clase_fila = "alta-actividad"
        badge_class = "badge badge-alta"
    elif dias_con_movimiento > 20:  # 21-40 días -> Medio
        nivel_actividad = "Medio"
        clase_fila = "media-actividad"
        badge_class = "badge badge-media"
    else:  # 0-20 días -> Bajo
        nivel_actividad = "Bajo"
        clase_fila = "baja-actividad"
        badge_class = "badge badge-baja"
    
    html_content += f"""
                            <tr class="{clase_fila}">
                                <td><strong>{fila['clues']}</strong></td>
                                <td>{fila['unidad']}</td>
                                <td><strong>{int(fila['Total_Movimientos'])}</strong></td>
                                <td>{int(fila['Dias_Con_Movimiento'])}</td>
                                <td>{int(fila['Dias_Sin_Movimiento'])}</td>
                                <td><strong>{fila['Porcentaje_Dias_Activos']}%</strong></td>
                                <td><span class="{badge_class}">{nivel_actividad}</span></td>
                            </tr>jupyter nbconvert --to script "codigo para html estatico.ipynb"

"""

html_content += f"""
                        </tbody>
                    </table>
                </div>
                
                <div style="margin-top: 15px; font-size: 12px; color: #666;">
                    <strong>Resumen de actividad (solo días hábiles):</strong><br>
                    <span class="badge badge-alta">Alto</span> 41-61 días activos | 
                    <span class="badge badge-media">Medio</span> 21-40 días activos | 
                    <span class="badge badge-baja">Bajo</span> 0-20 días activos<br>
                    <strong>Total de unidades monitoreadas:</strong> {len(historico_resumen)}<br>
                    <strong>Promedio de días activos:</strong> {historico_resumen['Dias_Con_Movimiento'].mean():.1f} días<br>
                    <strong>Nota:</strong> Todos los cálculos y gráficas excluyen fines de semana (sábados y domingos)
                </div>
            </div>
        </div>
        
        <!-- 3. GRÁFICA DE CLUES CON MOVIMIENTOS - TERCERA VISUALIZACIÓN -->
        <div class="dashboard">
            <div class="chart-row">
                <div class="chart-container chart-single">
                    <h2>CLUES con Movimientos por Fecha</h2>
                    {movimientos_html}
                </div>
            </div>
        </div>

        <!-- 4. GRÁFICA DE TOTAL DE MOVIMIENTOS - CUARTA VISUALIZACIÓN -->
        <div class="dashboard">
            <div class="chart-row">
                <div class="chart-container chart-single">
                    <h2>Total de Movimientos por Fecha</h2>
                    {total_movimientos_html}
                </div>
            </div>
        </div>

        <!-- 5. TABLA DE MOVIMIENTOS DEL DÍA - QUINTA VISUALIZACIÓN -->
        <div class="tabla-section">
            <div class="table-container">
                <h2>Tabla de Movimientos del Día - {hoy.strftime('%d/%m/%Y')}</h2>
                
                <!-- Botón de descarga para tabla del día -->
                <button class="download-btn" onclick="downloadExcel('tabla_hoy')">
                    Descargar Excel
                </button>
                
                <div class="resumen-tabla">
                    <strong>Resumen:</strong> {reportaron} unidades reportaron movimientos | {no_reportaron} unidades no reportaron
                </div>
                
                <div class="scrollable-table">
                    <table>
                        <thead>
                            <tr>
                                <th>CLUES</th>
                                <th>Unidad</th>
                                <th>Movimientos</th>
                                <th>Estado</th>
                            </tr>
                        </thead>
                        <tbody>
"""

for _, fila in tabla_hoy.iterrows():
    estado = "Reportó" if fila['claves'] > 0 else "No reportó"
    clase_fila = "reportaron" if fila['claves'] > 0 else "no-reportaron"
    
    html_content += f"""
                            <tr class="{clase_fila}">
                                <td><strong>{fila['clues']}</strong></td>
                                <td>{fila['unidad']}</td>
                                <td><strong>{fila['claves']}</strong></td>
                                <td><strong>{estado}</strong></td>
                            </tr>
"""

html_content += f"""
                        </tbody>
                    </table>
                </div>
                
                <div style="margin-top: 15px; font-size: 12px; color: #666;">
                    <strong>Total de unidades:</strong> {len(tabla_hoy)}<br>
                    <strong>Unidades con movimientos:</strong> {reportaron}<br>
                    <strong>Unidades sin movimientos:</strong> {no_reportaron}
                </div>
            </div>
        </div>

        <!-- 6. HEATMAP INTERACTIVO - ÚLTIMA VISUALIZACIÓN -->
        <div class="dashboard">
            <div class="chart-row">
                <div class="chart-container chart-single">
                    <h2>Heatmap Interactivo - Movimientos por Unidad y Día</h2>
                    <div class="heatmap-instructions">
                        <strong>Instrucciones:</strong> Pasa el cursor sobre las celdas para ver los detalles de movimientos por unidad y fecha. <strong>El heatmap no tiene zoom para mejor visualización.</strong>
                    </div>
                    <div class="heatmap-container">
                        {heatmap_html}
                    </div>
                </div>
            </div>
        </div>

        <!-- BOTÓN PARA DESCARGAR DATOS COMPLETOS DEL HEATMAP -->
        <div class="tabla-section">
            <div class="table-container">
                <h2>Datos Completos - Movimientos Detallados</h2>
                
                <!-- Botón de descarga para datos del heatmap -->
                <button class="download-btn" onclick="downloadExcel('heatmap_data')">
                    Descargar Excel Completo
                </button>
                
                <div class="resumen-tabla">
                    <strong>Contiene:</strong> Todos los movimientos diarios por unidad (CLUES y nombre) para el período completo analizado
                </div>
                
                <div style="margin-top: 15px; font-size: 12px; color: #666;">
                    <strong>Archivo incluye:</strong> Movimientos detallados por CLUES, unidad y fecha<br>
                    <strong>Total de unidades:</strong> {len(historico_df)}<br>
                    <strong>Período cubierto:</strong> {len(columnas_fechas)} días hábiles<br>
                    <strong>Columnas:</strong> CLUES, Unidad y fechas con movimientos
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>Este reporte se actualiza automáticamente cada vez que se ejecuta el script.</p>
            <p> IMSS Bienestar</p>
        </div>
    </div>

    <script>
        // Datos Excel en base64
        const excelData = {{
            'tabla_hoy': '{excel_tabla_hoy_b64}',
            'tabla_historica': '{excel_tabla_historica_b64}',
            'heatmap_data': '{excel_heatmap_b64}'
        }};

        // Nombres de archivos
        const fileNames = {{
            'tabla_hoy': 'movimientos_dia_{hoy.strftime('%Y%m%d')}.xlsx',
            'tabla_historica': 'resumen_historico_{datetime.now().strftime('%Y%m%d')}.xlsx',
            'heatmap_data': 'movimientos_detallados_{datetime.now().strftime('%Y%m%d')}.xlsx'
        }};

        function downloadExcel(dataType) {{
            const base64Data = excelData[dataType];
            const fileName = fileNames[dataType];
            
            // Convertir base64 a blob
            const byteCharacters = atob(base64Data);
            const byteNumbers = new Array(byteCharacters.length);
            for (let i = 0; i < byteCharacters.length; i++) {{
                byteNumbers[i] = byteCharacters.charCodeAt(i);
            }}
            const byteArray = new Uint8Array(byteNumbers);
            const blob = new Blob([byteArray], {{type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'}});
            
            // Crear enlace de descarga
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = fileName;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
        }}
    </script>
</body>
</html>
"""

with open("C:\\Users\\jose.valdez\\Downloads\\index.html", "w", encoding="utf-8") as f:
    f.write(html_content)



import webbrowser
webbrowser.open("C:\\Users\\jose.valdez\\Downloads\\index.html")