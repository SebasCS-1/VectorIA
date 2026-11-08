import streamlit as st
import requests
import pandas as pd
import plotly.express as px

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="VectorIA", page_icon="🦟", layout="wide")

st.markdown("""
<div style='background:#2E7D32;padding:25px;border-radius:15px;color:white;text-align:center;'>
<h1>🦟 VectorIA</h1>
<h3>Sistema Inteligente para la Vigilancia del Dengue</h3>
</div>
""", unsafe_allow_html=True)

st.info("Herramienta académica de apoyo a la vigilancia epidemiológica. No diagnostica dengue en personas.")

try:
    health=requests.get(f"{API_URL}/health",timeout=30).json()
    st.success("API conectada correctamente")
except Exception as e:
    st.error(f"No fue posible conectar con la API: {e}")
    st.stop()

with st.expander("Estado interno del sistema"):
    st.json(health)

st.header("1️⃣ Clasificación de mosquito")
imagen=st.file_uploader("Suba una fotografía de mosquito",type=['jpg','jpeg','png','bmp','webp'])
clase_detectada=None
if imagen is not None:
    c1,c2=st.columns([3,2])
    with c1:
        st.image(imagen,use_container_width=True)
    with c2:
        files={"archivo":(imagen.name,imagen.getvalue(),imagen.type)}
        r=requests.post(f"{API_URL}/mosquito/classify",files=files,timeout=60).json()
        clase_detectada=r.get('clase')
        st.metric('Especie detectada',clase_detectada)
        st.metric('Confianza',f"{float(r.get('confianza',0)):.2%}")
        if 'probabilidades' in r:
            probs=pd.DataFrame({'Clase':list(r['probabilidades'].keys()),'Probabilidad':list(r['probabilidades'].values())})
            fig=px.bar(probs,x='Clase',y='Probabilidad',color='Clase',title='Probabilidades CNN')
            st.plotly_chart(fig,use_container_width=True)

st.divider()
st.header('2️⃣ Pronóstico de dengue')
try:
    fechas=requests.get(f"{API_URL}/dengue/dates",timeout=30).json()['fechas']
    fecha=st.selectbox('Seleccione una fecha base',fechas)
    if st.button('Generar pronóstico'):
        p=requests.post(f"{API_URL}/dengue/predict",json={'fecha':fecha},timeout=60).json()
        a,b,c=st.columns(3)
        a.metric('Casos pronosticados',p['casos_pronosticados'])
        b.metric('Fecha base',p['fecha_base'])
        c.metric('Fecha objetivo',p['fecha_pronostico'])
        graf=pd.DataFrame({'Tipo':['Última observación','Pronóstico'],'Casos':[p['casos_ultima_observacion'],p['casos_pronosticados_decimal']]})
        st.plotly_chart(px.bar(graf,x='Tipo',y='Casos',title='Pronóstico LSTM'),use_container_width=True)
except Exception as e:
    st.error(str(e))

st.divider()
st.header('3️⃣ Recomendación preventiva')
col1,col2=st.columns(2)
with col1:
    casos=st.number_input('Casos pronosticados (4 semanas)',min_value=0.0,value=300.0)
    umbral=st.number_input('Umbral endémico',min_value=1.0,value=250.0)
with col2:
    lluvia=st.number_input('Lluvia acumulada 14 días (mm)',min_value=0.0,value=50.0)
    temperatura=st.number_input('Temperatura máxima (°C)',value=27.0)
if st.button('Generar recomendación preventiva'):
    payload={'casos_pred_4sem':casos,'casos_umbral_endemico':umbral,'lluvia_14d_mm':lluvia,'temp_max':temperatura,'aegypti_detectado':clase_detectada=='aegypti'}
    rec=requests.post(f"{API_URL}/action/recommend",json=payload,timeout=60).json()
    st.success(f"Acción recomendada: {rec.get('accion_legible','')}")
    if 'confianza' in rec:
        st.metric('Confianza',f"{float(rec['confianza']):.2%}")
    st.info(rec.get('motivo',''))

st.divider()
st.header('4️⃣ Análisis histórico del dengue')
try:
    df=pd.read_csv('data/raw/RNN/vectoria_casos_cantonal_anual_real.csv')
    anio=st.selectbox('Seleccione un año',sorted(df['anio'].unique()))
    datos=df[df['anio']==anio]
    prov=datos.groupby('provincia')['casos_dengue'].sum().reset_index()
    st.plotly_chart(px.bar(prov,x='provincia',y='casos_dengue',color='provincia',title=f'Casos por provincia ({anio})'),use_container_width=True)
    st.subheader('Top 10 cantones con más casos')
    st.dataframe(datos.sort_values('casos_dengue',ascending=False).head(10),use_container_width=True)
except Exception as e:
    st.error(f'Error cargando datos históricos: {e}')

st.divider()
st.warning('VectorIA es una herramienta académica de apoyo a la vigilancia epidemiológica y no realiza diagnósticos médicos.')
