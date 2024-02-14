import streamlit as st

# Importa las bibliotecas necesarias para tu análisis
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from scipy.stats import norm
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from google.oauth2 import service_account

SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
# KEY = 'credenciales/key.json'
KEY = st.secrets["google_sheets_creds"] 
SPREADSHEET_ID = '1VmL3MzzXOCarN9YRpHRCVdowd1jUR51SmLGxHH_K1Aw'

creds = service_account.Credentials.from_service_account_file(KEY, scopes=SCOPES)
service = build('sheets', 'v4', credentials=creds)

# Define la función para obtener los datos de la hoja de cálculo
def get_sheet_data(service, spreadsheet_id, range_name):
    # Llama al método values().get para obtener los datos de la hoja de cálculo
    sheet = service.spreadsheets()
    result = sheet.values().get(spreadsheetId=spreadsheet_id, range=range_name).execute()
    values = result.get('values', [])
    return values

# Define el rango de celdas que deseas leer
range_name = "'Hoja 1'!C1:AE"  # Por ejemplo, lee desde la celda A1 hasta la columna E (ajusta el rango según tus necesidades)
# Llama a la función para obtener los datos
sheet_data = get_sheet_data(service, SPREADSHEET_ID, range_name)
# Convierte los datos en un DataFrame de Pandas
df_sheet = pd.DataFrame(sheet_data[1:], columns=sheet_data[0])
# Define el rango de celdas que deseas leer 
range_name_puestos = "'Hoja 3'!A1:Z"  # Por ejemplo, lee desde la celda A1 hasta la columna E (ajusta el rango según tus necesidades)
# Llama a la función para obtener los datos
sheet_data_puestos = get_sheet_data(service, SPREADSHEET_ID, range_name_puestos)
# Convierte los datos en un DataFrame de Pandas
df_sheet_puestos = pd.DataFrame(sheet_data_puestos[1:], columns=sheet_data_puestos[0])
# Define el rango de celdas que deseas leer 
range_name_preguntas = "'Respuestas de formulario 1'!G1:EA"  # Por ejemplo, lee desde la celda A1 hasta la columna E (ajusta el rango según tus necesidades)
# Llama a la función para obtener los datos
sheet_data_preguntas = get_sheet_data(service, SPREADSHEET_ID, range_name_preguntas)
# Convierte los datos en un DataFrame de Pandas
df_sheet_preguntas = pd.DataFrame(sheet_data_preguntas[1:], columns=sheet_data_preguntas[0])

# Carga los datos y realiza el análisis
# Aquí puedes poner tu código de análisis
# empleados = pd.read_excel('Empleados.xlsx')
empleados = df_sheet.loc[df_sheet.iloc[:, 0] != "-"]
empleados['Edad'] = empleados['Edad'].astype(int)
empleados['Experiencia (años)'] = empleados['Experiencia (años)'].astype(int)
# areas_oficina = pd.read_excel('AreasOficina.xlsx')
areas_oficina = df_sheet_puestos 
# Calcular el producto punto entre los valores de cada empleado/líder y los valores de las áreas de trabajo
producto_punto = empleados.iloc[:, 4:].astype(int).dot(areas_oficina.iloc[:, 1:].astype(int).T)

# Iterar sobre cada área de trabajo y sus valores de índice
for index, area_trabajo in enumerate(areas_oficina['Área de trabajo']):
    # Obtener los valores correspondientes de producto_punto
    valores_producto_punto = producto_punto.iloc[:, index]
    # Crear una nueva columna en empleados y asignar los valores correspondientes de producto_punto
    empleados[area_trabajo] = valores_producto_punto

# Calcular el producto punto entre los valores de cada empleado/líder y los valores de las áreas de trabajo
producto_punto = empleados.iloc[:,-5:]

# Rango de número de clusters a probar
n_samples = len(producto_punto)
rangos_clusters = range(2, min(n_samples, 11))
# Almacenar las puntuaciones de silueta y la inercia
puntuaciones_silueta = []
inercias = []
# Calcular las métricas para cada número de clusters
for n_clusters in rangos_clusters:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(producto_punto)
    silhouette_avg = silhouette_score(producto_punto, cluster_labels)
    inertia = kmeans.inertia_
    puntuaciones_silueta.append(silhouette_avg)
    inercias.append(inertia)

# Encontrar el mejor valor de silueta y su índice
mejor_silueta = max(puntuaciones_silueta)
indice_mejor_silueta = puntuaciones_silueta.index(mejor_silueta)
mejor_numero_clusters = rangos_clusters[indice_mejor_silueta]

# Graficar los resultados
fig_cluster = plt.figure(figsize=(10, 3))
plt.subplot(1, 2, 1)
plt.plot(rangos_clusters, puntuaciones_silueta, marker='o')
plt.xlabel('Número de clusters')
plt.ylabel('Puntuación de silueta')
plt.title('Análisis de silueta')

plt.subplot(1, 2, 2)
plt.plot(rangos_clusters, inercias, marker='o')
plt.xlabel('Número de clusters')
plt.ylabel('Inercia')
plt.title('Análisis de la inercia')
#plt.tight_layout()

# Aplicar KMeans para obtener los clusters
kmeans = KMeans(n_clusters=mejor_numero_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(empleados.iloc[:,-5:])

# Agregar las etiquetas de cluster al DataFrame original
empleados['Cluster'] = cluster_labels
empleados['Cluster'] = empleados['Cluster'].astype(str)
 
# Obtener los valores únicos de la columna 'Area_de_trabajo'
areas_trabajo_unique = areas_oficina['Área de trabajo'].unique()

# Definir los colores específicos para cada área de trabajo
colores = ['red', 'blue', 'green', 'yellow', 'purple', 'cyan', 'magenta', 'orange', 'lightgreen', 'lightblue', 'pink', 'brown', 'gray', 'black']

# Crear el diccionario de mapeo de colores
color_map = dict(zip(areas_trabajo_unique, colores))

# Crear una nueva columna 'Color' en tu DataFrame que contenga los colores correspondientes a cada cluster
empleados['Color'] = empleados['Área de trabajo'].map(color_map)

# Data de la afinidad de los empleados
data = empleados.iloc[:, -7:-2]

# Nivel de confianza (95%)
alpha = 0.05

# Valor de z alfa/2
z_alpha_2 = norm.ppf(1 - alpha / 2)

# Configuración de subplots
fig_Hist, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 8))

# Iterar sobre cada columna del DataFrame y generar un gráfico para cada una
for i, column in enumerate(data.columns):
    # Datos específicos de la columna actual
    column_data = data[column]

    # Calcular la media y la desviación estándar
    mean = column_data.mean()
    std_dev = column_data.std()

    # Tamaño de la muestra
    n = len(column_data)

    # Calcular el intervalo de confianza para la media
    margin_of_error = z_alpha_2 * (std_dev / np.sqrt(n))
    confidence_interval = (mean - margin_of_error, mean + margin_of_error)

    # Calcular el índice de confianza para cada valor
    confidence_indices = np.where(
        (column_data >= confidence_interval[0]) & (column_data <= confidence_interval[1]),
        1,
        0
    )

    # Graficar el histograma con el intervalo de confianza y una línea de suavizado
    row = i // 2
    col = i % 2
    ax = axes[row, col]

    ax.hist(column_data, bins=5, density=True, alpha=0.7, color='blue', edgecolor='black')
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mean, std_dev)
    ax.plot(x, p, 'k', linewidth=2, label='Densidad de probabilidad')
    ax.axvline(x=confidence_interval[0], color='red', linestyle='--', linewidth=2, label='Intervalo de confianza')
    ax.axvline(x=confidence_interval[1], color='red', linestyle='--', linewidth=2)
    ax.set_xlabel(column)
    ax.set_ylabel('Densidad de probabilidad')
    ax.set_title(f'Hist. {column} con Intervalo de Confianza (95%)')
    ax.legend()

plt.tight_layout()

# Nivel de confianza (95%)
alpha = 0.05

# Valor de z alfa/2
z_alpha_2 = norm.ppf(1 - alpha / 2)

# Calcular intervalos de confianza para cada columna y agregar valores mayores al 95%
for column in data.columns:
    # Calcular la media y la desviación estándar
    mean = data[column].mean()
    std_dev = data[column].std()

    # Tamaño de la muestra
    n = len(data[column])

    # Calcular el intervalo de confianza para la media
    margin_of_error = z_alpha_2 * (std_dev / np.sqrt(n))
    confidence_interval = (mean - margin_of_error, mean + margin_of_error)

    # Agregar una columna al DataFrame con valores booleanos que indican si el valor supera el intervalo de confianza
    data[f'{column}_Above_CI'] = (data[column] > confidence_interval[1]).astype(int)

# Calcular el valor máximo de producto punto para cada empleado
max_producto_punto = empleados.iloc[:, -6:-2].max(axis=1)

# Calcular la media del producto punto para cada empleado
media_producto_punto = empleados.iloc[:, -6:-2].mean(axis=1)

# Calcular la dispersión del producto punto para cada empleado
dispersion_producto_punto = empleados.iloc[:, -6:-2].std(axis=1)

# Representa el porcentaje de dispersion respecto de la media
porcentaje_desviacion = (dispersion_producto_punto / media_producto_punto)


# Agregar las nuevas columnas al DataFrame
empleados['Max Producto Punto'] = max_producto_punto
empleados['Media Producto Punto'] = media_producto_punto
empleados['Dispersión Producto Punto'] = dispersion_producto_punto
empleados['Porc. de dispersion respecto de la media'] = porcentaje_desviacion
# Suponiendo que deseas agregar las últimas cuatro columnas de data a empleados
empleados = pd.concat([empleados, data.iloc[:, -5:]], axis=1)
togroupby = empleados.iloc[:, -5:].columns
try:
    # Definir grupos según el tipo de afinidad
    grupos_afinidad = empleados.groupby([np.array(togroupby)])

    # Inicializar diccionarios para almacenar estadísticas
    estadisticas_por_grupo = {
        'Cantidad de trabajadores': [],
        'Edad promedio': [],
        'Experiencia promedio (años)': [],
        'Edad mínima': [],
        'Edad máxima': [],
        'Empleados': []  # Agregar una nueva lista para almacenar los empleados de cada grupo
    }

    # Calcular las estadísticas para cada grupo
    for nombre_grupo, grupo in grupos_afinidad:
        estadisticas_por_grupo['Cantidad de trabajadores'].append(len(grupo))
        estadisticas_por_grupo['Edad promedio'].append(grupo['Edad'].mean())
        estadisticas_por_grupo['Experiencia promedio (años)'].append(grupo['Experiencia (años)'].mean())
        estadisticas_por_grupo['Edad mínima'].append(grupo['Edad'].min())
        estadisticas_por_grupo['Edad máxima'].append(grupo['Edad'].max())
        estadisticas_por_grupo['Empleados'].append(list(grupo['Nombre']))  # Agregar la lista de empleados

    # Convertir a DataFrame para mostrar las estadísticas
    estadisticas_df = pd.DataFrame(estadisticas_por_grupo)
    estadisticas_df.index = [str(index) for index in grupos_afinidad.groups.keys()]
except:
    pass

areas_trabajo = areas_oficina.iloc[:,1:].columns

# Crear una nueva columna 'Color' en tu DataFrame que contenga los colores correspondientes a cada cluster
empleados['Color'] = empleados['Área de trabajo'].map(color_map)

# Crear el gráfico de dispersión 3D con Plotly utilizando la columna 'Color' como colores de los marcadores
fig_clustering = go.Figure(data=go.Scatter3d(
    x=empleados['Facturación'],
    y=empleados['Sistemas'],
    z=empleados['Producción'],
    mode='markers',
    marker=dict(
        size=10,
        color=empleados['Color'],  # Utilizar la columna 'Color' como colores de los marcadores
        opacity=1
    ),
    text=empleados['Nombre']+', '+empleados['Área de trabajo'],
    name='Empleados'
))

# Añadir etiquetas y título
fig_clustering.update_layout(
    scene=dict(
        xaxis=dict(title='Facturación', tickfont=dict(size=14)),
        yaxis=dict(title='Sistemas', tickfont=dict(size=14)),
        zaxis=dict(title='Producción', tickfont=dict(size=14))
    ),
    title='Gráfico de dispersión 3D',
    width=1200,  # Ajustar el ancho del gráfico
    height=600  # Ajustar la altura del gráfico
)


# -----------------------------------------------------------------------------------------

# Configurar la página para que ocupe toda la pantalla
st.set_page_config(layout="wide")

# # Crear el sidebar para registrar los valores del usuario
# st.sidebar.title("Registro de Preguntas")

# # Solicitar al usuario que ingrese valores del 0 al 10 para cada área de trabajo
# for area_trabajo in areas_trabajo:
#     valor = st.sidebar.number_input(f"Ingrese un valor del 0 al 10 para {area_trabajo}:", min_value=0, max_value=10)
#     valores_usuario[area_trabajo] = valor

# # Mostrar los valores ingresados por el usuario en un DataFrame
# df_usuario = pd.DataFrame([valores_usuario])
# st.write("Valores ingresados por el usuario:")
# st.write(df_usuario)
# Define el contenido de la aplicación web
st.title('Análisis de Datos')

# Filtros
st.sidebar.title('Filtros')
# Agregar un slider para seleccionar el rango de edad
edad_minima = st.sidebar.slider('Edad mínima', min_value=20, max_value=70, value=20)
edad_maxima = st.sidebar.slider('Edad máxima', min_value=20, max_value=70, value=70)
# Filtro por nombre
nombre_busqueda = st.sidebar.multiselect("Buscar por nombre", empleados['Nombre'])
# Filtro por experiencia
experiencia_minima = st.sidebar.slider('Experiencia mínima (años)', min_value=0, max_value=30, value=0)
experiencia_maxima = st.sidebar.slider('Experiencia máxima (años)', min_value=0, max_value=30, value=30)
# Agregar un multiselect para seleccionar las áreas de trabajo
areas_trabajo_seleccionadas = st.sidebar.multiselect('Áreas de trabajo', areas_oficina.iloc[1:])
# Filtrar el DataFrame de empleados según los filtros seleccionados
empleados_filtrados = empleados.copy()
if nombre_busqueda:
    empleados_filtrados = empleados_filtrados[empleados_filtrados['Nombre'].isin(nombre_busqueda)]
empleados_filtrados = empleados_filtrados[(empleados_filtrados['Experiencia (años)'].astype(int) >= experiencia_minima) & 
                                        (empleados_filtrados['Experiencia (años)'].astype(int) <= experiencia_maxima)]
if areas_trabajo_seleccionadas:
    empleados_filtrados = empleados_filtrados[empleados_filtrados['Área de trabajo'].isin(areas_trabajo_seleccionadas)]

fig_notas = plt.figure(figsize=(10, 5))

# Crear el primer subplot para el promedio de notas
plt.subplot(1, 2, 1)
plt.bar(range(len(empleados_filtrados)), empleados_filtrados.iloc[:, 1])
plt.xticks(range(len(empleados_filtrados)), empleados_filtrados['Nombre'], rotation=90)  
plt.xlabel('Nombre de Empleado')
plt.ylabel('Edades')
plt.title('Registro de edades')

# Crear el segundo subplot para el promedio de notas
plt.subplot(1, 2, 2)
plt.bar(range(len(empleados_filtrados)), empleados_filtrados.iloc[:, 2])
plt.xticks(range(len(empleados_filtrados)), empleados_filtrados['Nombre'], rotation=90)  
plt.xlabel('Nombre de Empleado')
plt.ylabel('Antiguedad')
plt.title('Registro de antiguedad')

tab1, tab2, tab3 = st.tabs(["🗃 Datos","📈 Graficos","Examinar"])

recortado = empleados_filtrados.columns[:36].tolist() + empleados_filtrados.columns[-4:].tolist()

# Mostrar la tabla filtrada en Streamlit
with tab1:
    st.subheader("Datos")
    st.table(empleados_filtrados[recortado])

# Mostrar grafico de conocimientos
with tab2:
    st.subheader("Edad y experiencia")
    st.pyplot(fig_notas)
    st.subheader("Estadisticos sobre los puestos")
    st.pyplot(fig_Hist)
    st.subheader("Analisis de agrupamiento")
    st.pyplot(fig_cluster)
    st.subheader("Agrupamiento")
    st.plotly_chart(fig_clustering)

with tab3:
    # Crear el sidebar para registrar los valores del usuario
    st.title("Evaluacion de afinidad")
    st.write("El cálculo se basa en el valor de afinidad obtenido mediante el producto punto, seleccionando aquellos valores que superen el rango máximo del intervalo de confianza (95%) calculado a partir de los datos actuales de los empleados en cada puesto.")
    

    # Crear un diccionario para almacenar los valores ingresados por el usuario
    valores_usuario = {}
    
    # Crear un formulario para que el usuario ingrese los valores del 0 al 10 para cada área de trabajo
    with st.form("formulario_valores_usuario"):
        for area_trabajo in areas_trabajo:
            # Utilizar radio buttons para seleccionar el valor del 0 al 10
            valor = st.radio(f"Valor para {area_trabajo}:", options=list(range(11)), horizontal=True)
            valores_usuario[area_trabajo] = valor

        # Agregar un botón para enviar el formulario
        enviar_formulario = st.form_submit_button("Enviar")

    # Mostrar el DataFrame solo si todas las preguntas han sido respondidas
    if enviar_formulario:
        # Resto del código para procesar los datos y mostrar resultados...
        df_usuario = pd.DataFrame([valores_usuario])
        # Calcular el producto punto entre los valores de cada empleado/líder y los valores de las áreas de trabajo
        producto_punto = df_usuario.dot(areas_oficina.iloc[:, 1:].T)
        afinidades = []
        # Iterar sobre cada área de trabajo y sus valores de índice
        for index, area_trabajo in enumerate(areas_oficina['Área de trabajo']):
            # Obtener los valores correspondientes de producto_punto
            valores_producto_punto = producto_punto.iloc[:, index]
            
            # Crear una nueva columna en empleados y asignar los valores correspondientes de producto_punto
            afinidades.append(valores_producto_punto.values[0])
        # Definir los nombres de las características
        feature_names = areas_oficina['Área de trabajo'].unique()

        # Nivel de confianza (95%)
        alpha = 0.05

        # Valor de z alfa/2
        z_alpha_2 = norm.ppf(1 - alpha / 2)

        # Calcular la media y la desviación estándar
        mean = np.array(empleados.iloc[:, -16:-11]).mean()
        std_dev = np.array(empleados.iloc[:, -16:-11]).std()

        # Tamaño de la muestra
        n = len(empleados.iloc[:, -16:-11])

        # Calcular el intervalo de confianza para la media
        margin_of_error = z_alpha_2 * (std_dev / np.sqrt(n))

        # El intervalo de confidencia calcula el rango del 95% de confianza para las afinidades calculadas
        confidence_interval = (mean - margin_of_error, mean + margin_of_error)

        # Se selecciona el valor de afinidad que esté por encima del rango de confianza del 95%
        puesto = (afinidades > confidence_interval[1]).astype(int)

        # Obtener los nombres de las características que tienen un valor de 1 en el array de entrada
        selected_features = [feature_names[i] for i, val in enumerate(puesto) if val == 1]
        
        # Verificar si hay afinidades (valores de 1 en el puesto)
        if 1 in puesto:
            valor = [afinidades[i] for i, val in enumerate(puesto) if val == 1]
            st.write(f"Se calcula una afinidad de {valor[0]} puntos para el puesto en {selected_features[0]}")
        else:
            st.write("No existen afinidades para los resultados dados")

        # Encontrar el índice donde se encuentra el valor 1
        indice = np.where(np.array(puesto) == 1)[0]

        # Crear el diccionario de mapeo de colores
        color_maped = {str(i): colores[i] for i in range(len(colores))}

        # # Mapear los valores de los clusters a colores específicos
        # color_maped = {"0": 'red', "1": 'blue', "2": 'green', "3": 'yellow', "4": 'purple'}

        # Crear el gráfico de dispersión 3D con Plotly utilizando la columna 'Color' como colores de los marcadores
        fig_pred = go.Figure()

        # Agregar los puntos de los empleados
        fig_pred.add_trace(go.Scatter3d(
            x=empleados['Facturación'],
            y=empleados['Sistemas'],
            z=empleados['Producción'],
            mode='markers',
            marker=dict(
                size=8,
                color=empleados['Color'],  # Utilizar la columna 'Color' como colores de los marcadores
                opacity=0.8
            ),
            text=empleados['Nombre']+', '+empleados['Área de trabajo'],
            name='Empleados'
        ))

        # Agregar un punto adicional
        fig_pred.add_trace(go.Scatter3d(
            x=[afinidades[0]],
            y=[afinidades[1]],
            z=[afinidades[2]],
            mode='markers',
            marker=dict(
                size=10,
                color=color_maped[str(indice[0])],  # Color del punto adicional
                opacity=1.0,
                symbol='circle-open'
            ),
            name='Evaluado'
        ))

        # Añadir etiquetas y título
        fig_pred.update_layout(
            scene=dict(
                xaxis=dict(title='Facturación'),
                yaxis=dict(title='Sistemas'),
                zaxis=dict(title='Producción')
            ),
            title='Agrupamiento por afinidades',
            width=1200,  # Ajustar el ancho del gráfico
            height=600  # Ajustar la altura del gráfico
        )
        st.subheader("Evaluacion")
        st.plotly_chart(fig_pred)

# Agrega elementos interactivos como sliders, selectores, etc.

footer = """
<style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: white;
        color: black;
        text-align: center;
    }
    .footer a:link, .footer a:visited {
        color: blue;
        background-color: transparent;
        text-decoration: underline;
    }
    .footer a:hover, .footer a:active {
        color: red;
        background-color: transparent;
        text-decoration: underline;
    }
</style>
<div class="footer">
    <p>Developed with ❤ by <a href="https://ar.linkedin.com/in/rodrigo-gonz%C3%A1lez-8796a8197" target="_blank">RG</a></p>
</div>
"""
# Contenido de tu aplicación
if __name__ == '__main__':
    # Agregar el footer al final de la aplicación
    st.markdown(footer, unsafe_allow_html=True)