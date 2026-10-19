# Evaluación Parcial N°1: Machine Learning
## Inteligencia Musical y Predicción de Popularidad de Canciones (Spotify Tracks)
|Campo|Detalle|
|---|---|
|Integrantes| Catalina Villegas y Javiera Fuenzalida|
|Sección| 002D|
|Asignatura| Machine Learning|
|Docente| Ignacio Meza|

### **Metodología CRISP-DM**
El proyecto se desarrolla bajo el estándar CRISP-DM (*Cross-Industry Standard Process for Data Mining*):
1. **Comprensión del Negocio:** Definición de objetivos para estimar la popularidad y optimizar decisiones de difusión musical.
2. **Comprensión de los Datos:** Exploración del dataset Spotify Tracks (125 géneros y atributos de audio).
3. **Preparación de los Datos:** Limpieza de duplicados, tratamiento de valores nulos, acotamiento de outliers y codificación.
4. **Modelado:** Selección e implementación de algoritmos de regresión/clasificación.
5. **Evaluación:** Validación técnica de métricas predictivas frente a KPIs de negocio.
6. **Despliegue:** Documentación reproducible en formato Notebook (`.ipynb`) y Markdown (`.md`).

### **Fuentes de Datos y Herramientas Utilizadas**
* **Fuente de Datos:** Dataset `Spotify_Tracks_Dataset.csv` de Spotify con ~114,000 canciones y 20 variables (metadatos y variables acústicas como `danceability`, `energy`, `loudness`, `valence`, etc. Sin contar `unnamed`).
* **Google Colab / Jupyter:** Entorno colaborativo en la nube que garantiza ejecución reproducible del código.
* **Pandas & NumPy:** Procesamiento estructurado, limpieza y manipulación eficiente de matrices.
* **Matplotlib & Seaborn:** Generación de gráficos exploratorios para detección de patrones y sesgos.

### **Contexto y Descripción del problema de negocio**

Las plataformas de streaming musical operan en un entorno altamente dinámico donde miles de canciones son lanzadas diariamente. Para discográficas, creadores de contenido y la propia plataforma, anticipar el nivel de adopción o popularidad de un track permite optimizar la inversión en marketing, definir listas de reproducción editoriales y maximizar el engagement de la audiencia. Este proyecto aborda la necesidad de identificar qué características acústicas y metadatos influyen directamente en la popularidad de una pista musical en Spotify.

El dataset entregado contiene información sobre las pistas de Spotify alrededor de 125 géneros, además de contener información y especificaciones sobre los audios y su metadata.

**Objetivos del Proyecto**

El objetivo general es desarrollar una solución analítica de Machine Learning para predecir la popularidad de las canciones en Spotify a partir de sus atributos acústicos y metadatos.

* Implementar el pipeline completo de ciencia de datos bajo la metodología CRISP-DM.

* Analizar y preparar el conjunto de datos resolviendo problemas de duplicados, valores faltantes y sesgos de distribución.

* Identificar los factores acústicos clave (p. ej., danceability, energy, valence) correlacionados con el éxito comercial de una canción.

### **KPIs Principales del Negocio**

**Tasa Global de Hits**

Porcentaje del catálogo con `popularity >= 70`.

**Popularidad Promedio Por género**

Identificación de géneros con mayor atractivo comercial.

**Índice de Afinidad Sonora**

Grado de correlación entre atributos acústicos clave y la popularidad.


**Duración Óptima**

Intervalo de duración (en minutos) que concentra los niveles más altos de popularidad.

## **Analizando y tratando el DataSet**

### Librerias

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
```

###Carga Principal

```python
file_id = '1jFyz3QYDC5jhnQbgFOTukCZo5oeUtLHX'
url = f'https://drive.google.com/uc?export=download&id={file_id}'
df = pd.read_csv(url)
print(f'Dataset cargado: {df.shape[0]} filas x {df.shape[1]} columnas')
print('-' * 60)

print('\n1. PRIMERAS 5 FILAS (ejemplo completado):')
display(df.head())
print('-' * 60)
print(f"\n2. INFORMACIÓN:\n")
print(df.info())
print(f"\n3. ESTADÍSTICAS:\n{df.describe()}")
```

```text
Dataset cargado: 114000 filas x 21 columnas
------------------------------------------------------------

1. PRIMERAS 5 FILAS (ejemplo completado):
```

```text
   Unnamed: 0                track_id                 artists  \
0           0  5SuOikwiRyPMVoIQDJUgSV             Gen Hoshino   
1           1  4qPNDBW1i3p13qLCt0Ki3A            Ben Woodward   
2           2  1iJBSr7s7jYXzM8EGcbK5b  Ingrid Michaelson;ZAYN   
3           3  6lfxq3CG4xtTiEg7opyCyx            Kina Grannis   
4           4  5vjLSffimiIP26QG5WcN2K        Chord Overstreet   

                                          album_name  \
0                                             Comedy   
1                                   Ghost (Acoustic)   
2                                     To Begin Again   
3  Crazy Rich Asians (Original Motion Picture Sou...   
4                                            Hold On   

                   track_name  popularity  duration_ms  explicit  \
0                      Comedy          73       230666     False   
1            Ghost - Acoustic          55       149610     False   
2              To Begin Again          57       210826     False   
3  Can't Help Falling In Love          71       201933     False   
4                     Hold On          82       198853     False   

   danceability  energy  ...  loudness  mode  speechiness  acousticness  \
0         0.676  0.4610  ...    -6.746     0       0.1430        0.0322   
1         0.420  0.1660  ...   -17.235     1       0.0763        0.9240   
2         0.438  0.3590  ...    -9.734     1       0.0557        0.2100   
3         0.266  0.0596  ...   -18.515     1       0.0363        0.9050   
4         0.618  0.4430  ...    -9.681     1       0.0526        0.4690   

   instrumentalness  liveness  valence    tempo  time_signature  track_genre  
0          0.000001    0.3580    0.715   87.917               4     acoustic  
1          0.000006    0.1010    0.267   77.489               4     acoustic  
2          0.000000    0.1170    0.120   76.332               4     acoustic  
3          0.000071    0.1320    0.143  181.740               3     acoustic  
4          0.000000    0.0829    0.167  119.949               4     acoustic  

[5 rows x 21 columns]
```

```text
------------------------------------------------------------

2. INFORMACIÓN:

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 114000 entries, 0 to 113999
Data columns (total 21 columns):
 #   Column            Non-Null Count   Dtype  
---  ------            --------------   -----  
 0   Unnamed: 0        114000 non-null  int64  
 1   track_id          114000 non-null  object 
 2   artists           113999 non-null  object 
 3   album_name        113999 non-null  object 
 4   track_name        113999 non-null  object 
 5   popularity        114000 non-null  int64  
 6   duration_ms       114000 non-null  int64  
 7   explicit          114000 non-null  bool   
 8   danceability      114000 non-null  float64
 9   energy            114000 non-null  float64
 10  key               114000 non-null  int64  
 11  loudness          114000 non-null  float64
 12  mode              114000 non-null  int64  
 13  speechiness       114000 non-null  float64
 14  acousticness      114000 non-null  float64
 15  instrumentalness  114000 non-null  float64
 16  liveness          114000 non-null  float64
 17  valence           114000 non-null  float64
 18  tempo             114000 non-null  float64
 19  time_signature    114000 non-null  int64  
 20  track_genre       114000 non-null  object 
dtypes: bool(1), float64(9), int64(6), object(5)
memory usage: 17.5+ MB
None

3. ESTADÍSTICAS:
          Unnamed: 0     popularity   duration_ms   danceability  \
count  114000.000000  114000.000000  1.140000e+05  114000.000000   
mean    56999.500000      33.238535  2.280292e+05       0.566800   
std     32909.109681      22.305078  1.072977e+05       0.173542   
min         0.000000       0.000000  0.000000e+00       0.000000   
25%     28499.750000      17.000000  1.740660e+05       0.456000   
50%     56999.500000      35.000000  2.129060e+05       0.580000   
75%     85499.250000      50.000000  2.615060e+05       0.695000   
max    113999.000000     100.000000  5.237295e+06       0.985000   

              energy            key       loudness           mode  \
count  114000.000000  114000.000000  114000.000000  114000.000000   
mean        0.641383       5.309140      -8.258960       0.637553   
std         0.251529       3.559987       5.029337       0.480709   
min         0.000000       0.000000     -49.531000       0.000000   
25%         0.472000       2.000000     -10.013000       0.000000   
50%         0.685000       5.000000      -7.004000       1.000000   
75%         0.854000       8.000000      -5.003000       1.000000   
max         1.000000      11.000000       4.532000       1.000000   

         speechiness   acousticness  instrumentalness       liveness  \
count  114000.000000  114000.000000     114000.000000  114000.000000   
mean        0.084652       0.314910          0.156050       0.213553   
std         0.105732       0.332523          0.309555       0.190378   
min         0.000000       0.000000          0.000000       0.000000   
25%         0.035900       0.016900          0.000000       0.098000   
50%         0.048900       0.169000          0.000042       0.132000   
75%         0.084500       0.598000          0.049000       0.273000   
max         0.965000       0.996000          1.000000       1.000000   

             valence          tempo  time_signature  
count  114000.000000  114000.000000   114000.000000  
mean        0.474068     122.147837        3.904035  
std         0.259261      29.978197        0.432621  
min         0.000000       0.000000        0.000000  
25%         0.260000      99.218750        4.000000  
50%         0.464000     122.017000        4.000000  
75%         0.683000     140.071000        4.000000  
max         0.995000     243.372000        5.000000  
```

```python
print(f"Datos duplicados: {df.duplicated().sum()}")
print('-'*60)
print(f"Datos faltantes: \n{df.isna().sum()}")
print('-'*60)

print(f"Datos fuera de rango: {((df['popularity'] < 0) | (df['popularity'] > 100)).sum()}")
```

```text
Datos duplicados: 0
------------------------------------------------------------
Datos faltantes: 
Unnamed: 0          0
track_id            0
artists             1
album_name          1
track_name          1
popularity          0
duration_ms         0
explicit            0
danceability        0
energy              0
key                 0
loudness            0
mode                0
speechiness         0
acousticness        0
instrumentalness    0
liveness            0
valence             0
tempo               0
time_signature      0
track_genre         0
dtype: int64
------------------------------------------------------------
Datos fuera de rango: 0
```

###Tratamiento de Datos

```python
print(f"Moda de 'artistas': {df['artists'].mode()[0]}, {(df['artists'] == (df['artists'].mode()[0])).sum()} canciones en que se repiten")
print(f"Moda de 'album_name': {df['album_name'].mode()[0]}, {(df['album_name'] == (df['album_name'].mode()[0])).sum()} canciones en que se repiten")
print(f"Moda de 'track_name':  {df['track_name'].mode()[0]}, {(df['track_name'] == (df['track_name'].mode()[0])).sum()} canciones en que se repiten")
```

```text
Moda de 'artistas': The Beatles, 279 canciones en que se repiten
Moda de 'album_name': Alternative Christmas 2022, 195 canciones en que se repiten
Moda de 'track_name':  Run Rudolph Run, 151 canciones en que se repiten
```

```python
#Rellenar columnas restantes con la moda de cada columna
df_clean = df.copy()
df_clean['artists'] = df['artists'].fillna(df['artists'].mode()[0])
df_clean['album_name'] = df['album_name'].fillna(df['album_name'].mode()[0])
df_clean['track_name'] = df['track_name'].fillna(df['track_name'].mode()[0])

print(f"Datos faltantes luego de rellenar: \n{df_clean.isna().sum()}")
```

```text
Datos faltantes luego de rellenar: 
Unnamed: 0          0
track_id            0
artists             0
album_name          0
track_name          0
popularity          0
duration_ms         0
explicit            0
danceability        0
energy              0
key                 0
loudness            0
mode                0
speechiness         0
acousticness        0
instrumentalness    0
liveness            0
valence             0
tempo               0
time_signature      0
track_genre         0
dtype: int64
```

### **Análisis Exploratorio de Datos (EDA) y KPIs**

```python
# KPI 1: Graficos Hits

UMBRAL_HIT = 70
sns.set_theme(style="whitegrid")

df_clean['is_hit'] = (df_clean['popularity'] >= UMBRAL_HIT).astype(int)

hit_ratio_global = df_clean['is_hit'].mean() * 100
print(f"Tasa de Hits Potenciales (Hit Ratio Global): {hit_ratio_global:.2f}%\n")

hit_ratio_por_genero = df_clean.groupby('track_genre')['is_hit'].agg(
    total_canciones='count',
    total_hits='sum',
    hit_ratio_pct=lambda x: x.mean() * 100
).reset_index().sort_values(by='hit_ratio_pct', ascending=False)

print("Hit Ratio por Género:")
print(hit_ratio_por_genero)


fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Gráfico 1: Histograma de Popularidad
sns.histplot(data=df_clean, x='popularity', kde=True, ax=axes[0], color='#1DB954', bins=20)
axes[0].axvline(UMBRAL_HIT, color='red', linestyle='--', linewidth=2, label=f'Umbral Hit (≥ {UMBRAL_HIT})')
axes[0].set_title('Distribución de Popularidad con Umbral de Hit')
axes[0].set_xlabel('Popularidad (0 - 100)')
axes[0].set_ylabel('Cantidad de Canciones')
axes[0].legend()

# Gráfico 2: Contador de Hits vs No Hits
sns.countplot(data=df_clean, x='is_hit', ax=axes[1], palette='viridis')
axes[1].set_title(f'Distribución de Hits en Spotify (Popularidad >= {UMBRAL_HIT})')
axes[1].set_xlabel('¿Es Hit?')
axes[1].set_ylabel('Cantidad de Canciones')
axes[1].set_xticks([0, 1])
axes[1].set_xticklabels(['No Hit (<70)', 'Hit (>=70)'])

total = len(df_clean)
for p in axes[1].patches:
    percentage = f'{100 * p.get_height() / total:.1f}%'
    
    x_pos = p.get_x() + p.get_width() / 2  
    y_pos = p.get_height()                 
    
    axes[1].annotate(percentage, (x_pos, y_pos), 
                     ha='center', va='bottom', 
                     fontsize=10, fontweight='bold', 
                     xytext=(0, 5), textcoords='offset points')

# Ajustar diseño y mostrar
plt.tight_layout()
plt.savefig('graficos_HIT.png')
plt.show()

```

```text
Tasa de Hits Potenciales (Hit Ratio Global): 4.80%

Hit Ratio por Género:
    track_genre  total_canciones  total_hits  hit_ratio_pct
80          pop             1000         317           31.7
20        dance             1000         245           24.5
31      electro             1000         241           24.1
65        k-pop             1000         225           22.5
53        house             1000         221           22.1
..          ...              ...         ...            ...
93      romance             1000           0            0.0
97    sertanejo             1000           0            0.0
96        samba             1000           0            0.0
105       study             1000           0            0.0
108       tango             1000           0            0.0

[114 rows x 4 columns]
```

```text
/tmp/ipykernel_2720/1731525370.py:32: FutureWarning: 

Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.

  sns.countplot(data=df_clean, x='is_hit', ax=axes[1], palette='viridis')
```

```text
<Figure size 1400x500 with 2 Axes>
```

```python
# KPI 2: Top 10 Géneros por Popularidad Promedio Y Géneros con mayor Porcentaje Hit

import matplotlib.pyplot as plt
import seaborn as sns

# Gráfico Izquierdo: Top 10 Géneros por Hit Ratio
hit_ratio_por_genero = df_clean.groupby('track_genre')['is_hit'].agg(
    total_canciones='count',
    total_hits='sum',
    hit_ratio_pct=lambda x: (x.mean() * 100)
).reset_index().sort_values(by='hit_ratio_pct', ascending=False).head(10)

# Gráfico Derecho: Top 10 Géneros por Popularidad Promedio
genre_popularity = df_clean.groupby('track_genre')['popularity'].mean().sort_values(ascending=False).head(10)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))  # Una fila, dos columnas

# Hit Ratio por Género
sns.barplot(
    data=hit_ratio_por_genero,
    x='hit_ratio_pct',
    y='track_genre',
    hue='track_genre',
    palette='magma',
    legend=False,
    ax=axes[0]
)
axes[0].set_title('Top 10 Géneros con Mayor Hit Ratio (%)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Tasa de Hits Potenciales (%)')
axes[0].set_ylabel('Género Musical')

# Popularidad Promedio por Género
sns.barplot(
    x=genre_popularity.values,
    y=genre_popularity.index,
    hue=genre_popularity.index,
    palette='viridis',
    legend=False,
    ax=axes[1]
)
axes[1].set_title('Top 10 Géneros Más Populares (Promedio)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Popularidad Promedio')
axes[1].set_ylabel('')

plt.tight_layout()
plt.savefig('graficos_generos.png')
plt.show()

```

```text
<Figure size 1800x700 with 2 Axes>
```

```python
# KPI 3: Mapa de Calor del Índice de Afinidad Sonora

columnas_interes = [
    'popularity', 'duration_ms', 'danceability', 'energy',
    'key', 'loudness', 'mode', 'speechiness', 'acousticness',
    'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature'
]

df_numerico = df_clean[columnas_interes]

matriz_corr = df_numerico.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(
    matriz_corr,
    annot=True,          # Muestra el valor numérico en cada celda
    cmap='coolwarm',     # Escala de azul (negativo) a rojo (positivo)
    fmt='.2f',           # Muestra solo 2 decimales
    linewidths=0.5,      # Añade una línea de separación entre celdas
    vmin=-1, vmax=1      # Normaliza la escala de color entre -1 y 1
)

plt.title('Matriz de Correlación - Características Musicales', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('mapa_calor.png')
plt.show()
```

```text
<Figure size 1000x800 with 2 Axes>
```

```python
# KPI 4: Duración Óptima vs Popularidad

# Convertir la duración a minutos para que el gráfico sea más legible
df_clean['duration_min'] = df['duration_ms'] / 60000


plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df_clean,
    x='duration_min',
    y='popularity',
    alpha=0.1,
    s=5,
    color='purple'
)

plt.title('Relación entre la Duración y la Popularidad de las Canciones', fontsize=14)
plt.xlabel('Duración (minutos)', fontsize=12)
plt.ylabel('Popularidad', fontsize=12)
plt.xlim(0, 10)

plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig('grafico_relacion.png')
plt.show()

```

```text
<Figure size 1000x600 with 1 Axes>
```

## **Evaluación de Sesgos, Ética y Privacidad de Datos**

### **Detección de Sesgos**
**Sesgo Geográfico y Comercial:** Los géneros pertenecientes a la industria anglosajona o latina comercial (p. ej., pop, dance,electro, k-pop) presentan puntuaciones históricamente más elevadas en el algoritmo de Spotify respecto a géneros regionales o tradicionales.

### Aspectos Éticos y Privacidad
* **Privacidad (GDPR / Ley 19.628):** El conjunto de datos recopila metadatos públicos de canciones e información sonora sin incluir datos de carácter personal identificable (PII) de los oyentes o usuarios.
* **Uso Ético de Algoritmos:** Las recomendaciones basadas en este modelo deben ser utilizadas como **herramientas de apoyo técnico** para productores e independientes, evitando descartar obras musicales de valor artístico por el solo hecho de poseer métricas distantes del estándar comercial dominante.

### **Conclusiones del Proyecto**
**Impacto de Atributos Acústicos:** El análisis exploratorio confirmó que la popularidad en Spotify guarda una relación estrecha con la sonoridad y el ritmo. Atributos como danceability y loudness presentan una correlación positiva con la popularidad, mientras que altos índices de acousticness e instrumentalness tienden a limitar el alcance comercial masivo.  

**Ventana de Duración Comercial:** El análisis del KPI de Duración Óptima determinó que las canciones con mayor popularidad se concentran en un rango acotado de entre 2 y 4 minutos. Pistas que superan este intervalo sufren una disminución progresiva en sus métricas de popularidad.  

**Distribución de Hits y Desbalance:** La Tasa Global de Hits (popularidad $\ge 70$) representa un porcentaje reducido del catálogo total analizado a través de los 125 géneros. Esta asimetría confirma la presencia de un desbalance de datos relevante que debe abordarse durante la fase de entrenamiento.  

**Gobernanza y Rigor Metodológico:** La estructuración del pipeline mediante CRISP-DM garantizó la detección temprana de duplicados y la evaluación de sesgos algorítmicos, asegurando un tratamiento ético de los metadatos sin comprometer la privacidad.  

Estrategia MusicalOptimización de Estructura SonoraMantener la duración entre 2:00 y 3:30 minutos e intensificar valores de danceability para maximizar la tasa de retención y la incorporación en listas de reproducción.  PreprocesamientoTratamiento del DesbalanceAplicar técnicas de rebalanceo (como SMOTE o ajuste de pesos de clase en Scikit-Learn) si la popularidad se aborda como una variable binaria (Hit vs. No Hit).  Ingeniería de FeaturesCreación de Variables CompuestasGenerar nuevas variables combinadas (p. ej., un índice de energía-sonoridad) para capturar interacciones no lineales entre las propiedades del audio.  Modelado FuturoUso de Ensambles No LinealesImplementar algoritmos basados en árboles de decisión (Random Forest, XGBoost) que superen las limitaciones de la regresión lineal ante la complejidad de los metadatos.  Ética y DiversidadMitigación del Sesgo de PromociónUtilizar el modelo como apoyo técnico y no como filtro absoluto, evitando penalizar géneros tradicionales o independientes que no buscan un perfil comercial estándar.  