# 🧠 Proyecto: Agente Multimodal con Redis y Gradio

## 📋 Requisitos del Entorno

| Componente  | Versión                | Descripción                                                |
| ----------- | ---------------------- | ---------------------------------------------------------- |
| **Windows** | 10                     | Sistema operativo base.                                    |
| **Python**  | 3.12.10                | Lenguaje principal para scripts ETL y embeddings.          |
| **Redis**   | 3.x                    | Almacenamiento clave-valor para cache y consultas rápidas. |
| **FFmpeg**  | 8.x (Essentials Build) | Requerido para el procesamiento de audio.                  |
| **FAISS**   | Compatible con CPU     | Motor de búsqueda vectorial optimizado.                    |

---

## ⚙️ Instalación de Dependencias

### 1️⃣ Instalar FFmpeg

**Opción A – Desde WinGet (recomendada):**

```bash
winget install "FFmpeg (Essentials Build)"
```

**Opción B – Manual:**

* Descarga desde [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)
* Extrae el contenido en una ruta segura, por ejemplo:

  ```
  C:\ffmpeg\
  ```

**Verificar instalación:**

```bash
ffmpeg -version
```

**Ubicación del ejecutable (PowerShell):**

```powershell
Get-Command ffmpeg | Select-Object Source
```

**Configurar la ruta en el script principal: `agents_to_gradio_redis3.py`**

```python
FFMPEG_BIN = r"C:\Users\oak\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg.Essentials_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0-essentials_build\bin"
```

---

## 🧩 Estructura del Proyecto

```bash
sem2_ia_multimodal/
│
├── sql_to_csv.py               # No ejecutar (solo referencia)
├── csv_to_embeddings.py        # Paso 1 - Genera embeddings desde CSV
├── agents_to_gradio_redis3.py  # Paso 2 - Lanza el agente con Redis y Gradio
└── data/
    ├── raw/                    # Archivos CSV originales
    ├── processed/              # Chunks procesados
    └── embeddings/             # Vectores FAISS y Redis
```

---

## 🚀 Flujo de Ejecución

Antes de ejecutar, asegúrate de establecer el directorio base del proyecto:

```python
BASE_DIR = r"C:\Users\oak\ti\tasks_frogrames\sem2_ia_multimodal"
```

### Paso 1: Generar Embeddings

Ejecuta el script:

```bash
python csv_to_embeddings.py
```

Este proceso:

* Lee el CSV de entrada (≈ 38 millones de registros)
* Divide el dataset en *chunks* de 500 registros
* Genera embeddings y los almacena localmente

### Paso 2: Levantar el Agente

Ejecuta el script:

```bash
python agents_to_gradio_redis3.py
```

El agente:

* Conecta a Redis 3
* Carga los embeddings en FAISS
* Expone una interfaz **Gradio** para interacción en tiempo real
  *(por ejemplo, búsqueda semántica, consultas naturales o audio-inputs)*

---

## 🔁 Flujo General del Sistema

[![flujo-mermaid.png](https://i.postimg.cc/D0N0YTGK/flujo-mermaid.png)](https://postimg.cc/YL1thT8X)

---

## 🧮 Recomendaciones de Rendimiento

* Utiliza **FAISS con índices HNSW o IVFFlat** para acelerar búsquedas vectoriales.
* Aumenta el `maxmemory` de Redis si el dataset excede 4 GB.
* Ejecuta el proceso en **modo batch** durante las horas de baja carga.
* Considera un **SSD** para almacenar los embeddings (mayor IOPS).

---

## 🧰 Comandos Útiles

| Tarea                        | Comando                           |
| ---------------------------- | --------------------------------- |
| Verificar Redis activo       | `redis-cli ping`                  |
| Monitorear memoria Redis     | `info memory`                     |
| Verificar instalación Python | `python --version`                |
| Instalar dependencias Python | `pip install -r requirements.txt` |

---
