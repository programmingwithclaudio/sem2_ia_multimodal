"""
Pipeline CORREGIDO - Sistema de Recomendación de Productos
FIX: Embeddings compatibles con búsqueda de lenguaje natural

Cambios clave:
1. crear_texto_producto() genera texto más natural
2. Metadata completa en Redis (nombre original + texto procesado)
3. Validación de dimensiones
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
import faiss
import redis
import json
import pickle
from typing import List, Dict, Tuple
from datetime import datetime
import logging


# CONFIGURACIÓN
class Config:
    """Configuración centralizada"""
    BASE_PATH = Path("datasets")
    INPUT_CSV = BASE_PATH / "catalogo_agente_ventas_ia.csv"
    PROCESSED_PATH = BASE_PATH / "processed"
    EMBEDDINGS_PATH = PROCESSED_PATH / "embeddings"
    
    CHUNK_SIZE = 100
    EMBEDDING_DIM = 384
    FAISS_NLIST = 50
    
    REDIS_HOST = 'localhost'
    REDIS_PORT = 6379
    REDIS_DB = 0
    
    MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'
    
    @classmethod
    def setup(cls):
        cls.PROCESSED_PATH.mkdir(exist_ok=True)
        cls.EMBEDDINGS_PATH.mkdir(exist_ok=True)


# PREPROCESAMIENTO
class DataProcessor:
    def __init__(self):
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def validar_datos(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info(f"Validando {len(df)} registros...")
        inicial = len(df)
        
        df = df.drop_duplicates(subset=['ID_Producto'], keep='first')
        
        campos_requeridos = [
            'ID_Producto', 'Codigo_Producto', 'Nombre_Producto',
            'Categoria', 'Subcategoria', 'Precio_Lista'
        ]
        df = df.dropna(subset=campos_requeridos)
        
        for col in ['Nombre_Producto', 'Categoria', 'Subcategoria']:
            df[col] = df[col].str.strip().str.title()
        
        df = df[df['Precio_Lista'] > 0]
        
        final = len(df)
        self.logger.info(f"Validación: {inicial} → {final} registros")
        return df
    
    def optimizar_tipos(self, df: pd.DataFrame) -> pd.DataFrame:
        optimizations = {
            'ID_Producto': 'int32',
            'Stock_Actual': 'int32',
            'Precio_Lista': 'float32',
            'Costo_Estandar': 'float32',
            'Margen_Bruto_%': 'float32',
            'Categoria': 'category',
            'Subcategoria': 'category',
            'Color': 'category',
        }
        
        for col, dtype in optimizations.items():
            if col in df.columns:
                try:
                    df[col] = df[col].astype(dtype)
                except Exception as e:
                    self.logger.warning(f"No se pudo convertir {col}: {e}")
        
        return df
    
    def agregar_features_calculados(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'Precio_Lista' in df.columns and 'Costo_Estandar' in df.columns:
            df['Ratio_Precio_Costo'] = (
                df['Precio_Lista'] / df['Costo_Estandar'].replace(0, 1)
            )
        
        if 'Unidades_Vendidas_Historicas' in df.columns:
            df['Velocidad_Venta'] = pd.qcut(
                df['Unidades_Vendidas_Historicas'], 
                q=5, 
                labels=['Muy_Lenta', 'Lenta', 'Media', 'Rápida', 'Muy_Rápida'],
                duplicates='drop'
            )
        
        return df
    
    def procesar_a_parquet(self, input_csv: Path, output_path: Path):
        self.logger.info(f"Leyendo CSV: {input_csv}")
        df = pd.read_csv(input_csv, encoding='utf-8')
        
        df = self.validar_datos(df)
        df = self.optimizar_tipos(df)
        df = self.agregar_features_calculados(df)
        
        output_file = output_path / "productos_procesados.parquet"
        df.to_parquet(output_file, engine='pyarrow', compression='snappy', index=False)
        
        metadata = {
            'fecha_proceso': datetime.now().isoformat(),
            'total_productos': len(df),
            'categorias': df['Categoria'].nunique(),
            'columnas': list(df.columns),
        }
        
        with open(output_path / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"✅ Parquet guardado: {output_file}")
        return df


# GENERACIÓN DE EMBEDDINGS
class EmbeddingGenerator:
    """CRÍTICO: Genera embeddings compatibles con búsqueda en lenguaje natural"""
    
    def __init__(self, model_name: str = Config.MODEL_NAME):
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Cargando modelo: {model_name}")
        
        self.model = SentenceTransformer(model_name)
        self.model.max_seq_length = 128
        self.scaler = StandardScaler()
        self.scaler_fitted = False
        
    def crear_texto_producto(self, row: pd.Series) -> str:
        """
        🔧 FIX CRÍTICO: Genera texto en lenguaje NATURAL
        
        ANTES (mal): "Mountain-200 Black 42 categoría Bikes tipo Mountain"
        AHORA (bien): "bicicleta de montaña Mountain-200 negra talla 42"
        """
        
        # 1. Nombre del producto (tal cual)
        nombre = row['Nombre_Producto']
        
        # 2. Categoría en lenguaje natural
        categoria_map = {
            'Bikes': 'bicicleta',
            'Accessories': 'accesorio',
            'Clothing': 'ropa',
            'Components': 'componente'
        }
        categoria = categoria_map.get(row['Categoria'], row['Categoria'].lower())
        
        # 3. Subcategoría en lenguaje natural
        subcategoria_map = {
            'Mountain Bikes': 'de montaña',
            'Road Bikes': 'de ruta',
            'Touring Bikes': 'de turismo',
            'Helmets': 'casco',
            'Gloves': 'guantes',
            'Jerseys': 'jersey',
        }
        subcategoria = subcategoria_map.get(row['Subcategoria'], row['Subcategoria'].lower())
        
        # 4. Construir descripción natural
        partes = [categoria, subcategoria, nombre]
        
        # 5. Agregar atributos importantes si existen
        if pd.notna(row.get('Color')):
            color_map = {
                'Black': 'negro', 'White': 'blanco', 'Red': 'rojo',
                'Blue': 'azul', 'Yellow': 'amarillo', 'Silver': 'plateado'
            }
            color = color_map.get(row['Color'], row['Color'].lower())
            partes.append(f"color {color}")
        
        if pd.notna(row.get('Talla')):
            partes.append(f"talla {row['Talla']}")
        
        # 6. Agregar contexto de clasificación (afecta relevancia)
        if pd.notna(row.get('Clasificacion_Ventas')):
            clasificacion_map = {
                'High': 'más vendido',
                'Medium': 'venta regular',
                'Low': 'venta ocasional'
            }
            clasificacion = clasificacion_map.get(
                row['Clasificacion_Ventas'], 
                row['Clasificacion_Ventas'].lower()
            )
            partes.append(clasificacion)
        
        texto_final = " ".join(partes)
        
        # DEBUG: Imprimir primeros 3 productos para verificar
        if row.name < 3:
            self.logger.info(f"\n📝 Ejemplo texto generado #{row.name}:")
            self.logger.info(f"   Original: {row['Nombre_Producto']}")
            self.logger.info(f"   Procesado: {texto_final}")
        
        return texto_final
    
    def extraer_features_numericos(self, df: pd.DataFrame) -> np.ndarray:
        """Extrae exactamente 5 features numéricos"""
        features_cols = [
            'Precio_Lista',
            'Margen_Bruto_%',
            'Stock_Actual',
            'Unidades_Vendidas_Historicas',
            'Score_Prioridad_Venta'
        ]
        
        # Asegurar que existan todas las columnas
        for col in features_cols:
            if col not in df.columns:
                df[col] = 0
        
        features = df[features_cols].fillna(0).values
        
        if not self.scaler_fitted:
            self.scaler.fit(features)
            self.scaler_fitted = True
        
        return self.scaler.transform(features).astype('float32')
    
    def generar_embeddings_batch(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Genera embeddings + retorna textos procesados para Redis"""
        
        self.logger.info(f"Generando embeddings para {len(df)} productos...")
        
        # 1. Crear textos en lenguaje natural
        textos_procesados = [self.crear_texto_producto(row) for _, row in df.iterrows()]
        
        # 2. Generar embeddings (384 dim)
        embeddings_texto = self.model.encode(
            textos_procesados,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        
        # 3. Features numéricos (5 dim)
        features_num = self.extraer_features_numericos(df)
        
        # 4. Concatenar: 384 + 5 = 389 dimensiones
        embeddings_final = np.hstack([
            embeddings_texto,
            features_num
        ]).astype('float32')
        
        # 5. Validar dimensiones
        expected_dim = 389
        actual_dim = embeddings_final.shape[1]
        
        if actual_dim != expected_dim:
            raise ValueError(
                f"❌ Dimensión incorrecta: esperado {expected_dim}, obtenido {actual_dim}"
            )
        
        self.logger.info(f"✅ Embeddings generados: {embeddings_final.shape}")
        self.logger.info(f"   Dimensión texto: 384")
        self.logger.info(f"   Dimensión features: 5")
        self.logger.info(f"   Dimensión total: {actual_dim}")
        
        return embeddings_final, textos_procesados
    
    def guardar_embeddings(self, embeddings: np.ndarray, output_path: Path):
        emb_file = output_path / "product_embeddings.npy"
        np.save(emb_file, embeddings)
        
        scaler_file = output_path / "scaler.pkl"
        with open(scaler_file, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        self.logger.info(f"💾 Embeddings guardados: {emb_file}")


# ALMACENAMIENTO FAISS + REDIS
class VectorStore:
    """CRÍTICO: Almacena metadata completa en Redis"""
    
    def __init__(self, dimension: int, nlist: int = Config.FAISS_NLIST):
        self.logger = logging.getLogger(__name__)
        self.dimension = dimension
        self.nlist = nlist
        
        self.redis = redis.Redis(
            host=Config.REDIS_HOST,
            port=Config.REDIS_PORT,
            db=Config.REDIS_DB,
            decode_responses=True
        )
        
        self.index = None
        
    def crear_indice_faiss(self, embeddings: np.ndarray):
        n_vectors = embeddings.shape[0]
        self.logger.info(f"Creando índice FAISS para {n_vectors} vectores...")
        
        if n_vectors < 1000:
            self.index = faiss.IndexFlatIP(self.dimension)
            self.logger.info("Usando IndexFlatIP (búsqueda exacta)")
        else:
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(
                quantizer, 
                self.dimension, 
                min(self.nlist, n_vectors // 10)
            )
            self.logger.info("Entrenando índice IVF...")
            self.index.train(embeddings)
        
        self.index.add(embeddings)
        self.logger.info(f"✅ Índice creado: {self.index.ntotal} vectores")
        
    def guardar_indice_faiss(self, output_path: Path):
        index_file = output_path / "productos.index"
        faiss.write_index(self.index, str(index_file))
        self.logger.info(f"💾 Índice FAISS guardado: {index_file}")
    
    def indexar_metadata_redis(self, df: pd.DataFrame, textos_procesados: List[str]):
        """
        🔧 FIX: Guarda tanto nombre original como texto procesado
        """
        self.logger.info("Indexando metadata en Redis...")
        
        # Limpiar Redis antes de indexar
        self.logger.info("🗑️  Limpiando Redis...")
        self.redis.flushdb()
        
        pipe = self.redis.pipeline()
        
        for idx, (row_idx, row) in enumerate(df.iterrows()):
            producto_id = idx  # Usar índice secuencial
            
            # Hash principal con metadata COMPLETA
            producto_key = f"producto:{producto_id}"
            producto_data = {
                # Datos originales
                'id': str(producto_id),
                'id_producto_original': str(row['ID_Producto']),
                'codigo': str(row['Codigo_Producto']),
                'name': str(row['Nombre_Producto']),  # ← nombre original
                'nombre': str(row['Nombre_Producto']),
                'category': str(row['Categoria']),
                'categoria': str(row['Categoria']),
                'subcategory': str(row['Subcategoria']),
                'subcategoria': str(row['Subcategoria']),
                'list_price': str(float(row['Precio_Lista'])),
                'precio': str(float(row['Precio_Lista'])),
                'stock_quantity': str(int(row.get('Stock_Actual', 0))),
                'stock': str(int(row.get('Stock_Actual', 0))),
                'margen': str(float(row.get('Margen_Bruto_%', 0))),
                
                # 🔧 NUEVO: Texto procesado para debugging
                'texto_procesado': textos_procesados[idx]
            }
            
            # Campos opcionales
            if pd.notna(row.get('Color')):
                producto_data['color'] = str(row['Color'])
            if pd.notna(row.get('Talla')):
                producto_data['talla'] = str(row['Talla'])
                producto_data['size'] = str(row['Talla'])
            if pd.notna(row.get('Clasificacion_Ventas')):
                producto_data['clasificacion'] = str(row['Clasificacion_Ventas'])
            
            pipe.hmset(producto_key, producto_data)
            
            # Índices adicionales
            pipe.sadd(f"cat:{row['Categoria']}", producto_id)
            pipe.sadd(f"subcat:{row['Subcategoria']}", producto_id)
            pipe.zadd("precios", {str(producto_id): float(row['Precio_Lista'])})
            
        pipe.execute()
        
        # Guardar mapeo en archivo para debugging
        mapeo_file = Config.EMBEDDINGS_PATH / "mapeo_indices.json"
        mapeo = {
            idx: {
                'id_original': int(row['ID_Producto']),
                'nombre': str(row['Nombre_Producto']),
                'texto_procesado': textos_procesados[idx]
            }
            for idx, (_, row) in enumerate(df.iterrows())
        }
        
        with open(mapeo_file, 'w', encoding='utf-8') as f:
            json.dump(mapeo, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"✅ {len(df)} productos indexados en Redis")
        self.logger.info(f"📄 Mapeo guardado en: {mapeo_file}")


# PIPELINE 
def ejecutar_pipeline_completo():
    """Pipeline corregido con validaciones"""
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("PIPELINE CORREGIDO - v2.0")
    logger.info("=" * 80)
    
    Config.setup()
    
    # PASO 1: Procesar CSV
    logger.info("\n[PASO 1/3] Procesando CSV → Parquet...")
    processor = DataProcessor()
    df = processor.procesar_a_parquet(Config.INPUT_CSV, Config.PROCESSED_PATH)
    
    # PASO 2: Generar Embeddings
    logger.info("\n[PASO 2/3] Generando embeddings...")
    emb_generator = EmbeddingGenerator()
    embeddings, textos_procesados = emb_generator.generar_embeddings_batch(df)
    emb_generator.guardar_embeddings(embeddings, Config.EMBEDDINGS_PATH)
    
    # PASO 3: Indexar
    logger.info("\n[PASO 3/3] Indexando en FAISS + Redis...")
    vector_store = VectorStore(dimension=embeddings.shape[1])
    vector_store.crear_indice_faiss(embeddings)
    vector_store.guardar_indice_faiss(Config.EMBEDDINGS_PATH)
    vector_store.indexar_metadata_redis(df, textos_procesados)  # ← Pasamos textos procesados
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ PIPELINE COMPLETADO")
    logger.info("=" * 80)
    logger.info(f"📁 Archivos en: {Config.PROCESSED_PATH}")
    logger.info(f"🔍 Productos indexados: {len(df)}")
    logger.info(f"📊 Dimensión embeddings: {embeddings.shape[1]}")
    
    return vector_store, df

if __name__ == "__main__":
    vector_store, df = ejecutar_pipeline_completo()
    
    print("\n" + "="*80)
    print("VALIDACIÓN DEL PIPELINE")
    print("="*80)
    
    # Verificar primeros 3 productos
    for idx in range(min(3, len(df))):
        data = vector_store.redis.hgetall(f"producto:{idx}")
        print(f"\n🔍 Producto {idx}:")
        print(f"   Nombre original: {data.get('name')}")
        print(f"   Texto procesado: {data.get('texto_procesado')}")
        print(f"   Categoría: {data.get('category')}")
        print(f"   Precio: ${data.get('list_price')}")
