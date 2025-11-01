import os
import sys
import json
import time
import asyncio
import redis
import faiss
import numpy as np
import pickle
import re
from io import BytesIO
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import gradio as gr
import threading
import subprocess
from pydub import AudioSegment

# ==================== CONFIGURACIÓN FFmpeg ====================
FFMPEG_BIN = r"C:\Users\oak\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg.Essentials_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0-essentials_build\bin"
os.environ["PATH"] = FFMPEG_BIN + os.pathsep + os.environ.get("PATH", "")
FFMPEG_PATH = os.path.join(FFMPEG_BIN, "ffmpeg.exe")
AudioSegment.converter = FFMPEG_PATH
AudioSegment.ffmpeg = FFMPEG_PATH

# ==================== INICIALIZACIÓN ====================
load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    print("❌ OpenAI API Key NO configurada")
    sys.exit(1)

print(f"✓ OpenAI API Key configurada: {openai_api_key[:8]}...")

MODEL = "gpt-4o-mini"
openai = OpenAI()

# ==================== RUTAS ====================
BASE_DIR = os.getenv("BASE_DIR")
DATASETS_DIR = os.path.join(BASE_DIR, "datasets", "processed")
EMBEDDINGS_DIR = os.path.join(DATASETS_DIR, "embeddings")
FAISS_INDEX_PATH = os.path.join(EMBEDDINGS_DIR, "productos.index")
SCALER_PATH = os.path.join(EMBEDDINGS_DIR, "scaler.pkl")
TEMP_AUDIO_DIR = os.path.join(BASE_DIR, "public", "audio_test_ias")

# ==================== CONFIGURACIÓN ====================
REDIS_CONFIG = {
    'host': 'localhost',
    'port': 6379,
    'db': 0,
    'decode_responses': True
}

WHATSAPP_NUMBER = "51907321211"
WHATSAPP_DISPLAY = "+51 907 321 211"

EMBEDDING_MODEL = 'paraphrase-multilingual-MiniLM-L12-v2'
# 🔧 FIX: Dimensión correcta (384 texto + 5 features = 389)
EMBEDDING_TEXT_DIM = 384
NUM_FEATURES = 5
TOTAL_DIM = 389  # ✅ CORREGIDO

# ==================== UTILIDADES DE NORMALIZACIÓN ====================
class DataNormalizer:
    """Normaliza datos del cliente antes de almacenar en Redis"""
    
    @staticmethod
    def sanitize_for_json(obj):
        """
        🔧 FIX CRÍTICO: Convierte tipos numpy/pandas a tipos nativos de Python
        Evita errores: "Object of type int64 is not JSON serializable"
        """
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: DataNormalizer.sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [DataNormalizer.sanitize_for_json(item) for item in obj]
        else:
            return obj
    
    @staticmethod
    def parse_customer_input(text: str) -> Dict[str, str]:
        """
        Parsea input del usuario que puede venir como:
        - "valerio quispe, 904875166"
        - "valerio quispe 904875166"
        - "nombre: juan perez, tel: 999888777"
        
        Returns: {'name': str, 'phone': str}
        """
        text = text.strip()
        
        # Patrón 1: "nombre, teléfono"
        if ',' in text:
            parts = text.split(',')
            if len(parts) >= 2:
                name = parts[0].strip()
                phone = DataNormalizer.extract_phone(parts[1])
                if name and phone:
                    return {
                        'name': DataNormalizer.normalize_name(name),
                        'phone': DataNormalizer.normalize_phone(phone)
                    }
        
        # Patrón 2: "nombre teléfono" (buscar número al final)
        phone_match = re.search(r'(\d{9,11})\s*$', text)
        if phone_match:
            phone = phone_match.group(1)
            name = text[:phone_match.start()].strip()
            if name and phone:
                return {
                    'name': DataNormalizer.normalize_name(name),
                    'phone': DataNormalizer.normalize_phone(phone)
                }
        
        # No se pudo parsear
        return {'name': '', 'phone': ''}
    
    @staticmethod
    def extract_phone(text: str) -> str:
        """Extrae número telefónico de un texto"""
        # Remover espacios y caracteres no numéricos
        text = re.sub(r'[^\d]', '', text)
        
        # Buscar secuencia de 9-11 dígitos
        match = re.search(r'\d{9,11}', text)
        return match.group(0) if match else ''
    
    @staticmethod
    def normalize_name(name: str) -> str:
        """Normaliza nombre: capitaliza, remueve extras"""
        name = name.strip()
        # Remover etiquetas como "nombre:", "name:", etc
        name = re.sub(r'^(nombre|name|client[e]?)\s*:\s*', '', name, flags=re.IGNORECASE)
        # Capitalizar cada palabra
        name = ' '.join(word.capitalize() for word in name.split())
        return name
    
    @staticmethod
    def normalize_phone(phone: str) -> str:
        """Normaliza teléfono: solo dígitos, formato consistente"""
        # Remover todo excepto dígitos
        phone = re.sub(r'[^\d]', '', phone)
        
        # Si empieza con +51 o 51, removerlo
        if phone.startswith('51') and len(phone) > 9:
            phone = phone[2:]
        
        return phone
    
    @staticmethod
    def validate_customer_data(name: str, phone: str) -> Tuple[bool, str]:
        """Valida datos del cliente"""
        if not name or len(name) < 2:
            return False, "Nombre debe tener al menos 2 caracteres"
        
        if not phone or len(phone) < 9:
            return False, "Teléfono debe tener al menos 9 dígitos"
        
        if not phone.isdigit():
            return False, "Teléfono debe contener solo números"
        
        return True, "OK"

# ==================== MÉTRICAS ====================
@dataclass
class PerformanceMetrics:
    search_time: float = 0
    openai_call: float = 0
    audio_processing: float = 0
    total_time: float = 0

# ==================== MOTOR DE BÚSQUEDA ====================
class ProductSearchEngine:
    def __init__(self):
        self.model = None
        self.index = None
        self.redis_client = None
        self.scaler = None
        self.initialized = False
        
    def initialize(self):
        try:
            print("🔧 Inicializando motor de búsqueda...")
            
            print(f"📥 Cargando modelo: {EMBEDDING_MODEL}")
            self.model = SentenceTransformer(EMBEDDING_MODEL)
            self.model.max_seq_length = 128
            
            if os.path.exists(SCALER_PATH):
                with open(SCALER_PATH, 'rb') as f:
                    self.scaler = pickle.load(f)
            
            if not os.path.exists(FAISS_INDEX_PATH):
                raise FileNotFoundError(f"❌ Índice FAISS no encontrado: {FAISS_INDEX_PATH}")
            
            self.index = faiss.read_index(FAISS_INDEX_PATH)
            
            # 🔧 FIX: Validar dimensión correcta
            if self.index.d != TOTAL_DIM:
                print(f"⚠️  ADVERTENCIA: Dimensión del índice ({self.index.d}) != esperada ({TOTAL_DIM})")
                print(f"   Si hay errores, regenera el índice con el pipeline corregido")
            
            print(f"✓ Índice FAISS: {self.index.ntotal} productos, {self.index.d}D")
            
            self.redis_client = redis.Redis(**REDIS_CONFIG)
            self.redis_client.ping()
            
            self.initialized = True
            print("✅ Motor inicializado\n")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            raise
    
    def create_query_embedding(self, query: str) -> np.ndarray:
        try:
            text_embedding = self.model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )[0]
            
            numeric_features = np.zeros(NUM_FEATURES, dtype='float32')
            query_embedding = np.hstack([text_embedding, numeric_features])
            
            query_embedding = query_embedding.astype('float32')
            norm = np.linalg.norm(query_embedding)
            if norm > 0:
                query_embedding = query_embedding / norm
            
            return query_embedding
            
        except Exception as e:
            print(f"❌ Error embedding: {e}")
            return None
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        if not self.initialized:
            self.initialize()
        
        try:
            print(f"\n🔍 Búsqueda: '{query}'")
            
            query_embedding = self.create_query_embedding(query)
            if query_embedding is None:
                return []
            
            query_embedding = query_embedding.reshape(1, -1).astype('float32')
            faiss.normalize_L2(query_embedding)
            
            scores, indices = self.index.search(query_embedding, top_k * 2)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:
                    continue
                
                producto_data = self.redis_client.hgetall(f"producto:{idx}")
                if not producto_data:
                    continue
                
                result = {
                    'id': idx,
                    'name': producto_data.get('name', 'N/A'),
                    'category': producto_data.get('category', 'N/A'),
                    'subcategory': producto_data.get('subcategory', 'N/A'),
                    'price': producto_data.get('list_price', '0'),
                    'stock': producto_data.get('stock_quantity', '0'),
                    'color': producto_data.get('color', 'N/A'),
                    'size': producto_data.get('size', 'N/A'),
                    'score': float(score)
                }
                
                results.append(result)
                if len(results) >= top_k:
                    break
            
            print(f"✅ {len(results)} resultados\n")
            return results
            
        except Exception as e:
            print(f"❌ Error búsqueda: {e}")
            return []

search_engine = ProductSearchEngine()

# ==================== ESTADO DE CONVERSACIÓN ====================
class ConversationState:
    def __init__(self):
        self.history: List[Dict] = []
        self.last_search_results: List[Dict] = []
        self.last_product_mentioned: Optional[Dict] = None
        self.cart: List[Dict] = []
        self.customer_data: Dict = {}
        self.order_confirmed: bool = False
        self.whatsapp_link: str = ""
        self.order_id: str = ""
        self.waiting_confirmation: bool = False
        self.metrics: PerformanceMetrics = PerformanceMetrics()
    
    def add_message(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        if len(self.history) > 20:
            self.history = self.history[-20:]
    
    def clear(self):
        self.history = []
        self.last_search_results = []
        self.last_product_mentioned = None
        self.cart = []
        self.customer_data = {}
        self.order_confirmed = False
        self.whatsapp_link = ""
        self.order_id = ""
        self.waiting_confirmation = False

conv_state = ConversationState()

# ==================== HERRAMIENTAS ====================

def search_products_tool(query: str, top_k: int = 5) -> str:
    """Busca productos"""
    print(f"\n🔧 TOOL: search_products('{query}')")
    
    start = time.time()
    results = search_engine.search(query, top_k)
    conv_state.last_search_results = results
    
    if results:
        conv_state.last_product_mentioned = results[0]
    
    conv_state.metrics.search_time = time.time() - start
    
    if not results:
        return json.dumps({
            "status": "no_results",
            "message": "No se encontraron productos"
        }, ensure_ascii=False)
    
    formatted = {
        "status": "success",
        "total_found": len(results),
        "products": []
    }
    
    for i, p in enumerate(results, 1):
        formatted["products"].append({
            "rank": i,
            "id": int(p['id']),
            "name": str(p['name']),
            "category": f"{p['category']} > {p['subcategory']}",
            "price": f"${p['price']}",
            "stock": f"{p['stock']} unidades",
            "color": str(p.get('color', 'N/A')),
            "size": str(p.get('size', 'N/A'))
        })
    
    return json.dumps(formatted, ensure_ascii=False, indent=2)

def add_to_cart_tool(product_id: int, quantity: int = 1) -> str:
    """Agrega producto al carrito"""
    print(f"\n🔧 TOOL: add_to_cart(id={product_id}, qty={quantity})")
    
    # 🔧 FIX: Buscar por ID o por rank (posición en resultados)
    product = None
    
    # Primero intentar encontrar por ID exacto
    for p in conv_state.last_search_results:
        if p['id'] == product_id:
            product = p
            break
    
    # Si no encuentra, intentar por rank (1-based index)
    if not product and 1 <= product_id <= len(conv_state.last_search_results):
        product = conv_state.last_search_results[product_id - 1]
        print(f"   ℹ️  Interpretando ID como rank #{product_id}")
    
    if not product:
        available_ids = [p['id'] for p in conv_state.last_search_results]
        return json.dumps({
            "status": "error",
            "message": f"Producto {product_id} no encontrado. IDs disponibles: {available_ids}"
        }, ensure_ascii=False)
    
    # Verificar si ya está en el carrito
    for item in conv_state.cart:
        if item['product_id'] == product['id']:
            item['quantity'] = int(item['quantity']) + int(quantity)  # 🔧 FIX
            item['subtotal'] = float(item['price']) * int(item['quantity'])
            total = float(sum(float(i['subtotal']) for i in conv_state.cart))
            print(f"   ✅ Cantidad actualizada en carrito")
            return json.dumps({
                "status": "updated",
                "message": "Cantidad actualizada",
                "product_name": item['name'],
                "new_quantity": int(item['quantity']),
                "cart_total_amount": f"{total:.2f}"
            }, ensure_ascii=False)
    
    # Validar stock
    stock = int(product.get('stock', 0))
    if stock < quantity:
        return json.dumps({
            "status": "insufficient_stock",
            "message": f"Stock insuficiente. Disponible: {stock}",
            "product_name": product['name']
        }, ensure_ascii=False)
    
    # Agregar nuevo producto al carrito
    cart_item = {
        'product_id': int(product['id']),  # 🔧 FIX: Convertir a int nativo
        'name': str(product['name']),
        'price': float(product['price']),
        'quantity': int(quantity),
        'color': str(product.get('color', 'N/A')),
        'size': str(product.get('size', 'N/A')),
        'subtotal': float(product['price']) * int(quantity)
    }
    
    conv_state.cart.append(cart_item)
    total = sum(item['subtotal'] for item in conv_state.cart)
    
    print(f"   ✅ Producto agregado: {cart_item['name']}")
    print(f"   💰 Subtotal: S/ {cart_item['subtotal']:.2f}")
    print(f"   🛒 Total carrito: S/ {total:.2f}")
    
    return json.dumps({
        "status": "success",
        "message": "Producto agregado al carrito",
        "product_name": cart_item['name'],
        "quantity": quantity,
        "price": f"S/ {cart_item['price']:.2f}",
        "subtotal": f"S/ {cart_item['subtotal']:.2f}",
        "cart_total_amount": f"S/ {total:.2f}",
        "cart_items_count": len(conv_state.cart)
    }, ensure_ascii=False)

def collect_customer_data_tool(raw_input: str) -> str:
    """
    🔧 FIX: Recolecta y normaliza datos del cliente
    Acepta formato libre: "valerio quispe, 904875166" o "valerio quispe 904875166"
    """
    print(f"\n🔧 TOOL: collect_customer_data('{raw_input}')")
    
    # Parsear input
    parsed = DataNormalizer.parse_customer_input(raw_input)
    
    if not parsed['name'] or not parsed['phone']:
        return json.dumps({
            "status": "error",
            "message": "No pude entender los datos. Por favor ingresa: nombre, teléfono"
        }, ensure_ascii=False)
    
    # Validar
    valid, error_msg = DataNormalizer.validate_customer_data(
        parsed['name'], 
        parsed['phone']
    )
    
    if not valid:
        return json.dumps({
            "status": "error",
            "message": error_msg
        }, ensure_ascii=False)
    
    # Guardar datos normalizados
    conv_state.customer_data = {
        'name': parsed['name'],
        'phone': parsed['phone'],
        'email': None,
        'address': None,
        'collected_at': datetime.now().isoformat()
    }
    
    conv_state.waiting_confirmation = True
    
    print(f"✅ Datos normalizados:")
    print(f"   Nombre: {parsed['name']}")
    print(f"   Teléfono: {parsed['phone']}")
    
    return json.dumps({
        "status": "success",
        "message": "Datos registrados correctamente",
        "customer": conv_state.customer_data
    }, ensure_ascii=False)

def confirm_order_tool() -> str:
    """
    🔧 FIX: Confirma pedido - Compatible con Redis 3.0+
    """
    print(f"\n🔧 TOOL: confirm_order() - EJECUTANDO")
    
    if not conv_state.cart:
        return json.dumps({
            "status": "error",
            "message": "No hay productos en el carrito"
        }, ensure_ascii=False)
    
    if not conv_state.customer_data.get('name') or not conv_state.customer_data.get('phone'):
        return json.dumps({
            "status": "error",
            "message": "Faltan datos del cliente"
        }, ensure_ascii=False)
    
    order_id = f"ORD-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    total = float(sum(item['subtotal'] for item in conv_state.cart))
    
    # 🔧 FIX: Sanitizar carrito antes de crear orden
    cart_sanitized = []
    for item in conv_state.cart:
        cart_sanitized.append({
            'product_id': int(item['product_id']),
            'name': str(item['name']),
            'price': float(item['price']),
            'quantity': int(item['quantity']),
            'color': str(item.get('color', 'N/A')),
            'size': str(item.get('size', 'N/A')),
            'subtotal': float(item['subtotal'])
        })
    
    order = {
        'order_id': order_id,
        'customer': conv_state.customer_data,
        'items': cart_sanitized,
        'total_items': len(cart_sanitized),
        'total_amount': float(total),
        'currency': 'PEN',
        'status': 'pending_payment',
        'created_at': datetime.now().isoformat(),
        'whatsapp_number': WHATSAPP_DISPLAY
    }
    
    try:
        redis_client = search_engine.redis_client
        redis_client.ping()
        print("✅ Redis conectado")
        
        order_key = f"order:{order_id}"
        
        # 🔧 FIX: Método compatible con TODAS las versiones de Redis
        # En lugar de hset(key, mapping=dict), usar hmset o hset con pares
        order_data = {
            'order_id': order_id,
            'customer_name': order['customer']['name'],
            'customer_phone': order['customer']['phone'],
            'customer_email': order['customer'].get('email') or '',
            'customer_address': order['customer'].get('address') or '',
            'total_amount': str(float(total)),
            'total_items': str(len(cart_sanitized)),
            'status': 'pending_payment',
            'created_at': order['created_at'],
            'order_json': json.dumps(order, ensure_ascii=False, default=str)
        }
        
        # ✅ MÉTODO UNIVERSAL: Usar hmset (compatible con Redis 2.x, 3.x, 4.x, 5.x)
        redis_client.hmset(order_key, order_data)
        print(f"✅ Hash guardado: {order_key}")
        
        # Índices secundarios
        date_key = datetime.now().strftime('%Y-%m-%d')
        redis_client.sadd(f"orders:date:{date_key}", order_id)
        
        customer_name_key = order['customer']['name'].lower().replace(' ', '_')
        redis_client.sadd(f"orders:customer:{customer_name_key}", order_id)
        
        redis_client.zadd("orders:by_amount", {order_id: total})
        redis_client.lpush("orders:all", order_id)
        
        print(f"✅✅✅ PEDIDO GUARDADO EN REDIS: {order_id}")
        
        # 🔧 FIX: Mensaje WhatsApp con formato correcto
        items_text = "\n".join([
            f"• {item['name']} x{item['quantity']} = S/ {item['subtotal']:.2f}"
            for item in conv_state.cart
        ])
        
        whatsapp_message = f"""Hola, soy {order['customer']['name']}.

📦 *Pedido #{order_id}*

*Productos:*
{items_text}

💰 *Total: S/ {total:.2f}*

📱 Mi teléfono: {order['customer']['phone']}

Quisiera completar mi compra. ¿Cómo procedo con el pago?"""
        
        import urllib.parse
        encoded_message = urllib.parse.quote(whatsapp_message)
        whatsapp_link = f"https://wa.me/{WHATSAPP_NUMBER}?text={encoded_message}"
        
        # Guardar en estado
        conv_state.order_confirmed = True
        conv_state.whatsapp_link = whatsapp_link
        conv_state.order_id = order_id
        conv_state.waiting_confirmation = False
        
        print(f"✅ Link WhatsApp generado")
        print(f"   {whatsapp_link[:80]}...")
        
        return json.dumps({
            "status": "success",
            "message": "PEDIDO CONFIRMADO Y GUARDADO EN REDIS",
            "order_id": order_id,
            "total": f"S/ {total:.2f}",
            "whatsapp_link": whatsapp_link,
            "whatsapp_display": WHATSAPP_DISPLAY
        }, ensure_ascii=False)
        
    except Exception as e:
        print(f"❌ ERROR AL GUARDAR EN REDIS: {e}")
        import traceback
        traceback.print_exc()
        return json.dumps({
            "status": "error",
            "message": f"Error al confirmar pedido: {str(e)}"
        }, ensure_ascii=False)

# ==================== TOOLS DEFINITIONS ====================
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": "Busca productos en el catálogo",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Término de búsqueda"},
                    "top_k": {"type": "integer", "default": 5}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "add_to_cart",
            "description": "Agrega producto al carrito. IMPORTANTE: Usa el 'id' exacto del producto que aparece en los resultados de búsqueda.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "integer", 
                        "description": "ID del producto devuelto por search_products (campo 'id')"
                    },
                    "quantity": {"type": "integer", "default": 1}
                },
                "required": ["product_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "collect_customer_data",
            "description": "Recolecta datos del cliente. Acepta formato: 'nombre, teléfono' o 'nombre teléfono'",
            "parameters": {
                "type": "object",
                "properties": {
                    "raw_input": {
                        "type": "string", 
                        "description": "Datos del cliente (ej: 'juan perez, 999888777')"
                    }
                },
                "required": ["raw_input"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "confirm_order",
            "description": "CONFIRMA y GUARDA el pedido en Redis. Llamar cuando el cliente confirme explícitamente.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]

# ==================== AUDIO ====================
def talker_non_blocking(message: str):
    def _play():
        try:
            response = openai.audio.speech.create(
                model="tts-1",
                voice="nova",
                input=message
            )
            audio = AudioSegment.from_file(BytesIO(response.content), format="mp3")
            
            os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            temp_path = os.path.join(TEMP_AUDIO_DIR, f"audio_{timestamp}.wav")
            
            audio.export(temp_path, format="wav")
            subprocess.call([
                os.path.join(FFMPEG_BIN, "ffplay.exe"),
                "-nodisp", "-autoexit", "-hide_banner", temp_path
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            time.sleep(0.3)
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except:
            pass
    threading.Thread(target=_play, daemon=True).start()

# ==================== SYSTEM MESSAGE ====================
system_message = """Eres un asistente de ventas profesional y eficiente.

FLUJO OBLIGATORIO:
1. Cliente busca → search_products
2. Muestras opciones con el ID EXACTO del campo 'id' en los resultados
   Ejemplo: "Encontré estos productos:
   - ID: 145 - Guantes Pro Gel - $45.00 (10 en stock)
   - ID: 156 - Guantes Classic - $30.00 (5 en stock)"
3. Cliente elige → add_to_cart(product_id=145)  ← USAR EL ID EXACTO
4. Pides: "Por favor dame tu nombre y teléfono (puedes escribir: nombre, teléfono)"
5. Llamas collect_customer_data con el texto exacto del usuario
6. Preguntas: "¿Confirmas tu pedido por S/ X?"
7. Cliente dice "sí/confirmo/ok" → confirm_order

🚨 REGLAS CRÍTICAS:
- SIEMPRE usa el campo 'id' exacto de los resultados de búsqueda
- Muestra los IDs claramente al usuario
- Cuando cliente responde afirmativamente a la confirmación, DEBES llamar confirm_order
- NO inventes datos, usa exactamente lo que el usuario escribió
- Acepta formato flexible: "juan perez, 999888777" o "juan perez 999888777"
- Sé conciso (máximo 3 oraciones)
- Después de confirm_order, menciona que el link aparecerá automáticamente"""

# ==================== HANDLERS ====================
def handle_tool_calls(tool_calls) -> List[Dict]:
    tool_responses = []
    
    for tool_call in tool_calls:
        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        
        print(f"\n🔧 Ejecutando: {function_name}")
        print(f"   Args: {json.dumps(arguments, ensure_ascii=False)}")
        
        if function_name == "search_products":
            result = search_products_tool(
                arguments.get('query'),
                arguments.get('top_k', 5)
            )
        elif function_name == "add_to_cart":
            result = add_to_cart_tool(
                arguments.get('product_id'),
                arguments.get('quantity', 1)
            )
        elif function_name == "collect_customer_data":
            result = collect_customer_data_tool(
                arguments.get('raw_input', '')
            )
        elif function_name == "confirm_order":
            result = confirm_order_tool()
        else:
            result = json.dumps({"error": "Función desconocida"})
        
        tool_responses.append({
            "role": "tool",
            "content": result,
            "tool_call_id": tool_call.id
        })
    
    return tool_responses

# ==================== CHAT ====================
async def optimized_chat(history: List[Dict]) -> Tuple[List[Dict], str]:
    messages = [{"role": "system", "content": system_message}] + history
    
    # 🔧 FIX: Detección mejorada de confirmación
    last_user_msg = history[-1]['content'].lower() if history else ""
    confirmation_words = ['sí', 'si', 'confirmo', 'ok', 'dale', 'yes', 'afirmativo', 'correcto']
    
    if conv_state.waiting_confirmation and any(word in last_user_msg for word in confirmation_words):
        print("\n🎯 DETECTADO: Usuario confirmó pedido, forzando confirm_order...")
        result = confirm_order_tool()
        result_data = json.loads(result)
        
        # Crear respuesta artificial
        if result_data['status'] == 'success':
            reply = f"¡Perfecto! Tu pedido ha sido confirmado exitosamente.\n\n"
            reply += f"📦 **Número de orden:** {conv_state.order_id}\n\n"
            reply += f"🟢 **[HAZ CLICK AQUÍ PARA CONTINUAR POR WHATSAPP]({conv_state.whatsapp_link})**\n\n"
            reply += f"📱 O escribe a: **{WHATSAPP_DISPLAY}**\n\n"
            reply += "Tu pedido está registrado. ¡Gracias por tu compra!"
        else:
            reply = f"❌ Hubo un problema: {result_data.get('message', 'Error desconocido')}"
        
        history.append({"role": "assistant", "content": reply})
        
        cart_info = f"✅ **PEDIDO CONFIRMADO**\n📦 {conv_state.order_id}\n💰 Total: S/ {sum(i['subtotal'] for i in conv_state.cart):.2f}"
        
        return history, cart_info
    
    # Flujo normal con OpenAI
    try:
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: openai.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )
        )
        
        if response.choices[0].finish_reason == "tool_calls":
            message_obj = response.choices[0].message
            tool_responses = handle_tool_calls(message_obj.tool_calls)
            
            messages.append(message_obj)
            messages.extend(tool_responses)
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: openai.chat.completions.create(
                    model=MODEL,
                    messages=messages
                )
            )
        
        reply = response.choices[0].message.content
        
        # 🔧 FIX: Agregar link si fue confirmado
        if conv_state.order_confirmed and conv_state.whatsapp_link:
            reply += f"\n\n📦 **Número de orden:** {conv_state.order_id}\n\n"
            reply += f"🟢 **[HAZ CLICK AQUÍ PARA CONTINUAR POR WHATSAPP]({conv_state.whatsapp_link})**\n\n"
            reply += f"📱 O escribe a: **{WHATSAPP_DISPLAY}**"
        
        history.append({"role": "assistant", "content": reply})
        
        # Audio no bloqueante
        threading.Thread(target=lambda: talker_non_blocking(reply), daemon=True).start()
        
        # Actualizar info del carrito
        cart_info = ""
        if conv_state.cart:
            total = sum(item['subtotal'] for item in conv_state.cart)
            cart_info = f"🛒 {len(conv_state.cart)} items | S/ {total:.2f}"
        
        if conv_state.customer_data.get('name'):
            cart_info += f"\n👤 {conv_state.customer_data['name']}"
            cart_info += f"\n📱 {conv_state.customer_data['phone']}"
        
        if conv_state.order_confirmed:
            cart_info += f"\n\n✅ **CONFIRMADO**\n📦 {conv_state.order_id}"
        
        return history, cart_info
        
    except Exception as e:
        print(f"❌ Error en chat: {e}")
        import traceback
        traceback.print_exc()
        
        error_msg = "Lo siento, hubo un error. ¿Puedes intentar de nuevo?"
        history.append({"role": "assistant", "content": error_msg})
        return history, "❌ Error"

def chat_wrapper(history: List[Dict]) -> Tuple[List[Dict], str]:
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError()
    except:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(optimized_chat(history))

# ==================== UI ====================
with gr.Blocks(title="Agente Ventas IA", theme=gr.themes.Soft()) as ui:
    gr.Markdown("""
    # 🛒 Agente de Ventas IA - VERSIÓN CORREGIDA
    
    ✅ Compatible con Redis 2.x/3.x/4.x/5.x (hmset universal)
    ✅ Normalización automática de datos
    ✅ Dimensiones vectoriales corregidas (389D)
    ✅ Fix serialización JSON (int64 → int)
    
    📱 WhatsApp: +51 907 321 211
    """)
    
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=550, type="messages", label="💬 Chat")
        with gr.Column(scale=1):
            cart_display = gr.Markdown(value="🛒 **Vacío**", label="Estado")
    
    with gr.Row():
        entry = gr.Textbox(
            label="Mensaje:", 
            placeholder="Ej: busco guantes | juan perez, 999888777 | sí confirmo",
            scale=4
        )
        clear = gr.Button("🗑️", scale=1)
    
    entry.submit(
        lambda msg, h: ("", h + [{"role": "user", "content": msg}]),
        inputs=[entry, chatbot],
        outputs=[entry, chatbot]
    ).then(
        chat_wrapper,
        inputs=[chatbot],
        outputs=[chatbot, cart_display]
    )
    
    clear.click(
        lambda: ([], "🛒 **Vacío**"),
        outputs=[chatbot, cart_display]
    ).then(
        lambda: conv_state.clear(),
        outputs=None
    )

# ==================== MAIN ====================
if __name__ == "__main__":
    print("\n" + "="*80)
    print("🚀 AGENTE DE VENTAS IA - VERSIÓN CORREGIDA")
    print("="*80)
    print("\n✅ MEJORAS IMPLEMENTADAS:")
    print("   • Redis universal (hmset compatible con 2.x/3.x/4.x/5.x)")
    print("   • Normalización automática de datos del cliente")
    print("   • Dimensiones vectoriales corregidas (389D)")
    print("   • Fix serialización JSON (numpy int64 → int nativo)")
    print("   • Generación correcta de enlaces WhatsApp")
    print("   • Validación robusta de datos")
    print("\n" + "="*80 + "\n")
    
    search_engine.initialize()
    
    print("\n🌐 Lanzando Gradio...")
    print("📍 URL: http://localhost:7860")
    print("📱 WhatsApp: +51 907 321 211")
    print("\n💡 TIPS:")
    print("   • Puedes escribir: 'valerio quispe, 904875166'")
    print("   • O simplemente: 'valerio quispe 904875166'")
    print("   • Para confirmar: 'sí', 'confirmo', 'ok', 'dale'")
    print("="*80 + "\n")
    
    ui.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True
    )