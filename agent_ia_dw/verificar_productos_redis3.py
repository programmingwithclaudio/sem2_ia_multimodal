"""
Script de prueba para verificar datos en Redis
"""

import redis
import json

# Conectar a Redis
r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

print("="*80)
print("🔍 VERIFICACIÓN DE REDIS")
print("="*80)

# 1. Verificar conexión
try:
    r.ping()
    print("✅ Redis conectado")
except Exception as e:
    print(f"❌ Redis NO conectado: {e}")
    exit(1)

# 2. Buscar productos
print("\n📦 PRODUCTOS EN REDIS:")
print("-"*80)

# Intentar obtener los primeros 10 productos
for idx in range(10):
    key = f"producto:{idx}"
    data = r.hgetall(key)
    
    if data:
        print(f"\n✓ Producto {idx}:")
        print(f"   ID: {data.get('id')}")
        print(f"   Nombre: {data.get('name')}")
        print(f"   Categoría: {data.get('category')}")
        print(f"   Precio: ${data.get('list_price')}")
        print(f"   Stock: {data.get('stock_quantity')} unidades")
        
        # Verificar texto procesado
        if 'texto_procesado' in data:
            print(f"   Texto: {data.get('texto_procesado')[:50]}...")
    else:
        print(f"⚠️  Producto {idx} no existe")

# 3. Buscar todos los productos
print("\n" + "="*80)
print("🔢 ESTADÍSTICAS:")
print("-"*80)

all_keys = r.keys("producto:*")
print(f"Total de productos indexados: {len(all_keys)}")

if all_keys:
    print(f"\nRango de IDs: producto:0 hasta producto:{len(all_keys)-1}")
    
    # Mostrar algunos IDs de muestra
    sample_ids = sorted([int(k.split(':')[1]) for k in all_keys])[:10]
    print(f"Primeros 10 IDs: {sample_ids}")

# 4. Verificar órdenes
print("\n" + "="*80)
print("🛒 ÓRDENES EN REDIS:")
print("-"*80)

order_keys = r.keys("order:*")
if order_keys:
    print(f"Total de órdenes: {len(order_keys)}")
    
    # Mostrar última orden
    latest_order_key = order_keys[-1] if order_keys else None
    if latest_order_key:
        order_data = r.hgetall(latest_order_key)
        print(f"\n📋 Última orden: {latest_order_key}")
        print(f"   Cliente: {order_data.get('customer_name')}")
        print(f"   Teléfono: {order_data.get('customer_phone')}")
        print(f"   Total: S/ {order_data.get('total_amount')}")
        print(f"   Estado: {order_data.get('status')}")
        
        # Mostrar JSON completo
        if 'order_json' in order_data:
            order_json = json.loads(order_data['order_json'])
            print(f"\n   Productos:")
            for item in order_json.get('items', []):
                print(f"      • {item['name']} x{item['quantity']} = S/ {item['subtotal']:.2f}")
else:
    print("⚠️  No hay órdenes registradas")

print("\n" + "="*80)
print("✅ VERIFICACIÓN COMPLETADA")
print("="*80)

# 5. Comandos útiles
print("\n💡 COMANDOS ÚTILES:")
print("-"*80)
print("Ver un producto específico:")
print("  redis-cli HGETALL producto:0")
print("\nVer todas las órdenes:")
print("  redis-cli KEYS 'order:*'")
print("\nVer una orden específica:")
print("  redis-cli HGETALL order:ORD-XXXXXXXXX")
print("\nLimpiar Redis (⚠️  CUIDADO):")
print("  redis-cli FLUSHDB")
print("="*80)