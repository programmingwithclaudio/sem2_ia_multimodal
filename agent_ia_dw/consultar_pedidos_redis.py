"""
Script para consultar y administrar pedidos en Redis
Uso: python consultar_pedidos_redis.py
"""

import redis
import json
from datetime import datetime
from typing import List, Dict

REDIS_CONFIG = {
    'host': 'localhost',
    'port': 6379,
    'db': 0,
    'decode_responses': True
}

def conectar_redis():
    try:
        client = redis.Redis(**REDIS_CONFIG)
        client.ping()
        print("✅ Conectado a Redis\n")
        return client
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def listar_todos_pedidos(r) -> List[str]:
    try:
        order_keys = r.keys("order:ORD-*")
        order_ids = [key.replace("order:", "") for key in order_keys]
        return sorted(order_ids, reverse=True)
    except Exception as e:
        print(f"❌ Error: {e}")
        return []

def obtener_pedido(r, order_id: str) -> Dict:
    try:
        order_key = f"order:{order_id}"
        order_data = r.hgetall(order_key)
        
        if not order_data:
            return None
        
        if 'order_json' in order_data:
            return json.loads(order_data['order_json'])
        
        return order_data
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def mostrar_pedido(order: Dict):
    print("\n" + "="*70)
    print(f"📦 PEDIDO: {order['order_id']}")
    print("="*70)
    
    customer = order.get('customer', {})
    print(f"\n👤 CLIENTE:")
    print(f"   Nombre: {customer.get('name', 'N/A')}")
    print(f"   Teléfono: {customer.get('phone', 'N/A')}")
    if customer.get('email'):
        print(f"   Email: {customer['email']}")
    
    items = order.get('items', [])
    print(f"\n🛒 PRODUCTOS ({len(items)} items):")
    for i, item in enumerate(items, 1):
        print(f"   {i}. {item['name']}")
        print(f"      {item.get('color', 'N/A')} | {item.get('size', 'N/A')}")
        print(f"      ${item['price']:.2f} x {item['quantity']} = ${item['subtotal']:.2f}")
    
    print(f"\n💰 TOTAL: S/ {order.get('total_amount', 0):.2f}")
    print(f"📊 Estado: {order.get('status', 'N/A').upper()}")
    print(f"📅 Creado: {order.get('created_at', 'N/A')[:19]}")
    print("="*70 + "\n")

def estadisticas(r):
    try:
        order_keys = r.keys("order:ORD-*")
        total = len(order_keys)
        
        ventas = 0
        hoy = 0
        fecha_hoy = datetime.now().strftime('%Y-%m-%d')
        
        for key in order_keys:
            data = r.hgetall(key)
            ventas += float(data.get('total_amount', 0))
            if data.get('created_at', '').startswith(fecha_hoy):
                hoy += 1
        
        print("\n" + "="*70)
        print("📊 ESTADÍSTICAS")
        print("="*70)
        print(f"📦 Total pedidos: {total}")
        print(f"📅 Pedidos hoy: {hoy}")
        print(f"💰 Ventas totales: S/ {ventas:.2f}")
        if total > 0:
            print(f"📊 Ticket promedio: S/ {ventas/total:.2f}")
        print("="*70 + "\n")
    except Exception as e:
        print(f"❌ Error: {e}")

def menu_principal():
    r = conectar_redis()
    if not r:
        return
    
    while True:
        print("\n" + "="*70)
        print("🛒 ADMINISTRADOR DE PEDIDOS")
        print("="*70)
        print("\n1. Ver todos los pedidos")
        print("2. Ver pedido específico")
        print("3. Ver estadísticas")
        print("4. Salir")
        
        opcion = input("\nOpción (1-4): ").strip()
        
        if opcion == "1":
            print("\n📋 TODOS LOS PEDIDOS:")
            order_ids = listar_todos_pedidos(r)
            if not order_ids:
                print("   No hay pedidos")
            else:
                print(f"   Total: {len(order_ids)}\n")
                for i, oid in enumerate(order_ids, 1):
                    order = obtener_pedido(r, oid)
                    if order:
                        customer = order.get('customer', {})
                        print(f"   {i}. {oid}")
                        print(f"      {customer.get('name', 'N/A')} | S/ {order.get('total_amount', 0):.2f}")
                        print(f"      {order.get('created_at', 'N/A')[:19]}\n")
        
        elif opcion == "2":
            oid = input("\nID del pedido (ej: ORD-20250101120000): ").strip()
            if not oid.startswith("ORD-"):
                oid = f"ORD-{oid}"
            pedido = obtener_pedido(r, oid)
            if pedido:
                mostrar_pedido(pedido)
            else:
                print(f"❌ Pedido {oid} no encontrado")
        
        elif opcion == "3":
            estadisticas(r)
        
        elif opcion == "4":
            print("\n👋 ¡Hasta luego!")
            break
        
        else:
            print("\n❌ Opción inválida")
        
        input("\nPresiona Enter para continuar...")

if __name__ == "__main__":
    print("""
╔════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║           🛒 ADMINISTRADOR DE PEDIDOS - REDIS                      ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
    """)
    
    try:
        menu_principal()
    except KeyboardInterrupt:
        print("\n\n👋 Cerrado")
    except Exception as e:
        print(f"\n❌ Error: {e}")