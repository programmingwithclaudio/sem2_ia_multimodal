import pandas as pd
import pyodbc
from datetime import datetime
import os

# Configuración
origen_config = {
    'server': r"DESKTOP-3NPIQBD",
    'database': "AdventureWorks",
    'username': "sa",
    'password': "meilitodev123"
}

output_dir = "datasets"
os.makedirs(output_dir, exist_ok=True)

def crear_conexion(config):
    """Crea conexión a SQL Server"""
    conn_str = (
        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
        f"SERVER={config['server']};"
        f"DATABASE={config['database']};"
        f"UID={config['username']};"
        f"PWD={config['password']}"
    )
    return pyodbc.connect(conn_str)

# ====== QUERY DEFINITIVA PARA AGENTE DE VENTAS ======
def extraer_catalogo_productos_ventas():
    """
    Query optimizada con TODO lo necesario para un agente de ventas:
    - Stock actual valorizado
    - Historial de descuentos aplicados
    - Métricas de ventas
    - Información completa del producto
    """
    
    print(f"\n[{datetime.now()}] Extrayendo catálogo completo para agente de ventas...")
    
    query_agente_ventas = """
    WITH VentasHistoricas AS (
        -- Histórico de ventas por producto
        SELECT 
            sod.ProductID,
            COUNT(DISTINCT sod.SalesOrderID) AS Total_Ordenes,
            SUM(sod.OrderQty) AS Cantidad_Total_Vendida,
            AVG(sod.UnitPrice) AS Precio_Promedio_Venta,
            MIN(sod.UnitPrice) AS Precio_Minimo_Venta,
            MAX(sod.UnitPrice) AS Precio_Maximo_Venta,
            AVG(sod.UnitPriceDiscount) AS Descuento_Promedio_Aplicado,
            MAX(sod.UnitPriceDiscount) AS Descuento_Maximo_Aplicado,
            SUM(sod.LineTotal) AS Ingresos_Totales_Historicos,
            MAX(soh.OrderDate) AS Ultima_Venta,
            MIN(soh.OrderDate) AS Primera_Venta
        FROM Sales.SalesOrderDetail sod
        INNER JOIN Sales.SalesOrderHeader soh ON sod.SalesOrderID = soh.SalesOrderID
        WHERE soh.OrderDate >= '2013-01-01'
        GROUP BY sod.ProductID
    ),
    InventarioActual AS (
        -- Stock actual por producto y ubicación
        SELECT 
            ProductID,
            SUM(Quantity) AS Stock_Total,
            COUNT(DISTINCT LocationID) AS Ubicaciones_Almacen
        FROM Production.ProductInventory
        GROUP BY ProductID
    )
    
    SELECT 
        -- ===== IDENTIFICACIÓN DEL PRODUCTO =====
        p.ProductID AS 'ID_Producto',
        p.ProductNumber AS 'Codigo_Producto',
        p.Name AS 'Nombre_Producto',
        pc.Name AS 'Categoria',
        psc.Name AS 'Subcategoria',
        p.Color AS 'Color',
        p.Size AS 'Talla',
        p.Weight AS 'Peso_Libras',
        um.Name AS 'Unidad_Medida',
        p.ProductLine AS 'Linea_Producto',
        p.Class AS 'Clase_Producto',
        p.Style AS 'Estilo',
        
        -- ===== PRECIOS Y COSTOS (CRÍTICO PARA DESCUENTOS) =====
        p.StandardCost AS 'Costo_Estandar',
        p.ListPrice AS 'Precio_Lista',
        ROUND((p.ListPrice - p.StandardCost), 2) AS 'Margen_Bruto_$',
        ROUND(((p.ListPrice - p.StandardCost) / NULLIF(p.ListPrice, 0) * 100), 2) AS 'Margen_Bruto_%',
        
        -- ===== INVENTARIO VALORIZADO =====
        ISNULL(inv.Stock_Total, 0) AS 'Stock_Actual',
        ROUND(ISNULL(inv.Stock_Total, 0) * p.StandardCost, 2) AS 'Valor_Inventario_Costo',
        ROUND(ISNULL(inv.Stock_Total, 0) * p.ListPrice, 2) AS 'Valor_Inventario_Precio_Lista',
        ISNULL(inv.Ubicaciones_Almacen, 0) AS 'Ubicaciones_Disponibles',
        
        -- ===== ESTADO DEL PRODUCTO =====
        CASE 
            WHEN p.DiscontinuedDate IS NOT NULL THEN 'Descontinuado'
            WHEN p.SellStartDate > GETDATE() THEN 'Proximo_Lanzamiento'
            WHEN p.SellEndDate < GETDATE() THEN 'Fuera_Catalogo'
            WHEN ISNULL(inv.Stock_Total, 0) = 0 THEN 'Sin_Stock'
            WHEN ISNULL(inv.Stock_Total, 0) <= 50 THEN 'Stock_Bajo'
            WHEN ISNULL(inv.Stock_Total, 0) <= 200 THEN 'Stock_Medio'
            ELSE 'Stock_Alto'
        END AS 'Estado_Inventario',
        
        CASE 
            WHEN p.MakeFlag = 1 THEN 'Fabricado_Interno'
            ELSE 'Comprado_Externo'
        END AS 'Tipo_Adquisicion',
        
        p.DaysToManufacture AS 'Dias_Fabricacion',
        p.ReorderPoint AS 'Punto_Reorden',
        p.SafetyStockLevel AS 'Stock_Seguridad',
        
        -- ===== HISTÓRICO DE VENTAS =====
        ISNULL(vh.Total_Ordenes, 0) AS 'Total_Ordenes_Historicas',
        ISNULL(vh.Cantidad_Total_Vendida, 0) AS 'Unidades_Vendidas_Historicas',
        ROUND(ISNULL(vh.Precio_Promedio_Venta, 0), 2) AS 'Precio_Promedio_Vendido',
        ROUND(ISNULL(vh.Precio_Minimo_Venta, 0), 2) AS 'Precio_Minimo_Vendido',
        ROUND(ISNULL(vh.Precio_Maximo_Venta, 0), 2) AS 'Precio_Maximo_Vendido',
        ROUND(ISNULL(vh.Ingresos_Totales_Historicos, 0), 2) AS 'Ingresos_Totales_Historicos',
        
        -- ===== ANÁLISIS DE DESCUENTOS (CRÍTICO PARA IA) =====
        ROUND(ISNULL(vh.Descuento_Promedio_Aplicado, 0) * 100, 2) AS 'Descuento_Promedio_Historico_%',
        ROUND(ISNULL(vh.Descuento_Maximo_Aplicado, 0) * 100, 2) AS 'Descuento_Maximo_Historico_%',
        
        -- Rango de descuento sugerido (basado en margen)
        ROUND(((p.ListPrice - p.StandardCost) / NULLIF(p.ListPrice, 0) * 100) * 0.3, 2) AS 'Descuento_Minimo_Sugerido_%',
        ROUND(((p.ListPrice - p.StandardCost) / NULLIF(p.ListPrice, 0) * 100) * 0.7, 2) AS 'Descuento_Maximo_Sugerido_%',
        
        -- ===== MÉTRICAS DE ROTACIÓN =====
        CASE 
            WHEN ISNULL(inv.Stock_Total, 0) = 0 THEN 0
            ELSE ROUND(ISNULL(vh.Cantidad_Total_Vendida, 0) / NULLIF(inv.Stock_Total, 1), 2)
        END AS 'Indice_Rotacion',
        
        DATEDIFF(day, vh.Ultima_Venta, GETDATE()) AS 'Dias_Desde_Ultima_Venta',
        
        -- ===== FECHAS DE CONTROL =====
        CONVERT(date, p.SellStartDate) AS 'Fecha_Inicio_Venta',
        CONVERT(date, p.SellEndDate) AS 'Fecha_Fin_Venta',
        CONVERT(date, p.DiscontinuedDate) AS 'Fecha_Descontinuado',
        CONVERT(date, vh.Primera_Venta) AS 'Primera_Venta_Real',
        CONVERT(date, vh.Ultima_Venta) AS 'Ultima_Venta_Real',
        
        -- ===== CLASIFICACIÓN PARA IA =====
        CASE 
            WHEN ISNULL(vh.Total_Ordenes, 0) >= 100 AND ISNULL(inv.Stock_Total, 0) >= 100 THEN 'Estrella'
            WHEN ISNULL(vh.Total_Ordenes, 0) >= 100 AND ISNULL(inv.Stock_Total, 0) < 100 THEN 'Alta_Demanda_Bajo_Stock'
            WHEN ISNULL(vh.Total_Ordenes, 0) < 50 AND ISNULL(inv.Stock_Total, 0) >= 200 THEN 'Sobrestockeado'
            WHEN ISNULL(vh.Total_Ordenes, 0) = 0 THEN 'Sin_Ventas'
            ELSE 'Normal'
        END AS 'Clasificacion_Ventas',
        
        CASE 
            WHEN ((p.ListPrice - p.StandardCost) / NULLIF(p.ListPrice, 0)) >= 0.50 THEN 'Margen_Alto'
            WHEN ((p.ListPrice - p.StandardCost) / NULLIF(p.ListPrice, 0)) >= 0.30 THEN 'Margen_Medio'
            ELSE 'Margen_Bajo'
        END AS 'Clasificacion_Margen'
        
    FROM Production.Product p
    LEFT JOIN Production.ProductSubcategory psc ON p.ProductSubcategoryID = psc.ProductSubcategoryID
    LEFT JOIN Production.ProductCategory pc ON psc.ProductCategoryID = pc.ProductCategoryID
    LEFT JOIN Production.UnitMeasure um ON p.SizeUnitMeasureCode = um.UnitMeasureCode
    LEFT JOIN VentasHistoricas vh ON p.ProductID = vh.ProductID
    LEFT JOIN InventarioActual inv ON p.ProductID = inv.ProductID
    
    WHERE p.SellEndDate IS NULL OR p.SellEndDate >= DATEADD(year, -2, GETDATE())
    
    ORDER BY 
        ISNULL(vh.Ingresos_Totales_Historicos, 0) DESC,
        ISNULL(inv.Stock_Total, 0) DESC
    """
    
    conn = crear_conexion(origen_config)
    df_catalogo = pd.read_sql(query_agente_ventas, conn)
    conn.close()
    
    # ===== POST-PROCESAMIENTO: CALCULAR MÉTRICAS ADICIONALES =====
    
    # Precio con descuento sugerido
    df_catalogo['Precio_Con_Descuento_Minimo'] = (
        df_catalogo['Precio_Lista'] * (1 - df_catalogo['Descuento_Minimo_Sugerido_%'] / 100)
    ).round(2)
    
    df_catalogo['Precio_Con_Descuento_Maximo'] = (
        df_catalogo['Precio_Lista'] * (1 - df_catalogo['Descuento_Maximo_Sugerido_%'] / 100)
    ).round(2)
    
    # Rentabilidad por unidad
    df_catalogo['Rentabilidad_Unitaria_Min'] = (
        df_catalogo['Precio_Con_Descuento_Minimo'] - df_catalogo['Costo_Estandar']
    ).round(2)
    
    df_catalogo['Rentabilidad_Unitaria_Max'] = (
        df_catalogo['Precio_Con_Descuento_Maximo'] - df_catalogo['Costo_Estandar']
    ).round(2)
    
    # Potencial de ingresos con stock actual
    df_catalogo['Potencial_Ingresos_Stock_Actual'] = (
        df_catalogo['Stock_Actual'] * df_catalogo['Precio_Lista']
    ).round(2)
    
    # Prioridad de venta (score para IA)
    df_catalogo['Score_Prioridad_Venta'] = (
        (df_catalogo['Margen_Bruto_%'] * 0.3) +  # 30% peso al margen
        (df_catalogo['Stock_Actual'] / 100 * 0.2) +  # 20% peso al stock
        (df_catalogo['Indice_Rotacion'] * 10 * 0.3) +  # 30% peso a rotación
        ((100 - df_catalogo['Dias_Desde_Ultima_Venta'].fillna(365)) / 100 * 0.2)  # 20% peso a frescura
    ).round(2)
    
    return df_catalogo

# ====== EXPORTAR EN CHUNKS SI ES NECESARIO ======
def guardar_con_chunks(df, nombre_base, chunk_size=10000):
    """
    Guarda DataFrame en múltiples archivos si excede chunk_size
    """
    total_registros = len(df)
    
    if total_registros <= chunk_size:
        # Un solo archivo
        output_file = os.path.join(output_dir, f"{nombre_base}.csv")
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"   ✓ Archivo único: {nombre_base}.csv ({total_registros:,} registros)")
        return [output_file]
    else:
        # Múltiples chunks
        num_chunks = (total_registros // chunk_size) + 1
        archivos = []
        
        for i in range(num_chunks):
            inicio = i * chunk_size
            fin = min((i + 1) * chunk_size, total_registros)
            
            df_chunk = df.iloc[inicio:fin]
            output_file = os.path.join(output_dir, f"{nombre_base}_chunk_{i+1:02d}.csv")
            df_chunk.to_csv(output_file, index=False, encoding='utf-8-sig')
            archivos.append(output_file)
            
            print(f"   ✓ Chunk {i+1}/{num_chunks}: {nombre_base}_chunk_{i+1:02d}.csv ({len(df_chunk):,} registros)")
        
        return archivos

# ====== EJECUCIÓN PRINCIPAL ======
if __name__ == "__main__":
    print(f"{'='*70}")
    print(f"EXTRACCIÓN OPTIMIZADA PARA AGENTE DE VENTAS CON IA")
    print(f"AdventureWorks - Catálogo con Stock Valorizado y Descuentos")
    print(f"{'='*70}")
    
    try:
        inicio = datetime.now()
        
        # Extraer catálogo completo
        df_catalogo = extraer_catalogo_productos_ventas()
        
        # Guardar (con chunks si es necesario)
        archivos = guardar_con_chunks(df_catalogo, "catalogo_agente_ventas_ia", chunk_size=10000)
        
        # ===== ESTADÍSTICAS =====
        print(f"\n{'='*70}")
        print(f"ESTADÍSTICAS DEL CATÁLOGO")
        print(f"{'='*70}")
        print(f"Total productos: {len(df_catalogo):,}")
        print(f"Con stock disponible: {(df_catalogo['Stock_Actual'] > 0).sum():,}")
        print(f"Valor total inventario (costo): ${df_catalogo['Valor_Inventario_Costo'].sum():,.2f}")
        print(f"Valor total inventario (precio lista): ${df_catalogo['Valor_Inventario_Precio_Lista'].sum():,.2f}")
        print(f"Margen promedio: {df_catalogo['Margen_Bruto_%'].mean():.2f}%")
        print(f"Descuento promedio histórico: {df_catalogo['Descuento_Promedio_Historico_%'].mean():.2f}%")
        
        print(f"\nCLASIFICACIÓN DE PRODUCTOS:")
        print(df_catalogo['Clasificacion_Ventas'].value_counts().to_string())
        
        print(f"\nCLASIFICACIÓN POR MARGEN:")
        print(df_catalogo['Clasificacion_Margen'].value_counts().to_string())
        
        print(f"\n{'='*70}")
        print(f"Archivos generados: {len(archivos)}")
        print(f"Directorio: {output_dir}")
        print(f"Tiempo ejecución: {(datetime.now() - inicio).total_seconds():.2f} segundos")
        print(f"{'='*70}")
        print(f"[{datetime.now()}] ¡Catálogo listo para agente de IA!")
        
        # Mostrar muestra de datos
        print(f"\n📊 MUESTRA DE PRODUCTOS TOP (primeros 5):")
        print(df_catalogo[['Nombre_Producto', 'Categoria', 'Precio_Lista', 'Stock_Actual', 
                          'Margen_Bruto_%', 'Descuento_Promedio_Historico_%', 
                          'Score_Prioridad_Venta']].head().to_string(index=False))
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        raise