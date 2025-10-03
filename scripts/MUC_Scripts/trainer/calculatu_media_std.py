import pandas as pd
import numpy as np
from pathlib import Path

def analyze_f1_scores(csv_files, name_column='name', f1_column='f1'):
    """
    Lee 3 archivos CSV y calcula la media y desviación estándar de la columna F1
    para las filas que tienen el mismo nombre en los tres archivos.
    
    Parameters:
    csv_files (list): Lista con las rutas de los 3 archivos CSV
    name_column (str): Nombre de la columna que contiene los nombres/identificadores
    f1_column (str): Nombre de la columna que contiene los valores F1
    
    Returns:
    pandas.DataFrame: DataFrame con nombres, media y desviación estándar
    """
    
    # Verificar que se proporcionen exactamente 3 archivos
    if len(csv_files) != 3:
        raise ValueError("Se requieren exactamente 3 archivos CSV")
    
    # Leer los archivos CSV
    dataframes = []
    for i, file_path in enumerate(csv_files):
        try:
            df = pd.read_csv(file_path)
            # Agregar sufijo al nombre del archivo para identificar origen
            df_renamed = df.rename(columns={f1_column: f'{f1_column}_file{i+1}'})
            dataframes.append(df_renamed[[name_column, f'{f1_column}_file{i+1}']])
            print(f"✓ Archivo {i+1} leído: {file_path} ({len(df)} filas)")
        except Exception as e:
            print(f"✗ Error leyendo {file_path}: {e}")
            return None
    
    # Hacer merge de los tres DataFrames usando el nombre como clave
    merged_df = dataframes[0]
    for i, df in enumerate(dataframes[1:], 2):
        merged_df = merged_df.merge(df, on=name_column, how='inner')
    
    print(f"✓ Filas con nombres coincidentes en los 3 archivos: {len(merged_df)}")
    
    if len(merged_df) == 0:
        print("⚠️ No se encontraron nombres coincidentes en los 3 archivos")
        return pd.DataFrame(columns=[name_column, 'f1_mean', 'f1_std', 'count'])
    
    # Calcular media y desviación estándar
    f1_columns = [f'{f1_column}_file{i}' for i in range(1, 4)]
    
    results = []
    for _, row in merged_df.iterrows():
        name = row[name_column]
        f1_values = [row[col] for col in f1_columns if pd.notna(row[col])]
        
        if len(f1_values) >= 2:  # Al menos 2 valores para calcular std
            mean_f1 = np.mean(f1_values)
            std_f1 = np.std(f1_values)  # Desviación estándar muestral
            count = len(f1_values)
        else:
            mean_f1 = f1_values[0] if f1_values else np.nan
            std_f1 = 0.0 if len(f1_values) == 1 else np.nan
            count = len(f1_values)
        
        results.append({
            name_column: name,
            'f1_mean': mean_f1,
            'f1_std': std_f1,
            'count': count,
            'f1_file1': row[f'{f1_column}_file1'],
            'f1_file2': row[f'{f1_column}_file2'],
            'f1_file3': row[f'{f1_column}_file3']
        })
    
    result_df = pd.DataFrame(results)
    
    # Ordenar por media F1 descendente
    result_df = result_df.sort_values('f1_mean', ascending=False).reset_index(drop=True)
    
    return result_df

def main():
    """Función principal para ejecutar el análisis"""
    
    # CONFIGURACIÓN - Modifica estas rutas según tus archivos
    csv_files = [
        'rejectionSampling/dev/reward_ALL_S9.csv',  # Cambia por la ruta de tu primer archivo
        'rejectionSampling/dev/reward_ALL_S42.csv',  # Cambia por la ruta de tu segundo archivo
        'rejectionSampling/dev/reward_ALL_S16.csv'   # Cambia por la ruta de tu tercer archivo
    ]
    
    # CONFIGURACIÓN - Modifica estos nombres de columnas si son diferentes
    name_column = 'file'  # Nombre de la columna con los identificadores
    f1_column = 'f1'      # Nombre de la columna con los valores F1
    
    print("🔍 Analizando archivos CSV...")
    print("=" * 50)
    
    # Verificar que los archivos existen
    missing_files = []
    for file_path in csv_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("⚠️ Los siguientes archivos no se encontraron:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        print("\n💡 Por favor, actualiza las rutas en la variable 'csv_files'")
        return
    
    # Realizar el análisis
    results = analyze_f1_scores(csv_files, name_column, f1_column)
    
    if results is not None and len(results) > 0:
        print("\n📊 RESULTADOS:")
        print("=" * 50)
        
        # Mostrar estadísticas generales
        print(f"Total de elementos analizados: {len(results)}")
        print(f"F1 Score promedio general: {results['f1_mean'].mean():.4f}")
        print(f"Desviación estándar promedio: {results['f1_std'].mean():.4f}")
        
        print(f"\n📋 Top 10 resultados:")
        print("-" * 80)
        
        # Mostrar los primeros 10 resultados
        display_cols = [name_column, 'f1_mean', 'f1_std', 'f1_file1', 'f1_file2', 'f1_file3']
        print(results[display_cols].head(10).to_string(index=False, float_format='%.4f'))
        
        # Guardar resultados en un archivo CSV
        output_file = 'f1_analysis_results.csv'
        results.to_csv(output_file, index=False, float_format='%.6f')
        print(f"\n💾 Resultados guardados en: {output_file}")
        
        # Mostrar elementos con mayor variabilidad
        high_std = results[results['f1_std'] > results['f1_std'].quantile(0.8)]
        if len(high_std) > 0:
            print(f"\n⚠️ Elementos con mayor variabilidad (top 20% std):")
            print("-" * 50)
            print(high_std[display_cols].head(5).to_string(index=False, float_format='%.4f'))

if __name__ == "__main__":
    main()