import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import os
import glob

# Configuração da página
st.set_page_config(
    page_title="Análise de Anomalias - Pageviews",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .anomaly-critical {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
    }
    .anomaly-warning {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
    }
    .anomaly-info {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
</style>
""", unsafe_allow_html=True)

def load_and_prepare_data(uploaded_file):
    """Carrega e prepara os dados do CSV"""
    try:
        # Ler o CSV
        df = pd.read_csv(uploaded_file)
        
        # Verificar se as colunas necessárias existem
        if 'Date + hour (YYYYMMDDHH)' in df.columns and 'Views' in df.columns:
            # Formato novo: "Date + hour (YYYYMMDDHH)", "Views"
            date_hour_col = 'Date + hour (YYYYMMDDHH)'
            
            # Extrair data e hora da string
            df['Date'] = pd.to_datetime(df[date_hour_col].str.extract(r'(\w+ \d+, \d+)')[0], format='%b %d, %Y')
            
            # Extrair hora da string (formato "12AM", "1PM", etc.)
            hour_str = df[date_hour_col].str.extract(r'(\d+)(AM|PM)')
            hour_num = hour_str[0].astype(int)
            am_pm = hour_str[1]
            
            # Converter para formato 24h
            df['Hour'] = hour_num.where(am_pm == 'AM', hour_num + 12)
            df['Hour'] = df['Hour'].where(df['Hour'] != 12, 0)  # 12AM = 0h
            df['Hour'] = df['Hour'].where(df['Hour'] != 24, 12)  # 12PM = 12h
            
        elif 'Date' in df.columns and 'Hour' in df.columns and 'Views' in df.columns:
            # Formato antigo: "Date", "Hour", "Views"
            df['Date'] = pd.to_datetime(df['Date'], format='%b %d, %Y', errors='coerce')
            
        else:
            st.error("❌ O CSV deve conter as colunas: 'Date + hour (YYYYMMDDHH)' e 'Views' OU 'Date', 'Hour' e 'Views'")
            return None
        
        # Verificar se a conversão foi bem-sucedida
        if df['Date'].isna().any():
            st.error("❌ Erro ao converter datas. Verifique o formato das datas no CSV.")
            return None
        
        # Criar coluna de data e hora combinada
        df['DateTime'] = df['Date'] + pd.to_timedelta(df['Hour'], unit='h')
        
        # Ordenar por data e hora
        df = df.sort_values(['Date', 'Hour']).reset_index(drop=True)
        
        return df
        
    except Exception as e:
        st.error(f"❌ Erro ao processar o arquivo: {str(e)}")
        return None

def calculate_statistics(df):
    """Calcula estatísticas básicas dos dados"""
    views = df['Views'].values
    
    # Verificar se há dados
    if len(views) == 0:
        return {
            'total': 0,
            'count': 0,
            'mean': 0.0,
            'median': 0.0,
            'std': 0.0,
            'min': 0,
            'max': 0,
            'q1': 0.0,
            'q3': 0.0,
            'iqr': 0.0,
            'lower_bound': 0.0,
            'upper_bound': 0.0
        }
    
    stats = {
        'total': int(np.sum(views)),
        'count': len(views),
        'mean': float(np.mean(views)),
        'median': float(np.median(views)),
        'std': float(np.std(views)),
        'min': int(np.min(views)),
        'max': int(np.max(views))
    }
    
    # Calcular quartis
    q1 = np.percentile(views, 25)
    q3 = np.percentile(views, 75)
    iqr = q3 - q1
    
    stats['q1'] = float(q1)
    stats['q3'] = float(q3)
    stats['iqr'] = float(iqr)
    stats['lower_bound'] = float(q1 - 1.5 * iqr)
    stats['upper_bound'] = float(q3 + 1.5 * iqr)
    
    return stats

def calculate_hourly_means(df):
    """Calcula a média histórica para cada hora do dia"""
    hourly_means = df.groupby('Hour')['Views'].mean().to_dict()
    return hourly_means

def detect_low_values_by_hour(df, hourly_means, threshold_percentage=0.5):
    """
    Detecta valores baixos comparando com a média histórica de cada hora
    threshold_percentage: percentual da média histórica (ex: 0.5 = 50% da média)
    """
    low_values = []
    
    for _, row in df.iterrows():
        hour = row['Hour']
        current_value = row['Views']
        historical_mean = hourly_means.get(hour, 0)
        
        # Se a média histórica for muito baixa (menos que 5), não considerar como anomalia
        if historical_mean < 5:
            continue
            
        # Se o valor atual for menor que X% da média histórica dessa hora
        if current_value < (historical_mean * threshold_percentage):
            low_values.append(row)
    
    return pd.DataFrame(low_values) if low_values else pd.DataFrame()

def detect_anomalies(df, stats, threshold_percentage=0.5, extreme_threshold=0.8, start_hour=0, end_hour=23):
    """Detecta diferentes tipos de anomalias"""
    # Verificar se há dados
    if len(df) == 0:
        return {
            'zero_views': pd.DataFrame(),
            'very_low_views': pd.DataFrame(),
            'statistical_outliers': pd.DataFrame(),
            'extreme_variations': pd.DataFrame(),
            'missing_hours': []
        }
    
    # Filtrar dados pelo range de horas
    df_filtered = df[(df['Hour'] >= start_hour) & (df['Hour'] <= end_hour)]
    
    # Calcular médias históricas por hora
    hourly_means = calculate_hourly_means(df)
    
    # Detectar valores baixos comparando com a média histórica de cada hora
    very_low_views = detect_low_values_by_hour(df_filtered, hourly_means, threshold_percentage)
    
    anomalies = {
        'zero_views': df_filtered[df_filtered['Views'] == 0],
        'very_low_views': very_low_views,
        'statistical_outliers': df_filtered[
            (df_filtered['Views'] < stats['lower_bound']) | 
            (df_filtered['Views'] > stats['upper_bound'])
        ],
        'extreme_variations': df_filtered[
            (abs(df_filtered['Views'] - stats['mean']) / stats['mean'] > extreme_threshold) &
            ((df_filtered['Views'] > stats['mean'] * 2) | (df_filtered['Views'] < stats['mean'] * 0.3))
        ]
    }
    
    # Verificar horas faltando (apenas no range selecionado)
    expected_hours = set(range(start_hour, end_hour + 1))  # Range selecionado
    daily_hours = df.groupby('Date')['Hour'].apply(set).to_dict()
    missing_hours = []
    
    for date, hours in daily_hours.items():
        missing = expected_hours - hours
        if missing:
            missing_hours.append({
                'date': date,
                'missing_hours': sorted(list(missing))
            })
    
    anomalies['missing_hours'] = missing_hours
    
    return anomalies

def create_time_series_chart(df, anomalies, stats):
    """Cria gráfico de série temporal com anomalias destacadas e horas faltando preenchidas"""
    fig = go.Figure()
    
    # Criar série temporal completa com todas as horas
    if len(df) > 0:
        # Obter range de datas
        min_date = df['Date'].min()
        max_date = df['Date'].max()
        
        # Criar DataFrame completo com todas as horas
        complete_dates = pd.date_range(start=min_date, end=max_date, freq='D')
        complete_data = []
        
        for date in complete_dates:
            for hour in range(24):
                datetime_combo = pd.to_datetime(f"{date.strftime('%Y-%m-%d')} {hour:02d}:00:00")
                
                # Verificar se existe dados para esta data/hora
                existing_data = df[(df['Date'] == date) & (df['Hour'] == hour)]
                
                if not existing_data.empty:
                    # Usar dados existentes
                    views = existing_data['Views'].iloc[0]
                    is_missing = False
                else:
                    # Preencher com 0 para horas faltando
                    views = 0
                    is_missing = True
                
                complete_data.append({
                    'DateTime': datetime_combo,
                    'Views': views,
                    'IsMissing': is_missing
                })
        
        complete_df = pd.DataFrame(complete_data)
        
        # Linha principal (dados existentes)
        existing_data = complete_df[~complete_df['IsMissing']]
        if not existing_data.empty:
            fig.add_trace(go.Scatter(
                x=existing_data['DateTime'],
                y=existing_data['Views'],
                mode='lines+markers',
                name='Pageviews',
                line=dict(color='steelblue', width=1),
                marker=dict(size=3),
                opacity=0.7,
                hovertemplate='<b>%{x|%d/%m/%Y %H}h</b><br>Views: %{y}<extra></extra>'
            ))
        
        # Pontos para horas faltando (com 0)
        missing_data = complete_df[complete_df['IsMissing']]
        if not missing_data.empty:
            fig.add_trace(go.Scatter(
                x=missing_data['DateTime'],
                y=missing_data['Views'],
                mode='markers',
                name='Horas Faltando*',
                marker=dict(color='lightgray', size=4, symbol='circle'),
                hovertemplate='<b>%{x|%d/%m/%Y %H}h</b><br>Views: %{y} (Faltando)<extra></extra>'
            ))
    
    # Destacar anomalias (evitando sobreposição)
    if not anomalies['zero_views'].empty:
        fig.add_trace(go.Scatter(
            x=anomalies['zero_views']['DateTime'],
            y=anomalies['zero_views']['Views'],
            mode='markers',
            name='Valores Zero',
            marker=dict(color='red', size=8, symbol='x'),
            hovertemplate='<b>%{x|%d/%m/%Y %H}h</b><br>Views: %{y} (Zero)<extra></extra>'
        ))
    
    # Valores muito baixos (excluindo os que já são zero)
    if not anomalies['very_low_views'].empty:
        # Filtrar para excluir valores zero (que já são mostrados acima)
        very_low_non_zero = anomalies['very_low_views'][anomalies['very_low_views']['Views'] > 0]
        if not very_low_non_zero.empty:
            fig.add_trace(go.Scatter(
                x=very_low_non_zero['DateTime'],
                y=very_low_non_zero['Views'],
                mode='markers',
                name='Valores Muito Baixos',
                marker=dict(color='orange', size=6),
                hovertemplate='<b>%{x|%d/%m/%Y %H}h</b><br>Views: %{y} (Muito Baixo)<extra></extra>'
            ))
    
    # Outliers estatísticos (excluindo os que já são zero ou muito baixos)
    if not anomalies['statistical_outliers'].empty:
        # Filtrar para excluir valores que já são mostrados em outras categorias
        outliers_filtered = anomalies['statistical_outliers'][
            (anomalies['statistical_outliers']['Views'] > 0) &
            (~anomalies['statistical_outliers']['DateTime'].isin(anomalies['very_low_views']['DateTime']))
        ]
        if not outliers_filtered.empty:
            fig.add_trace(go.Scatter(
                x=outliers_filtered['DateTime'],
                y=outliers_filtered['Views'],
                mode='markers',
                name='Outliers Estatísticos',
                marker=dict(color='purple', size=6),
                hovertemplate='<b>%{x|%d/%m/%Y %H}h</b><br>Views: %{y} (Outlier)<extra></extra>'
            ))
    
    # Linha da média
    fig.add_hline(
        y=stats['mean'],
        line_dash="dash",
        line_color="green",
        annotation_text=f"Média: {stats['mean']:.0f}",
        annotation_position="top right"
    )
    
    fig.update_layout(
        title='Série Temporal de Pageviews com Anomalias Destacadas',
        xaxis_title='Data e Hora',
        yaxis_title='Pageviews',
        hovermode='x unified',
        height=500
    )
    
    return fig


def create_hourly_analysis_chart(df):
    """Cria análise por hora do dia"""
    hourly_stats = df.groupby('Hour')['Views'].agg(['mean', 'std', 'count', 'min', 'max']).reset_index()
    
    fig = go.Figure()
    
    # Gráfico de barras com barras de erro
    fig.add_trace(go.Bar(
        x=hourly_stats['Hour'],
        y=hourly_stats['mean'],
        name='Média por Hora',
        marker_color='steelblue',
        error_y=dict(
            type='data', 
            array=hourly_stats['std'], 
            visible=True,
            color='rgba(0,0,0,0.3)'
        ),
        hovertemplate='<b>Hora:</b> %{x}h<br><b>Média:</b> %{y:.0f}<br><b>Desvio:</b> ±%{error_y.array:.0f}<extra></extra>'
    ))
    
    # Adicionar linha de tendência
    fig.add_trace(go.Scatter(
        x=hourly_stats['Hour'],
        y=hourly_stats['mean'],
        mode='lines+markers',
        name='Tendência',
        line=dict(color='red', width=2),
        marker=dict(size=6),
        hovertemplate='<b>Hora:</b> %{x}h<br><b>Média:</b> %{y:.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Média de Pageviews por Hora do Dia',
        xaxis_title='Hora do Dia',
        yaxis_title='Pageviews',
        height=500,
        hovermode='x unified',
        xaxis=dict(tickmode='linear', tick0=0, dtick=1, range=[-0.5, 23.5])
    )
    
    return fig

def create_anomaly_summary_chart(anomalies):
    """Cria gráfico resumo das anomalias"""
    anomaly_counts = {
        'Valores Zero': len(anomalies['zero_views']),
        'Valores Muito Baixos': len(anomalies['very_low_views']),
        'Outliers Estatísticos': len(anomalies['statistical_outliers']),
        'Variações Extremas': len(anomalies['extreme_variations']),
        'Dias com Horas Faltando': len(anomalies['missing_hours'])
    }
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(anomaly_counts.keys()),
            y=list(anomaly_counts.values()),
            marker_color=['red', 'orange', 'purple', 'blue', 'gray']
        )
    ])
    
    fig.update_layout(
        title='Resumo de Anomalias Detectadas',
        xaxis_title='Tipo de Anomalia',
        yaxis_title='Quantidade',
        height=400
    )
    
    return fig

def generate_report_text(df, stats, anomalies, brand_name="N/A"):
    """Gera texto do relatório"""
    report_lines = []
    report_lines.append("RELATÓRIO DE ANÁLISE DE ANOMALIAS - PAGEVIEWS")
    report_lines.append("=" * 60)
    report_lines.append(f"Marca: {brand_name.upper()}")
    report_lines.append(f"Período: {df['Date'].min().strftime('%d/%m/%Y')} a {df['Date'].max().strftime('%d/%m/%Y')}")
    report_lines.append(f"Total de registros: {len(df):,}")
    report_lines.append("")
    
    report_lines.append("ESTATÍSTICAS GERAIS:")
    report_lines.append(f"• Total de pageviews: {stats['total']:,}")
    report_lines.append(f"• Média por hora: {stats['mean']:.1f}")
    report_lines.append(f"• Mediana: {stats['median']:.1f}")
    report_lines.append(f"• Desvio padrão: {stats['std']:.1f}")
    report_lines.append("")
    
    report_lines.append("ANOMALIAS DETECTADAS:")
    report_lines.append(f"• Valores zero: {len(anomalies['zero_views'])} registros")
    report_lines.append(f"• Valores muito baixos: {len(anomalies['very_low_views'])} registros")
    report_lines.append(f"• Outliers estatísticos: {len(anomalies['statistical_outliers'])} registros")
    report_lines.append(f"• Variações extremas: {len(anomalies['extreme_variations'])} registros")
    report_lines.append(f"• Dias com horas faltando: {len(anomalies['missing_hours'])} dias")
    report_lines.append("")
    
    # Análise de cobertura por hora
    all_hours = set(range(24))
    hours_with_data = set(df['Hour'].unique())
    missing_hours = all_hours - hours_with_data
    
    # Calcular cobertura real considerando dias com horas faltando
    total_expected_hours = len(df['Date'].unique()) * 24
    total_actual_hours = len(df)
    coverage_percentage = (total_actual_hours / total_expected_hours) * 100 if total_expected_hours > 0 else 100
    
    report_lines.append("COBERTURA POR HORA:")
    report_lines.append(f"• Horas com dados: {total_actual_hours:,}/{total_expected_hours:,} ({coverage_percentage:.1f}%)")
    if missing_hours:
        report_lines.append(f"• Horas sem dados: {sorted(missing_hours)}")
    else:
        report_lines.append("• Todas as 24 horas possuem dados")
    
    # Informação adicional sobre dias com problemas
    if anomalies['missing_hours']:
        total_missing_hours = sum(len(day['missing_hours']) for day in anomalies['missing_hours'])
        report_lines.append(f"• Total de horas faltando: {total_missing_hours}")
    
    return "\n".join(report_lines)

@st.cache_data
def load_multiple_datasets(uploaded_files):
    """Carrega múltiplos datasets e retorna um dicionário com os dados"""
    datasets = {}
    
    for uploaded_file in uploaded_files:
        # Extrair nome da marca do nome do arquivo
        brand_name = uploaded_file.name.replace('.csv', '').replace('_', ' ').title()
        
        # Carregar dados
        df = load_and_prepare_data(uploaded_file)
        if df is not None:
            datasets[brand_name] = df
    
    return datasets

@st.cache_data
def load_csv_files_from_directory():
    """Carrega automaticamente todos os arquivos CSV da pasta csvs/"""
    datasets = {}
    
    # Buscar todos os arquivos CSV na pasta csvs/
    csv_files = glob.glob("csvs/*.csv")
    
    for csv_file in csv_files:
        try:
            # Extrair nome da marca do nome do arquivo
            brand_name = os.path.basename(csv_file).replace('.csv', '').replace('_', ' ').title()
            
            # Carregar dados usando pandas diretamente
            df = pd.read_csv(csv_file)
            
            # Verificar se as colunas necessárias existem
            if 'Date + hour (YYYYMMDDHH)' in df.columns and 'Views' in df.columns:
                # Formato novo: "Date + hour (YYYYMMDDHH)", "Views"
                date_hour_col = 'Date + hour (YYYYMMDDHH)'
                
                # Extrair data e hora da string
                df['Date'] = pd.to_datetime(df[date_hour_col].str.extract(r'(\w+ \d+, \d+)')[0], format='%b %d, %Y')
                
                # Extrair hora da string (formato "12AM", "1PM", etc.)
                hour_str = df[date_hour_col].str.extract(r'(\d+)(AM|PM)')
                hour_num = hour_str[0].astype(int)
                am_pm = hour_str[1]
                
                # Converter para formato 24h
                df['Hour'] = hour_num.where(am_pm == 'AM', hour_num + 12)
                df['Hour'] = df['Hour'].where(df['Hour'] != 12, 0)  # 12AM = 0h
                df['Hour'] = df['Hour'].where(df['Hour'] != 24, 12)  # 12PM = 12h
                
            elif 'Date' in df.columns and 'Hour' in df.columns and 'Views' in df.columns:
                # Formato antigo: "Date", "Hour", "Views"
                df['Date'] = pd.to_datetime(df['Date'], format='%b %d, %Y', errors='coerce')
                
            else:
                continue  # Pular arquivos que não têm o formato correto
            
            # Verificar se a conversão foi bem-sucedida
            if df['Date'].isna().any():
                continue  # Pular se há problemas na conversão de datas
            
            # Criar coluna de data e hora combinada
            df['DateTime'] = df['Date'] + pd.to_timedelta(df['Hour'], unit='h')
            
            # Ordenar por data e hora
            df = df.sort_values(['Date', 'Hour']).reset_index(drop=True)
            
            datasets[brand_name] = df
            
        except Exception as e:
            continue  # Pular arquivos com erro
    
    return datasets

def create_comparison_chart(datasets, stats_dict, selected_brands=None, aggregation='hourly'):
    """Cria gráfico de comparação entre marcas selecionadas
    
    Args:
        datasets: Dicionário com datasets das marcas
        stats_dict: Dicionário com estatísticas das marcas
        selected_brands: Lista de marcas selecionadas
        aggregation: 'hourly' para dados por hora, 'daily' para dados por dia
    """
    fig = go.Figure()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Se não especificou marcas, usar todas
    if selected_brands is None:
        selected_brands = list(datasets.keys())
    
    # Ordenar marcas por total de views (maior para menor) para tooltip consistente
    brand_totals = []
    for brand_name in selected_brands:
        if brand_name in datasets:
            total_views = datasets[brand_name]['Views'].sum()
            brand_totals.append((brand_name, total_views))
    
    # Ordenar por total de views (decrescente)
    brand_totals.sort(key=lambda x: x[1], reverse=True)
    ordered_brands = [brand[0] for brand in brand_totals]
    
    for i, brand_name in enumerate(ordered_brands):
        if brand_name in datasets:
            df = datasets[brand_name]
            color = colors[i % len(colors)]
            
            if aggregation == 'hourly':
                # Usar dados por hora (DateTime)
                fig.add_trace(go.Scatter(
                    x=df['DateTime'],
                    y=df['Views'],
                    mode='lines+markers',
                    name=brand_name,
                    line=dict(color=color, width=1.5),
                    marker=dict(size=3),
                    hovertemplate=f'<b>{brand_name}</b><br>' +
                                  'Data: %{x|%d/%m/%Y}<br>' +
                                  'Hora: %{x|%H}h<br>' +
                                  'Views: %{y:,}<extra></extra>'
                ))
            else:
                # Agrupar por data para ter um valor por dia
                daily_views = df.groupby('Date')['Views'].sum().reset_index()
                
                fig.add_trace(go.Scatter(
                    x=daily_views['Date'],
                    y=daily_views['Views'],
                    mode='lines+markers',
                    name=brand_name,
                    line=dict(color=color, width=2),
                    marker=dict(size=4),
                    hovertemplate=f'<b>{brand_name}</b><br>' +
                                  'Data: %{x|%d/%m/%Y}<br>' +
                                  'Views: %{y:,}<extra></extra>'
                ))
    
    title_suffix = "por Hora" if aggregation == 'hourly' else "por Dia"
    fig.update_layout(
        title=f'Comparação de Pageviews entre Marcas Selecionadas - {title_suffix}',
        xaxis_title='Data e Hora' if aggregation == 'hourly' else 'Data',
        yaxis_title='Pageviews',
        height=500,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_comparison_metrics(datasets, stats_dict):
    """Cria métricas de comparação entre marcas"""
    comparison_data = []
    
    for brand_name, stats in stats_dict.items():
        comparison_data.append({
            'Marca': brand_name,
            'Total Pageviews': stats['total'],
            'Total Pageviews Formatado': f"{stats['total']:,}",
            'Média por Hora': stats['mean'],
            'Média por Hora Formatada': f"{stats['mean']:.1f}",
            'Mediana': f"{stats['median']:.1f}",
            'Desvio Padrão': f"{stats['std']:.1f}",
            'Máximo': f"{stats['max']:,}",
            'Mínimo': f"{stats['min']:,}"
        })
    
    return pd.DataFrame(comparison_data)

def create_anomaly_analysis_chart(anomaly_comparison_data):
    """Cria gráfico visual para análise de anomalias por marca"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Total de Anomalias', 'Anomalias Críticas', 'Outliers Estatísticos', 'Horas Faltando'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )
    
    brands = [item['Marca'] for item in anomaly_comparison_data]
    
    # Total de anomalias
    total_anomalies = [item['Total Anomalias'] for item in anomaly_comparison_data]
    fig.add_trace(go.Bar(x=brands, y=total_anomalies, name='Total', marker_color='lightcoral'), row=1, col=1)
    
    # Anomalias críticas (zero + baixos)
    critical_anomalies = [item['Valores Zero'] + item['Valores Baixos'] for item in anomaly_comparison_data]
    fig.add_trace(go.Bar(x=brands, y=critical_anomalies, name='Críticas', marker_color='red'), row=1, col=2)
    
    # Outliers estatísticos
    outliers = [item['Outliers'] for item in anomaly_comparison_data]
    fig.add_trace(go.Bar(x=brands, y=outliers, name='Outliers', marker_color='orange'), row=2, col=1)
    
    # Horas faltando
    missing_hours = [item['Horas Faltando'] for item in anomaly_comparison_data]
    fig.add_trace(go.Bar(x=brands, y=missing_hours, name='Horas Faltando', marker_color='gray'), row=2, col=2)
    
    fig.update_layout(height=600, showlegend=False, title_text="Análise Visual de Anomalias por Marca")
    return fig

def calculate_anomaly_severity_score(anomaly_data):
    """Calcula um score de severidade das anomalias (0-100)"""
    # Pesos para diferentes tipos de anomalias
    weights = {
        'zero_views': 10,      # Muito crítico
        'very_low_views': 5,   # Crítico
        'statistical_outliers': 2,  # Moderado
        'extreme_variations': 3,    # Moderado-alto
        'missing_hours': 1     # Baixo
    }
    
    total_records = anomaly_data.get('total_records', 1)
    
    # Calcular score ponderado
    score = (
        len(anomaly_data['zero_views']) * weights['zero_views'] +
        len(anomaly_data['very_low_views']) * weights['very_low_views'] +
        len(anomaly_data['statistical_outliers']) * weights['statistical_outliers'] +
        len(anomaly_data['extreme_variations']) * weights['extreme_variations'] +
        len(anomaly_data['missing_hours']) * weights['missing_hours']
    )
    
    # Se não há anomalias, retornar 0
    if score == 0:
        return 0
    
    # Normalizar por total de registros (usar uma base mais conservadora)
    # Assumir que 1% de anomalias críticas = score 50
    normalized_score = min(100, (score / total_records) * 5000)
    
    return normalized_score

def get_anomaly_status(score):
    """Retorna o status baseado no score de severidade"""
    if score >= 50:
        return "🔴 Crítico"
    elif score >= 25:
        return "🟡 Atenção"
    elif score >= 10:
        return "🟠 Moderado"
    elif score > 0:
        return "🟢 Baixo"
    else:
        return "✅ Saudável"

def check_password():
    """Verifica se a senha está correta"""
    def password_entered():
        """Verifica se a senha inserida está correta"""
        if "password" in st.session_state:
            if st.session_state["password"] == st.secrets.get("password", "admin123"):
                st.session_state["password_correct"] = True
                del st.session_state["password"]  # Remove a senha da sessão por segurança
            else:
                st.session_state["password_correct"] = False

    # Retorna True se a senha estiver correta
    if st.session_state.get("password_correct", False):
        return True

    # Tela de login
    st.markdown("""
    <div style="text-align: center; padding: 2rem;">
        <h1 style="color: #1f77b4; margin-bottom: 2rem;">🔐 Acesso Restrito</h1>
        <p style="font-size: 1.2rem; color: #666;">Digite a senha para acessar o dashboard</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Input da senha
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        password = st.text_input(
            "Senha:",
            type="password",
            key="password_input",
            help="Digite a senha para acessar o dashboard"
        )
        
        if st.button("🚀 ENTRAR", type="primary", width="stretch"):
            if password:
                st.session_state["password"] = password
                password_entered()
                st.rerun()  # Força o rerun para aplicar a mudança
            else:
                st.error("❌ Por favor, digite a senha!")
    
    if "password_correct" in st.session_state:
        if not st.session_state["password_correct"]:
            st.error("❌ Senha incorreta. Tente novamente.")
    
    return False

def main():
    # Cabeçalho principal
    st.markdown('<h1 class="main-header">📊 Análise de Anomalias - Pageviews</h1>', unsafe_allow_html=True)
    
    # Carregar CSVs automaticamente da pasta (com cache)
    datasets = load_csv_files_from_directory()

    if datasets:
        st.sidebar.subheader("Marcas Carregadas")
        
        # Definir BYD como padrão se disponível
        available_brands = list(datasets.keys())
        default_index = 0
        if "Byd" in available_brands:
            default_index = available_brands.index("Byd")
        
        selected_brand = st.sidebar.selectbox(
            "Selecione a marca para análise:",
            options=available_brands,
            index=default_index,
        )
        # Informações da marca selecionada
        selected_df = datasets[selected_brand]
        selected_stats = calculate_statistics(selected_df)
        
        st.sidebar.metric("Total Views", f"{selected_stats['total']:,}")
        st.sidebar.metric("Média/Hora", f"{selected_stats['mean']:.0f}")
        
        # Dataset selecionado
        df = datasets[selected_brand]
        
        # Date picker geral para filtrar todos os dados
        st.sidebar.subheader("📅 Filtro de Período Geral")
        st.sidebar.caption("Filtra todos os dados para o período selecionado")
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            # Garantir que o valor padrão esteja dentro do range válido
            min_date = df['Date'].min().date()
            max_date = df['Date'].max().date()
            default_start = st.session_state.get('global_start_date', min_date)
            
            # Se o valor salvo estiver fora do range, usar o min_date
            if default_start < min_date or default_start > max_date:
                default_start = min_date
            
            start_date_global = st.date_input(
                "Data Inicial",
                value=default_start,
                min_value=min_date,
                max_value=max_date,
                help="Data inicial para análise geral",
                key="global_start_date"
            )
        
        with col2:
            # Garantir que o valor padrão esteja dentro do range válido
            default_end = st.session_state.get('global_end_date', max_date)
            
            # Se o valor salvo estiver fora do range, usar o max_date
            if default_end < min_date or default_end > max_date:
                default_end = max_date
            
            end_date_global = st.date_input(
                "Data Final",
                value=default_end,
                min_value=min_date,
                max_value=max_date,
                help="Data final para análise geral",
                key="global_end_date"
            )
        
        # Validar se a data final é maior que a inicial
        if end_date_global < start_date_global:
            st.sidebar.error("⚠️ **Erro:** Data final deve ser maior que a data inicial!")
            st.sidebar.caption("Ajuste as datas para continuar a análise")
            # Usar dados originais se as datas estiverem incorretas
            df_filtered_global = df
        else:
            # Aplicar filtro de data global
            start_datetime_global = pd.to_datetime(start_date_global)
            end_datetime_global = pd.to_datetime(end_date_global) + pd.Timedelta(days=1)
            
            df_filtered_global = df[(df['Date'] >= start_datetime_global) & (df['Date'] < end_datetime_global)]
            
            # Verificar se há dados após o filtro
            if len(df_filtered_global) == 0:
                st.sidebar.warning("⚠️ **Atenção:** Nenhum dado encontrado para o período selecionado!")
                st.sidebar.caption("Usando todos os dados disponíveis")
                df_filtered_global = df
        
        # Configurações de detecção
        st.sidebar.subheader("🔍 Parâmetros de Detecção")
        
        # Range de horas
        st.sidebar.markdown("**⏰ Range de Horas**")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_hour = st.selectbox(
                "Hora inicial",
                options=list(range(0, 24)),
                index=0,
                format_func=lambda x: f"{x:02d}h",
                help="Hora inicial para análise de anomalias"
            )
        with col2:
            end_hour = st.selectbox(
                "Hora final",
                options=list(range(0, 24)),
                index=23,
                format_func=lambda x: f"{x:02d}h",
                help="Hora final para análise de anomalias"
            )
        
        # Ajustar end_hour se for menor que start_hour
        if end_hour <= start_hour:
            end_hour = start_hour
            st.sidebar.warning("⚠️ Hora final ajustada para ser maior que a inicial")
        
        extreme_threshold = st.sidebar.slider(
            "Limite para variações extremas (%)",
            min_value=50,
            max_value=200,
            value=80,
            help="Variações acima deste percentual da média serão consideradas extremas"
        ) / 100
        
        
        # Calcular estatísticas
        stats = calculate_statistics(df_filtered_global)
        
        # Calcular médias históricas por hora
        hourly_means = calculate_hourly_means(df_filtered_global)
        
        st.sidebar.caption("Valores baixos são detectados comparando com a média histórica de cada hora específica")
        
        # Configurar threshold de percentual da média histórica
        threshold_percentage = st.sidebar.slider(
            "Limite para valores baixos (% da média histórica da hora)",
            min_value=10,
            max_value=80,
            value=50,
            help="Valores abaixo deste percentual da média histórica da hora serão considerados baixos"
        ) / 100
        # Resumo das marcas
        with st.sidebar.expander("📋 Detalhes das marcas", expanded=False):
            for brand_name, brand_df in datasets.items():
                total_views = brand_df['Views'].sum()
                date_range = f"{brand_df['Date'].min().strftime('%d/%m/%Y')} a {brand_df['Date'].max().strftime('%d/%m/%Y')}"
                total_records = len(brand_df)
                avg_views = brand_df['Views'].mean()
                
                st.markdown(f"""
                **🏢 {brand_name}**
                - 📈 **{total_views:,}** views totais
                - 📅 **{date_range}**
                - 📊 **{total_records:,}** registros
                - 📉 **{avg_views:.0f}** views/hora (média)
                """)
                st.markdown("---")
        
    
        # Detectar anomalias
        anomalies = detect_anomalies(df_filtered_global, stats, threshold_percentage, extreme_threshold, start_hour, end_hour)
        
        # Métricas principais
        st.subheader(f"📈 Métricas Principais - {selected_brand.upper()}")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total de Pageviews",
                value=f"{stats['total']:,}",
                delta=None
            )
        
        with col2:
            st.metric(
                label="Média por Hora",
                value=f"{stats['mean']:.0f}",
                delta=None
            )
        
        with col3:
            total_anomalies = (len(anomalies['zero_views']) + 
                             len(anomalies['very_low_views']) + 
                             len(anomalies['statistical_outliers']) + 
                             len(anomalies['extreme_variations']) + 
                             len(anomalies['missing_hours']))
            st.metric(
                label="Total de Anomalias",
                value=total_anomalies,
                delta=None
            )
        
        with col4:
            critical_anomalies = len(anomalies['zero_views']) + len(anomalies['very_low_views'])
            st.metric(
                label="Anomalias Críticas",
                value=critical_anomalies,
                delta=None
            )
        
        
        # Tabs para diferentes visualizações
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🚨 Anomalias", 
            "📊 Série Temporal", 
            "⏰ Análise por Hora", 
            "📋 Relatório",
            "🔄 Comparação de Marcas"
        ])
        
        with tab1:
            st.subheader("🚨 Detalhes das Anomalias")
            
            # Resumo das anomalias
            st.plotly_chart(
                create_anomaly_summary_chart(anomalies),
                config={
                    "displayModeBar": True,
                    "modeBarButtonsToRemove": ["pan2d", "lasso2d", "select2d"],
                    "displaylogo": False
                }
            )
            
            # Detalhes específicos - Valores Zero
            if not anomalies['zero_views'].empty:
                st.subheader("❌ Valores Zero")
                st.dataframe(
                    anomalies['zero_views'][['Date', 'Hour', 'Views']].style.format({
                        'Date': lambda x: x.strftime('%d/%m/%Y'),
                        'Views': '{:,.0f}'
                    }),
                    width="stretch"
                )
            
            # Valores Muito Baixos
            if not anomalies['very_low_views'].empty:
                st.subheader("⚠️ Valores Muito Baixos")
                st.dataframe(
                    anomalies['very_low_views'][['Date', 'Hour', 'Views']].style.format({
                        'Date': lambda x: x.strftime('%d/%m/%Y'),
                        'Views': '{:,.0f}'
                    }),
                    width="stretch"
                )
            
            # Outliers Estatísticos
            if not anomalies['statistical_outliers'].empty:
                st.subheader("📈 Outliers Estatísticos (Top 10)")
                outliers_display = anomalies['statistical_outliers'].copy()
                outliers_display['Variation_%'] = ((outliers_display['Views'] - stats['mean']) / stats['mean'] * 100).round(1)
                st.dataframe(
                    outliers_display[['Date', 'Hour', 'Views', 'Variation_%']].head(10).style.format({
                        'Date': lambda x: x.strftime('%d/%m/%Y'),
                        'Views': '{:,.0f}',
                        'Variation_%': '{:+.1f}%'
                    }),
                    width="stretch"
                )
            
            # Variações extremas
            if not anomalies['extreme_variations'].empty:
                st.subheader("📊 Variações Extremas")
                extreme_display = anomalies['extreme_variations'].copy()
                extreme_display['Variation_%'] = ((extreme_display['Views'] - stats['mean']) / stats['mean'] * 100).round(1)
                st.dataframe(
                    extreme_display[['Date', 'Hour', 'Views', 'Variation_%']].style.format({
                        'Date': lambda x: x.strftime('%d/%m/%Y'),
                        'Views': '{:,.0f}',
                        'Variation_%': '{:+.1f}%'
                    }),
                    width="stretch"
                )
            
            # Horas faltando
            if anomalies['missing_hours']:
                st.subheader("⏰ Horas Faltando")
                missing_df = pd.DataFrame(anomalies['missing_hours'])
                missing_df['missing_hours_str'] = missing_df['missing_hours'].apply(
                    lambda x: ', '.join([f"{h:02d}h" for h in x])
                )
                st.dataframe(
                    missing_df[['date', 'missing_hours_str']].style.format({
                        'date': lambda x: x.strftime('%d/%m/%Y')
                    }),
                    width="stretch"
                )
        
        with tab2:
            st.plotly_chart(
                create_time_series_chart(df_filtered_global, anomalies, stats),
                key="time_series_chart",
                config={
                    "displayModeBar": True,
                    "modeBarButtonsToRemove": ["pan2d", "lasso2d", "select2d"],
                    "displaylogo": False
                }
            )
        
        with tab3:
            # Gráfico geral por hora (sem filtro)
            st.subheader("📊 Análise Geral por Hora (Todo o Período)")
            st.plotly_chart(
                create_hourly_analysis_chart(df),
                key="hourly_analysis_general",
                config={
                    "displayModeBar": True,
                    "modeBarButtonsToRemove": ["pan2d", "lasso2d", "select2d"],
                    "displaylogo": False
                }
            )
            
            st.markdown("---")
            
            # Informação sobre o filtro global aplicado
            st.info(f"📊 **Análise por Hora** - Período: {start_date_global} a {end_date_global}")
            st.caption("Use o filtro de período geral na sidebar para ajustar o período de análise")
        
        with tab4:
            st.subheader("📋 Relatório Completo")
            
            # Gerar relatório
            report_text = generate_report_text(df_filtered_global, stats, anomalies, selected_brand)
            st.text_area("-", report_text, height=400)
            
            # Botão de download
            st.download_button(
                label="📥 Baixar Relatório",
                data=report_text,
                file_name=f"{selected_brand.lower()}-analise-pageview-ga4-{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
            with tab5:
                st.subheader("🔄 Comparação entre Marcas")
                
                # Seleção de marcas para comparação
                st.subheader("🎯 Seleção de Marcas")
                available_brands = [brand.upper() for brand in datasets.keys()]
                col1 = st.columns(1)[0]
                with col1:
                    selected_brands_for_comparison = st.multiselect(
                        "Selecione as marcas para comparar:",
                        options=available_brands,
                        default=available_brands[:3] if len(available_brands) >= 3 else available_brands,
                        help="Escolha quais marcas deseja comparar"
                    )
                
                if selected_brands_for_comparison:
                    # Aplicar filtro global também na comparação
                    selected_datasets = {}
                    for brand_upper in selected_brands_for_comparison:
                        # Encontrar a chave original no dicionário (case-insensitive)
                        original_brand = next((k for k in datasets.keys() if k.upper() == brand_upper), None)
                        if original_brand:
                            brand_df = datasets[original_brand]
                            # Aplicar o mesmo filtro de data global
                            brand_df_filtered = brand_df[(brand_df['Date'] >= start_datetime_global) & (brand_df['Date'] < end_datetime_global)]
                            selected_datasets[brand_upper] = brand_df_filtered
                    
                    all_stats = {}
                    for brand_name, brand_df in selected_datasets.items():
                        all_stats[brand_name] = calculate_statistics(brand_df)
                    
                    # Gráfico de comparação
                    st.subheader("📈 Evolução Temporal Comparativa")
                    
                    # Opção para alternar entre visualização diária e horária
                    col1, col2 = st.columns([3, 1])
                    with col2:
                        aggregation_mode = st.radio(
                            "Granularidade:",
                            options=['hourly', 'daily'],
                            format_func=lambda x: "Por Hora" if x == 'hourly' else "Por Dia",
                            horizontal=True,
                            help="Escolha se deseja ver os dados por hora ou agregados por dia"
                        )
                    
                    st.plotly_chart(
                        create_comparison_chart(selected_datasets, all_stats, selected_brands_for_comparison, aggregation_mode),
                        key="comparison_chart",
                        config={
                            "displayModeBar": True,
                            "modeBarButtonsToRemove": ["pan2d", "lasso2d", "select2d"],
                            "displaylogo": False
                        }
                    )
                
                    # Métricas comparativas
                    st.subheader("📊 Métricas Comparativas")
                    comparison_df = create_comparison_metrics(selected_datasets, all_stats)
                    
                    # Mostrar apenas colunas formatadas na tabela
                    display_df = comparison_df[['Marca', 'Total Pageviews Formatado', 'Média por Hora Formatada', 'Mediana', 'Desvio Padrão', 'Máximo', 'Mínimo']]
                    display_df.columns = ['Marca', 'Total Pageviews', 'Média por Hora', 'Mediana', 'Desvio Padrão', 'Máximo', 'Mínimo']
                    st.dataframe(display_df, width="stretch")
                
                    # Ranking de performance
                    st.subheader("🏆 Ranking de Performance")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**📈 Top 3 - Maior Volume Total**")
                        # Garantir que sempre mostre 3 entradas
                        top_total = comparison_df.nlargest(min(3, len(comparison_df)), 'Total Pageviews')[['Marca', 'Total Pageviews Formatado']]
                        top_total.columns = ['Marca', 'Total Pageviews']
                        # Adicionar ranking
                        top_total['Ranking'] = range(1, len(top_total) + 1)
                        top_total = top_total[['Ranking', 'Marca', 'Total Pageviews']]
                        st.dataframe(top_total, width="stretch")
                    
                    with col2:
                        st.markdown("**📊 Top 3 - Maior Média por Hora**")
                        # Garantir que sempre mostre 3 entradas
                        top_avg = comparison_df.nlargest(min(3, len(comparison_df)), 'Média por Hora')[['Marca', 'Média por Hora Formatada']]
                        top_avg.columns = ['Marca', 'Média por Hora']
                        # Adicionar ranking
                        top_avg['Ranking'] = range(1, len(top_avg) + 1)
                        top_avg = top_avg[['Ranking', 'Marca', 'Média por Hora']]
                        st.dataframe(top_avg, width="stretch")
                    
                    with col3:
                        st.markdown("**📊 Top 3 - Menor Desvio Padrão**")
                        # Converter para float e usar a coluna numérica original para ordenação
                        comparison_df_temp = comparison_df.copy()
                        comparison_df_temp['Desvio Padrão Num'] = comparison_df_temp['Desvio Padrão'].astype(float)
                        # Garantir que sempre mostre 3 entradas
                        top_stable = comparison_df_temp.nsmallest(min(3, len(comparison_df_temp)), 'Desvio Padrão Num')[['Marca', 'Desvio Padrão']]
                        # Adicionar ranking
                        top_stable['Ranking'] = range(1, len(top_stable) + 1)
                        top_stable = top_stable[['Ranking', 'Marca', 'Desvio Padrão']]
                        st.dataframe(top_stable, width="stretch")
                
                    # Análise de anomalias comparativa
                    st.subheader("🚨 Análise de Anomalias por Marca")
                    
                    anomaly_comparison = []
                    anomaly_details = {}
                    
                    for brand_name, brand_df in selected_datasets.items():
                        brand_stats = all_stats[brand_name]
                        # Usar threshold de 50% da média histórica para comparação
                        brand_anomalies = detect_anomalies(brand_df, brand_stats, 0.5, 0.8, 0, 23)
                        
                        # Calcular score de severidade
                        anomaly_data_with_records = brand_anomalies.copy()
                        anomaly_data_with_records['total_records'] = len(brand_df)
                        severity_score = calculate_anomaly_severity_score(anomaly_data_with_records)
                        status = get_anomaly_status(severity_score)
                        
                        total_anomalies = (len(brand_anomalies['zero_views']) + 
                                         len(brand_anomalies['very_low_views']) + 
                                         len(brand_anomalies['statistical_outliers']) + 
                                         len(brand_anomalies['extreme_variations']) + 
                                         len(brand_anomalies['missing_hours']))
                        
                        critical_anomalies = len(brand_anomalies['zero_views']) + len(brand_anomalies['very_low_views'])
                        
                        anomaly_comparison.append({
                            'Marca': brand_name,
                            'Status': status,
                            'Score Severidade': f"{severity_score:.1f}",
                            'Total Anomalias': total_anomalies,
                            'Críticas': critical_anomalies,
                            'Valores Zero': len(brand_anomalies['zero_views']),
                            'Valores Baixos': len(brand_anomalies['very_low_views']),
                            'Outliers': len(brand_anomalies['statistical_outliers']),
                            'Variações Extremas': len(brand_anomalies['extreme_variations']),
                            'Horas Faltando': len(brand_anomalies['missing_hours'])
                        })
                        
                        # Guardar detalhes para análise posterior
                        anomaly_details[brand_name] = brand_anomalies
                    
                    # Ordenar por score de severidade (maior primeiro)
                    anomaly_comparison.sort(key=lambda x: float(x['Score Severidade']), reverse=True)
                    
                    # Gráfico visual das anomalias
                    st.plotly_chart(
                        create_anomaly_analysis_chart(anomaly_comparison),
                        key="anomaly_analysis_chart",
                        config={
                            "displayModeBar": True,
                            "modeBarButtonsToRemove": ["pan2d", "lasso2d", "select2d"],
                            "displaylogo": False
                        }
                    )
                    
                 
        
        # Upload de arquivos - no final da sidebar
        st.sidebar.markdown("---")
        st.sidebar.subheader("📁 Upload de Dados Adicionais")
        st.sidebar.caption("Adicione mais arquivos CSV para análise")
        
        uploaded_files = st.sidebar.file_uploader(
            "Selecione os arquivos CSV",
            type=['csv'],
            accept_multiple_files=True,
            help="📋 **Formato esperado:**\n• Date, Hour, Views\n• Date + hour (YYYYMMDDHH), Views\n\n💡 **Dica:** Faça upload de múltiplos arquivos para comparar marcas"
        )
        
        # Se há arquivos enviados, também carregar eles (com cache)
        if uploaded_files:
            uploaded_datasets = load_multiple_datasets(uploaded_files)
            datasets.update(uploaded_datasets)  # Combinar com CSVs da pasta
            st.sidebar.success(f"✅ {len(uploaded_files)} arquivo(s) adicionado(s) com sucesso!")
            st.rerun()  # Recarregar para mostrar as novas marcas
    else:
        # Mostrar mensagem quando não há dados válidos
        st.error("❌ Nenhum arquivo válido foi carregado. Verifique o formato dos dados.")
        st.info("💡 **Dica:** Os arquivos devem conter colunas: 'Date + hour (YYYYMMDDHH)' e 'Views' OU 'Date', 'Hour' e 'Views'")

if __name__ == "__main__":
    # Verificar senha antes de mostrar o dashboard
    if check_password():
        main()
    else:
        st.stop()  # Para a execução se a senha estiver incorreta
