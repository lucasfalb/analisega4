import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from datetime import datetime, timedelta
import statistics
from collections import defaultdict
import base64
import time
import os
import glob
import hashlib

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

def detect_anomalies(df, stats, low_threshold=50, extreme_threshold=0.8, start_hour=0, end_hour=23):
    """Detecta diferentes tipos de anomalias"""
    # Filtrar dados pelo range de horas
    df_filtered = df[(df['Hour'] >= start_hour) & (df['Hour'] <= end_hour)]
    
    anomalies = {
        'zero_views': df_filtered[df_filtered['Views'] == 0],
        'very_low_views': df_filtered[(df_filtered['Views'] > 0) & (df_filtered['Views'] < low_threshold)],
        'statistical_outliers': df_filtered[
            (df_filtered['Views'] < stats['lower_bound']) | 
            (df_filtered['Views'] > stats['upper_bound'])
        ],
        'extreme_variations': df_filtered[
            abs(df_filtered['Views'] - stats['mean']) / stats['mean'] > extreme_threshold
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
    """Cria gráfico de série temporal com anomalias destacadas"""
    fig = go.Figure()
    
    # Linha principal
    fig.add_trace(go.Scatter(
        x=df['DateTime'],
        y=df['Views'],
        mode='lines',
        name='Pageviews',
        line=dict(color='steelblue', width=1),
        opacity=0.7
    ))
    
    # Destacar anomalias
    if not anomalies['zero_views'].empty:
        fig.add_trace(go.Scatter(
            x=anomalies['zero_views']['DateTime'],
            y=anomalies['zero_views']['Views'],
            mode='markers',
            name='Valores Zero',
            marker=dict(color='red', size=8, symbol='x'),
            hovertemplate='<b>%{x}</b><br>Views: %{y}<extra></extra>'
        ))
    
    if not anomalies['very_low_views'].empty:
        fig.add_trace(go.Scatter(
            x=anomalies['very_low_views']['DateTime'],
            y=anomalies['very_low_views']['Views'],
            mode='markers',
            name='Valores Muito Baixos',
            marker=dict(color='orange', size=6),
            hovertemplate='<b>%{x}</b><br>Views: %{y}<extra></extra>'
        ))
    
    if not anomalies['statistical_outliers'].empty:
        fig.add_trace(go.Scatter(
            x=anomalies['statistical_outliers']['DateTime'],
            y=anomalies['statistical_outliers']['Views'],
            mode='markers',
            name='Outliers Estatísticos',
            marker=dict(color='purple', size=6),
            hovertemplate='<b>%{x}</b><br>Views: %{y}<extra></extra>'
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

def generate_report_text(df, stats, anomalies):
    """Gera texto do relatório"""
    report_lines = []
    report_lines.append("RELATÓRIO DE ANÁLISE DE ANOMALIAS - PAGEVIEWS")
    report_lines.append("=" * 60)
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
    
    report_lines.append("COBERTURA POR HORA:")
    report_lines.append(f"• Horas com dados: {len(hours_with_data)}/24 ({len(hours_with_data)/24*100:.1f}%)")
    if missing_hours:
        report_lines.append(f"• Horas sem dados: {sorted(missing_hours)}")
    else:
        report_lines.append("• Todas as 24 horas possuem dados")
    
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

def create_comparison_chart(datasets, stats_dict, selected_brands=None):
    """Cria gráfico de comparação entre marcas selecionadas"""
    fig = go.Figure()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Se não especificou marcas, usar todas
    if selected_brands is None:
        selected_brands = list(datasets.keys())
    
    for i, brand_name in enumerate(selected_brands):
        if brand_name in datasets:
            df = datasets[brand_name]
            color = colors[i % len(colors)]
            
            # Agrupar por data para ter um valor por dia
            daily_views = df.groupby('Date')['Views'].sum().reset_index()
            
            fig.add_trace(go.Scatter(
                x=daily_views['Date'],
                y=daily_views['Views'],
                mode='lines+markers',
                name=brand_name,
                line=dict(color=color, width=2),
                marker=dict(size=4),
                hovertemplate=f'<b>{brand_name}</b><br>%{{x}}<br>Views: %{{y:,}}<extra></extra>'
            ))
    
    fig.update_layout(
        title='Comparação de Pageviews entre Marcas Selecionadas',
        xaxis_title='Data',
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
            'CV (%)': stats['std']/stats['mean']*100,
            'CV (%) Formatado': f"{stats['std']/stats['mean']*100:.1f}%",
            'Máximo': f"{stats['max']:,}",
            'Mínimo': f"{stats['min']:,}"
        })
    
    return pd.DataFrame(comparison_data)

def check_password():
    """Verifica se a senha está correta"""
    def password_entered():
        """Verifica se a senha inserida está correta"""
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
    
    # Campo de senha
    password_input = st.text_input(
        "Senha", 
        type="password", 
        key="password",
        help="Digite a senha para acessar o dashboard de análise de anomalias",
        placeholder="Digite sua senha aqui..."
    )
    
    # Botão ENTRAR
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 ENTRAR", type="primary", use_container_width=True):
            password_entered()
    
    if "password_correct" in st.session_state:
        if not st.session_state["password_correct"]:
            st.error("❌ Senha incorreta. Tente novamente.")
    
    return False

def main():
    # Cabeçalho principal
    st.markdown('<h1 class="main-header">📊 Análise de Anomalias - Pageviews</h1>', unsafe_allow_html=True)
    
  
    # Sidebar para configurações
    st.sidebar.header("⚙️ Configurações")
    
    # Upload de arquivos
    st.sidebar.markdown("### 📁 Upload de Dados")
    
    uploaded_files = st.sidebar.file_uploader(
        "Selecione os arquivos CSV",
        type=['csv'],
        accept_multiple_files=True,
        help="📋 **Formato esperado:**\n• Date, Hour, Views\n• Date + hour (YYYYMMDDHH), Views\n\n💡 **Dica:** Faça upload de múltiplos arquivos para comparar marcas"
    )
    # Carregar CSVs automaticamente da pasta (com cache)
    datasets = load_csv_files_from_directory()
    
    # Se há arquivos enviados, também carregar eles (com cache)
    if uploaded_files:
        uploaded_datasets = load_multiple_datasets(uploaded_files)
        datasets.update(uploaded_datasets)  # Combinar com CSVs da pasta
    
    if datasets:
        # Informações sobre marcas carregadas
        st.sidebar.markdown("---")
        st.sidebar.subheader("📊 Marcas Carregadas")
        
        # Seleção de marca na sidebar
        st.sidebar.subheader("🎯 Análise Individual")
        selected_brand = st.sidebar.selectbox(
            "Escolha uma marca para análise detalhada:",
            options=list(datasets.keys()),
            help="Selecione uma marca para análise individual nas abas específicas"
        )
        
        # Informações da marca selecionada
        selected_df = datasets[selected_brand]
        selected_stats = calculate_statistics(selected_df)
        
        st.sidebar.metric("Total Views", f"{selected_stats['total']:,}")
        st.sidebar.metric("Média/Hora", f"{selected_stats['mean']:.0f}")
        st.sidebar.metric("CV (%)", f"{selected_stats['std']/selected_stats['mean']*100:.1f}%")
        
        # Dataset selecionado
        df = datasets[selected_brand]
        
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
        
        low_threshold = st.sidebar.slider(
            "Limite para valores baixos",
            min_value=10,
            max_value=200,
            value=50,
            help="Valores abaixo deste limite serão considerados muito baixos"
        )
        
        extreme_threshold = st.sidebar.slider(
            "Limite para variações extremas (%)",
            min_value=50,
            max_value=200,
            value=80,
            help="Variações acima deste percentual da média serão consideradas extremas"
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
        
        # Calcular estatísticas
        stats = calculate_statistics(df)
        
        # Detectar anomalias
        anomalies = detect_anomalies(df, stats, low_threshold, extreme_threshold, start_hour, end_hour)
        
        # Métricas principais
        st.subheader("📈 Métricas Principais")
        
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
            total_anomalies = len(anomalies['zero_views']) + len(anomalies['very_low_views']) + len(anomalies['statistical_outliers'])
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
        
        # Alertas
        if critical_anomalies > 0:
            st.error(f"⚠️ **ATENÇÃO**: {critical_anomalies} anomalias críticas detectadas!")
        elif total_anomalies > 20:
            st.warning(f"⚠️ {total_anomalies} anomalias detectadas. Verifique os dados.")
        else:
            st.success("✅ Nenhuma instabilidade crítica detectada.")
        
        # Tabs para diferentes visualizações
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Série Temporal", 
            "⏰ Análise por Hora", 
            "🚨 Anomalias", 
            "📋 Relatório",
            "🔄 Comparação de Marcas"
        ])
        
        with tab1:
            st.plotly_chart(
                create_time_series_chart(df, anomalies, stats),
                use_container_width=True,
                key="time_series_chart"
            )
        
        with tab2:
            # Gráfico geral por hora (sem filtro)
            st.subheader("📊 Análise Geral por Hora (Todo o Período)")
            st.plotly_chart(
                create_hourly_analysis_chart(df),
                use_container_width=True,
                key="hourly_analysis_general"
            )
            
            st.markdown("---")
            
            # Date picker para análise detalhada
            st.subheader("📅 Seleção de Período para Análise Detalhada")
            col1, col2 = st.columns(2)
            
            with col1:
                start_date = st.date_input(
                    "Data Inicial",
                    value=df['Date'].min().date(),
                    min_value=df['Date'].min().date(),
                    max_value=df['Date'].max().date(),
                    help="Selecione a data inicial para análise detalhada"
                )
            
            with col2:
                end_date = st.date_input(
                    "Data Final",
                    value=df['Date'].max().date(),
                    min_value=df['Date'].min().date(),
                    max_value=df['Date'].max().date(),
                    help="Selecione a data final para análise detalhada"
                )
                
                # Filtrar dados pelo período selecionado
                start_datetime = pd.to_datetime(start_date)
                end_datetime = pd.to_datetime(end_date) + pd.Timedelta(days=1)
                
                df_filtered = df[(df['Date'] >= start_datetime) & (df['Date'] < end_datetime)]
                
                if len(df_filtered) == 0:
                    st.warning("⚠️ Nenhum dado encontrado para o período selecionado.")
                else:
                    st.info(f"📊 Analisando {len(df_filtered)} registros de {start_date} a {end_date}")
                    
                    # Gráfico para o período selecionado
                    st.plotly_chart(
                        create_hourly_analysis_chart(df_filtered),
                        use_container_width=True,
                        key="hourly_analysis_filtered"
                    )
                    
                    # Análise detalhada por hora para o período selecionado
                    st.subheader("📊 Análise Detalhada por Hora (Período Selecionado)")
                
                    # Criar DataFrame com todas as horas do dia (0-23)
                    all_hours = pd.DataFrame({'Hour': range(24)})
                    
                    # Calcular estatísticas para horas com dados (período filtrado)
                    hourly_stats = df_filtered.groupby('Hour')['Views'].agg([
                        'count', 'mean', 'std', 'min', 'max', 
                        ('q25', lambda x: x.quantile(0.25)),
                        ('q75', lambda x: x.quantile(0.75))
                    ]).round(1)
                    
                    # Fazer merge com todas as horas
                    hourly_detailed = all_hours.merge(hourly_stats, on='Hour', how='left')
                    
                    # Preencher valores NaN com 0 ou 'N/A'
                    hourly_detailed['count'] = hourly_detailed['count'].fillna(0).astype(int)
                    hourly_detailed['mean'] = hourly_detailed['mean'].fillna(0)
                    hourly_detailed['std'] = hourly_detailed['std'].fillna(0)
                    hourly_detailed['min'] = hourly_detailed['min'].fillna(0)
                    hourly_detailed['max'] = hourly_detailed['max'].fillna(0)
                    hourly_detailed['q25'] = hourly_detailed['q25'].fillna(0)
                    hourly_detailed['q75'] = hourly_detailed['q75'].fillna(0)
                    
                    # Calcular CV e amplitude
                    hourly_detailed['cv'] = np.where(
                        hourly_detailed['mean'] > 0, 
                        (hourly_detailed['std'] / hourly_detailed['mean'] * 100).round(1),
                        0
                    )
                    hourly_detailed['range'] = hourly_detailed['max'] - hourly_detailed['min']
                    
                    # Renomear colunas para melhor visualização
                    hourly_detailed.columns = ['Hora', 'Registros', 'Média', 'Desvio Padrão', 'Mínimo', 'Máximo', 'Q1', 'Q3', 'CV (%)', 'Amplitude']
                    
                    st.dataframe(hourly_detailed, use_container_width=True)
                
                    # Identificar horas com maior variabilidade
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("⚠️ Horas com Maior Variabilidade")
                        # Filtrar apenas horas com dados (registros > 0)
                        hours_with_data = hourly_detailed[hourly_detailed['Registros'] > 0]
                        high_variability = hours_with_data[hours_with_data['CV (%)'] > 30].sort_values('CV (%)', ascending=False)
                        if not high_variability.empty:
                            st.dataframe(high_variability[['Hora', 'Média', 'CV (%)', 'Amplitude']], use_container_width=True)
                        else:
                            st.success("✅ Nenhuma hora com variabilidade excessiva (CV > 30%)")
                    
                    with col2:
                        st.subheader("📈 Horas com Maior Tráfego")
                        # Filtrar apenas horas com dados
                        hours_with_data = hourly_detailed[hourly_detailed['Registros'] > 0]
                        top_hours = hours_with_data.nlargest(3, 'Média')[['Hora', 'Média', 'Máximo', 'CV (%)']]
                        st.dataframe(top_hours, use_container_width=True)
                    
                    # Mostrar horas sem dados
                    hours_without_data = hourly_detailed[hourly_detailed['Registros'] == 0]
                    if not hours_without_data.empty:
                        st.subheader("❌ Horas Sem Dados")
                        st.warning(f"⚠️ {len(hours_without_data)} horas não possuem dados: {', '.join([f'{h}h' for h in hours_without_data['Hora']])}")
                        st.dataframe(hours_without_data[['Hora']], use_container_width=True)
                    
            with tab3:
                st.subheader("🚨 Detalhes das Anomalias")
                
                # Resumo das anomalias
                st.plotly_chart(
                    create_anomaly_summary_chart(anomalies),
                    use_container_width=True
                )
                
                # Detalhes específicos
                col1, col2 = st.columns(2)
                
                with col1:
                    if not anomalies['zero_views'].empty:
                        st.subheader("❌ Valores Zero")
                        st.dataframe(
                            anomalies['zero_views'][['Date', 'Hour', 'Views']].style.format({
                                'Date': lambda x: x.strftime('%d/%m/%Y'),
                                'Views': '{:,.0f}'
                            }),
                            use_container_width=True
                        )
                    
                    if not anomalies['very_low_views'].empty:
                        st.subheader("⚠️ Valores Muito Baixos")
                        st.dataframe(
                            anomalies['very_low_views'][['Date', 'Hour', 'Views']].head(10).style.format({
                                'Date': lambda x: x.strftime('%d/%m/%Y'),
                                'Views': '{:,.0f}'
                            }),
                            use_container_width=True
                        )
                
                with col2:
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
                            use_container_width=True
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
                        use_container_width=True
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
                        use_container_width=True
                    )
            
            with tab4:
                st.subheader("📋 Relatório Completo")
                
                # Gerar relatório
                report_text = generate_report_text(df, stats, anomalies)
                st.text_area("Relatório", report_text, height=400)
                
                # Botão de download
                st.download_button(
                    label="📥 Baixar Relatório",
                    data=report_text,
                    file_name=f"anomaly_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
                
                # Estatísticas detalhadas
                st.subheader("📊 Estatísticas Detalhadas")
                stats_df = pd.DataFrame([
                    {'Métrica': 'Total de Pageviews', 'Valor': f"{stats['total']:,}"},
                    {'Métrica': 'Média', 'Valor': f"{stats['mean']:.1f}"},
                    {'Métrica': 'Mediana', 'Valor': f"{stats['median']:.1f}"},
                    {'Métrica': 'Desvio Padrão', 'Valor': f"{stats['std']:.1f}"},
                    {'Métrica': 'Mínimo', 'Valor': f"{stats['min']:,}"},
                    {'Métrica': 'Máximo', 'Valor': f"{stats['max']:,}"},
                    {'Métrica': 'Q1 (25%)', 'Valor': f"{stats['q1']:.1f}"},
                    {'Métrica': 'Q3 (75%)', 'Valor': f"{stats['q3']:.1f}"},
                    {'Métrica': 'IQR', 'Valor': f"{stats['iqr']:.1f}"},
                    {'Métrica': 'Coeficiente de Variação', 'Valor': f"{(stats['std']/stats['mean']*100):.1f}%"}
                ])
                st.dataframe(stats_df, use_container_width=True)
            
            with tab5:
                st.subheader("🔄 Comparação entre Marcas")
                
                # Seleção de marcas para comparação
                st.subheader("🎯 Seleção de Marcas")
                available_brands = list(datasets.keys())
                col1 = st.columns(1)[0]
                with col1:
                    selected_brands_for_comparison = st.multiselect(
                        "Selecione as marcas para comparar:",
                        options=available_brands,
                        default=available_brands[:3] if len(available_brands) >= 3 else available_brands,
                        help="Escolha quais marcas deseja comparar"
                    )
                
                if selected_brands_for_comparison:
                    # Calcular estatísticas apenas para marcas selecionadas
                    selected_datasets = {brand: datasets[brand] for brand in selected_brands_for_comparison}
                    all_stats = {}
                    for brand_name, brand_df in selected_datasets.items():
                        all_stats[brand_name] = calculate_statistics(brand_df)
                    
                    # Gráfico de comparação
                    st.subheader("📈 Evolução Temporal Comparativa")
                    st.plotly_chart(
                        create_comparison_chart(selected_datasets, all_stats, selected_brands_for_comparison),
                        use_container_width=True,
                        key="comparison_chart"
                    )
                
                    # Métricas comparativas
                    st.subheader("📊 Métricas Comparativas")
                    comparison_df = create_comparison_metrics(selected_datasets, all_stats)
                    
                    # Mostrar apenas colunas formatadas na tabela
                    display_df = comparison_df[['Marca', 'Total Pageviews Formatado', 'Média por Hora Formatada', 'Mediana', 'Desvio Padrão', 'CV (%) Formatado', 'Máximo', 'Mínimo']]
                    display_df.columns = ['Marca', 'Total Pageviews', 'Média por Hora', 'Mediana', 'Desvio Padrão', 'CV (%)', 'Máximo', 'Mínimo']
                    st.dataframe(display_df, use_container_width=True)
                
                    # Ranking de performance
                    st.subheader("🏆 Ranking de Performance")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**📈 Maior Volume Total**")
                        top_total = comparison_df.nlargest(1, 'Total Pageviews')[['Marca', 'Total Pageviews Formatado']]
                        top_total.columns = ['Marca', 'Total Pageviews']
                        st.dataframe(top_total, use_container_width=True)
                    
                    with col2:
                        st.markdown("**📊 Maior Média por Hora**")
                        top_avg = comparison_df.nlargest(1, 'Média por Hora')[['Marca', 'Média por Hora Formatada']]
                        top_avg.columns = ['Marca', 'Média por Hora']
                        st.dataframe(top_avg, use_container_width=True)
                    
                    with col3:
                        st.markdown("**🎯 Menor Variabilidade (CV)**")
                        top_stable = comparison_df.nsmallest(1, 'CV (%)')[['Marca', 'CV (%) Formatado']]
                        top_stable.columns = ['Marca', 'CV (%)']
                        st.dataframe(top_stable, use_container_width=True)
                
                    # Análise de anomalias comparativa
                    st.subheader("🚨 Anomalias por Marca")
                    
                    anomaly_comparison = []
                    for brand_name, brand_df in selected_datasets.items():
                        brand_stats = all_stats[brand_name]
                        brand_anomalies = detect_anomalies(brand_df, brand_stats, low_threshold, extreme_threshold, start_hour, end_hour)
                        
                        total_anomalies = (len(brand_anomalies['zero_views']) + 
                                         len(brand_anomalies['very_low_views']) + 
                                         len(brand_anomalies['statistical_outliers']) + 
                                         len(brand_anomalies['extreme_variations']))
                        
                        anomaly_comparison.append({
                            'Marca': brand_name,
                            'Total Anomalias': total_anomalies,
                            'Valores Zero': len(brand_anomalies['zero_views']),
                            'Valores Baixos': len(brand_anomalies['very_low_views']),
                            'Outliers': len(brand_anomalies['statistical_outliers']),
                            'Variações Extremas': len(brand_anomalies['extreme_variations']),
                            'Horas Faltando': len(brand_anomalies['missing_hours'])
                        })
                    
                    anomaly_df = pd.DataFrame(anomaly_comparison)
                    st.dataframe(anomaly_df, use_container_width=True)
                    
                    # Resumo executivo
                    st.subheader("📋 Resumo Executivo")
                    
                    best_performer = comparison_df.loc[comparison_df['Total Pageviews'].idxmax(), 'Marca']
                    most_stable = comparison_df.loc[comparison_df['CV (%)'].idxmin(), 'Marca']
                    least_anomalies = anomaly_df.loc[anomaly_df['Total Anomalias'].idxmin(), 'Marca']
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.success(f"🏆 **Melhor Performance:** {best_performer}")
                    
                    with col2:
                        st.info(f"🎯 **Mais Estável:** {most_stable}")
                    
                    with col3:
                        st.warning(f"⚠️ **Menos Anomalias:** {least_anomalies}")
                else:
                    st.info("👆 Selecione pelo menos uma marca para comparar")
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
