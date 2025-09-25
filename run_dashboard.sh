#!/bin/bash

echo "🚀 Iniciando Dashboard de Análise de Anomalias..."
echo "📊 Acesse: http://localhost:8501"
echo ""

# Verificar se as dependências estão instaladas
if ! python3 -c "import streamlit, pandas, numpy, plotly" 2>/dev/null; then
    echo "📦 Instalando dependências..."
    pip3 install -r requirements.txt
fi

# Executar o dashboard (configurações estão em .streamlit/config.toml)
streamlit run anomaly_dashboard.py
