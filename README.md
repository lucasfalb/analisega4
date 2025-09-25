# 📊 Dashboard de Análise de Anomalias - Pageviews

Um dashboard interativo em Python para detectar e analisar anomalias em dados de pageviews de sites.

## 🚀 Funcionalidades

- **Upload de CSV**: Faça upload de arquivos CSV com dados de pageviews
- **Detecção Automática de Anomalias**: 
  - Valores zero ou muito baixos
  - Outliers estatísticos
  - Variações extremas da média
  - Horas faltando nos dados
- **Visualizações Interativas**: Gráficos dinâmicos com Plotly
- **Relatórios Exportáveis**: Gere relatórios em texto para download
- **Configuração Flexível**: Ajuste parâmetros de detecção conforme necessário

## 📋 Requisitos

- Python 3.8+
- Bibliotecas listadas em `requirements.txt`

## 🛠️ Instalação

1. Clone ou baixe os arquivos
2. Instale as dependências:
```bash
pip install -r requirements.txt
```

## 🎯 Como Usar

1. Execute o dashboard:
```bash
streamlit run anomaly_dashboard.py
```

2. Acesse o dashboard no navegador (geralmente `http://localhost:8501`)

3. Faça upload de um arquivo CSV com uma das seguintes estruturas:

**Formato 1 - Com hora separada:**
   - `Date`: Data no formato "Jul 1, 2025"
   - `Hour`: Hora do dia (0-23)
   - `Views`: Número de pageviews

**Formato 2 - Data e hora combinadas:**
   - `Date + hour (YYYYMMDDHH)`: Data e hora no formato "Jul 1, 2025, 12AM"
   - `Views`: Número de pageviews

4. Ajuste os parâmetros de detecção no menu lateral

5. Explore os resultados nas diferentes abas

## 📊 Exemplos de CSV

**Formato 1:**
```csv
Date,Hour,Views
"Jul 1, 2025",13,1637
"Jul 1, 2025",15,1622
"Jul 1, 2025",16,1604
"Jul 1, 2025",21,1453
"Jul 1, 2025",14,1436
```

**Formato 2:**
```csv
Date + hour (YYYYMMDDHH),Views
"Jul 1, 2025, 12AM",643
"Jul 1, 2025, 1AM",474
"Jul 1, 2025, 2AM",218
"Jul 1, 2025, 3AM",129
"Jul 1, 2025, 4AM",163
```

## 🔍 Tipos de Anomalias Detectadas

### Valores Zero
- Registros com 0 pageviews
- Indica possível instabilidade total do site

### Valores Muito Baixos
- Abaixo do limite configurável (padrão: 50)
- Pode indicar problemas técnicos ou queda de tráfego

### Outliers Estatísticos
- Valores fora do intervalo IQR × 1.5
- Detecta picos ou quedas anômalas

### Variações Extremas
- Diferenças superiores a 80% da média
- Identifica mudanças drásticas no padrão

### Horas Faltando
- Períodos sem dados (12h-23h esperadas)
- Indica problemas no sistema de coleta

## 📈 Visualizações Disponíveis

1. **Série Temporal**: Linha do tempo com anomalias destacadas
2. **Distribuição**: Histograma dos valores de pageviews
3. **Análise por Hora**: Média de pageviews por hora do dia
4. **Resumo de Anomalias**: Gráfico de barras com contagem por tipo
5. **Relatório Completo**: Texto detalhado com estatísticas

## ⚙️ Configurações

- **Limite para valores baixos**: Define o que é considerado "muito baixo"
- **Limite para variações extremas**: Percentual de variação da média considerado extremo

## 📁 Arquivos

- `anomaly_dashboard.py`: Dashboard principal
- `requirements.txt`: Dependências Python
- `simple_anomaly_analysis.py`: Script de análise simples (sem interface)
- `README.md`: Este arquivo

## 🎨 Interface

O dashboard possui uma interface moderna e responsiva com:
- Menu lateral para configurações
- Métricas principais em cards
- Tabs organizadas para diferentes visualizações
- Alertas visuais para anomalias críticas
- Botões de download para relatórios

## 🔧 Personalização

Você pode facilmente modificar:
- Limites de detecção de anomalias
- Cores e estilos dos gráficos
- Tipos de anomalias detectadas
- Formato dos relatórios

## 📞 Suporte

Para dúvidas ou problemas, verifique:
1. Formato do arquivo CSV
2. Colunas obrigatórias presentes
3. Dependências instaladas corretamente
4. Versão do Python compatível

