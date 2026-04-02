# ⚽ Otimização Multi-Período de Elenco

[![Julia](https://img.shields.io/badge/Julia-1.9+-9558B2?style=flat&logo=julia)](https://julialang.org/)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit)](https://streamlit.io/)

Projeto de pesquisa e aplicação prática em **Pesquisa Operacional** para o planejamento de elencos de futebol moderno. O projeto utiliza Programação Linear Inteira Mista (MILP) para otimizar as decisões de compra, venda e escalação de jogadores ao longo de múltiplas janelas de transferência, respeitando regras financeiras rigorosas.

O modelo central é desenvolvido em **Julia (JuMP + Gurobi)** para máxima performance de otimização, acompanhado por um dashboard interativo em **Python (Streamlit)** para análise de cenários e visualização do elenco.

---

## 📋 Sumário
1. [Visão Geral](#1-visão-geral)
2. [Arquitetura e Fluxo de Dados](#2-arquitetura-e-fluxo-de-dados)
3. [Modos de Otimização](#3-modos-de-otimização)
4. [A Formulação Matemática](#4-a-formulação-matemática)
5. [Requisitos e Instalação](#5-requisitos-e-instalação)
6. [Como Executar](#6-como-executar)
7. [Configuração](#7-configuração)
8. [Saídas e Dashboards](#8-saídas-e-dashboards)

---

## 🎯 1. Visão Geral

O modelo atua como um "Diretor de Futebol Virtual", decidindo janela a janela:
- 🛒 **Quem contratar** (baseado em Overall, Potencial, Custo e Encaixe Tático).
- 💰 **Quem vender** (para gerar caixa ou adequar a folha salarial).
- 🏃 **Quem escalar** (formando os XI iniciais para maximizar a performance em campo).
- 📊 **Como gerenciar o orçamento** ao longo dos anos, mantendo a saúde financeira.

---

## 🏗️ 2. Arquitetura e Fluxo de Dados

A arquitetura do projeto é dividida em ingestão, processamento, otimização matemática e visualização.

```text
.
├── main.jl                    # Ponto de entrada p/ rodar a otimização
├── analyze_results.jl         # Script rápido para análise textual dos resultados
├── config/                    # Configurações gerais (experimentos e mercado)
│   ├── experiment.toml        
│   └── market_settings.toml   
├── data/                      # Dados (brutos e processados)
│   ├── raw/                   # CSVs com atributos dos jogadores (ex: FIFA/EA FC)
│   └── processed/             # Base tratada, idades corrigidas, posições
├── src/                       # Código-fonte Julia
│   ├── data_loader.jl         # Pipeline de leitura e projeção técnica/financeira
│   ├── model*.jl              # Formulações MILP (estocástica e determinística)
│   └── solver_engine.jl       # Instanciação do solver (Gurobi)
├── analysis/                  # Scripts e Dashboard Python
│   ├── streamlit_dashboard.py # Interface interativa
│   └── plots.py               
├── output/                    # CSVs gerados após a otimização
└── scraper/                   # Scripts auxiliares para coleta de dados
```

---

## 🧠 3. Modos de Otimização

O projeto suporta duas abordagens para resolver o problema estrutural do clube:

### 3.1 Modo Determinístico (Janela a Janela)
- Otimiza as ações assumindo que o **futuro é perfeitamente conhecido**.
- Variáveis de estado em janela $t$ (elenco, orçamento) dependem estritamente das decisões tomadas em $t-1$.
- Excelente para planejamento basal e cenários de "melhor caso" garantido.
- Arquivo principal: `src/model_deterministic.jl`

### 3.2 Modo Estocástico (Árvore de Cenários / Node-Indexed)
- Planejamento sob **incerteza**. Modelado em formato de árvore de decisão ponderada por probabilidades.
- Considera cenários paralelos: lesões de jogadores chave, choques no mercado, inflação de valores, mudanças forçadas de esquema tático.
- Busca o melhor plano de ação que maximize o valor esperado entre todas as ramificações de futuros possíveis.
- Arquivo principal: `src/model_stochastic.jl`

---

## 📐 4. A Formulação Matemática

O núcleo da otimização tenta **Maximizar** uma função objetivo composta por:
- **Performance Técnica:** Soma do *Overall Rating* do time titular.
- **Química/Entrosamento:** Bônus para combinações adequadas na escalação natural.
- **Valor Terminal:** Avaliação financeira do clube e do patrimônio do elenco na última janela.

**Sujeito a Restrições (Constraints):**
- Limite mínimo e máximo de jogadores no elenco.
- Restrições Táticas estritas (ex: `1 GK`, `2 CB`, `1 ST`, etc.) por janela.
- Conservação de fluxo (quem entra, deve estar disponível; quem sai, libera a vaga e o salário).
- Restrições orçamentárias rígidas de Fluxo de Caixa.
- Limites "Soft" de teto salarial (com variáveis de *slack* que aplicam punições na função objetivo).

---

## ⚙️ 5. Requisitos e Instalação

### Julia (Motor de Otimização)
- Julia **1.9+**
- Instale as dependências (CSV, DataFrames, JuMP, Gurobi, etc.) rodando:
  ```bash
  julia --project=. -e 'using Pkg; Pkg.instantiate()'
  ```

### Solver
- O projeto usa o **Gurobi** (licença acadêmica/comercial necessária).
- Para alternar para solvers open-source (como o *HiGHS* ou *GLPK*), é necessário editar o `solver_engine.jl`.

### Python (Dashboard Interativo)
Ideal para usufruir da UI após rodar a otimização.
```bash
pip install -r analysis/requirements-streamlit.txt
```

---

## 🚀 6. Como Executar

### 6.1 Rodar a Otimização Completa
O comando básico processa os dados, gera as projeções e resolve o MILP com o Gurobi baseando-se no `experiment.toml`.
```bash
julia --project=. main.jl
```

*(Opcional) Passando um arquivo de configuração customizado:*
```bash
julia --project=. main.jl --config config/experiment.toml
```

### 6.2 Avaliar Resultados Rapidamente (CLI)
```bash
julia --project=. analyze_results.jl
```
*(Use `--deterministic` ou `--stochastic` para forçar a análise do arquivo desejado).*

### 6.3 Lançar o Dashboard Visual (Streamlit)
Veja seu time no nível do gramado e entenda o orçamento visualmente:
```bash
streamlit run analysis/streamlit_dashboard.py
```

---

## 🔧 7. Configuração

Todas as regras de negócio podem ser alteradas sem mexer no código através de arquivos `.toml`.

- **`config/experiment.toml`**: 
  - `[simulation]`: Número de janelas e filtro de limite de jogadores no subconjunto de escolha.
  - `[optimization]`: Caixa inicial, orçamento periódico e pesos na função objetivo.
  - `[constraints]`: Limite numérico do elenco, atrito financeiro de negociação (taxas para agentes).
  - `[stochastic]`: Probabilidades, ramificações e eventos estocásticos para a simulação de incerteza.
  - `[formation_catalog.*]` / `[formation_plan]`: Definição de formações táticas disponíveis no tempo.

- **`config/market_settings.toml`**:
  - Calibrações e multiplicadores financeiros de mercado por liga (La Liga, Premier League, Brasileirão, etc).

---

## 📊 8. Saídas e Dashboards

Os arquivos gerados na pasta `/output/` servem como "tabela fato" das ações sugeridas:

- `squad_decisions.csv`: As decisões detalhadas (Venda/Compra/Fica) de todos os jogadores indexadas no tempo.
- `budget_evolution.csv`: Demonstração dos resultados, gastos, lucros e saldo atual a cada passo de tempo.
- `formation_diagnostics.csv`: Resumo de qualidade tática e OVR médio do XI titular em campo.
- Formatos `*_nodes.csv`: Arquivos equivalentes gerados pela resolução estocástica na árvore de cenários.

---

> 👨‍💻 **Autor**: Daniel Rebouças de Sousa Barros
> 📄 **Licença**: Projeto acadêmico/profissional (Uso Técnico e Educacional).
