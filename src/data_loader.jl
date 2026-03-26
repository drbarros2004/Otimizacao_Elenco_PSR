using CSV, DataFrames, Dates

# --- CONFIGURAÇÃO DE CAMINHOS ---
const PATH_BASE = "data/raw/player_stats.csv"
const PATH_CUSTOM = "data/raw/stats_brasileirao_with_correct_teams.csv"
const OUTPUT_PATH = "data/processed/processed_player_data.csv"

# Mapeamento exato das colunas do seu Scraper
const COLUNAS_SCRAPER = [
    :player_id, :name, :dob, :positions, :overall_rating, 
    :potential, :value, :wage, :international_reputation,
    :club_name, :club_league_name
]

function calcular_idade(dob_valor)
    # Se for missing, retorna o padrão
    if ismissing(dob_valor) return 25 end
    
    try
        # 1. Se o CSV.read já converteu para Date automaticamente
        nascimento = if dob_valor isa Date
            dob_valor
        else
            # 2. Se for String, tenta o formato ISO (yyyy-mm-dd)
            # Nota: Julia usa 'yyyy-mm-dd' para Date.
            Date(strip(string(dob_valor))) 
        end
        
        hoje = Date(2026, 03, 26)
        return year(hoje) - year(nascimento) - (monthday(hoje) < monthday(nascimento) ? 1 : 0)
    catch e
        # Se falhar, pelo menos nos avise o porquê (só na primeira vez)
        @warn "Falha ao calcular idade para o valor: $dob_valor. Erro: $e"
        return 25
    end
end

function carregar_e_limpar_dados()
    println("📂 Iniciando leitura com colunas do SoFIFA...")
    
    # 1. Carregar as tabelas
    df_base = CSV.read(PATH_BASE, DataFrame)
    df_custom = CSV.read(PATH_CUSTOM, DataFrame)
    
    # 2. Unir as tabelas
    df_total = vcat(df_base, df_custom, cols=:union)
    
    # 3. Remover duplicatas por player_id
    unique!(df_total, :player_id)
    
    # 4. Selecionar e Renomear colunas para facilitar o modelo
    df_limpo = df_total[:, intersect(names(df_total), String.(COLUNAS_SCRAPER))]
    
    # 5. Criar a coluna 'Age' (essencial para a evolução Jovens/Veteranos)
    # Baseado na sua metodologia de curvas logarítmicas/lineares
    df_limpo.age = [calcular_idade(d) for d in df_limpo.dob]
    
    # 6. Tratamento de Nulos
    df_limpo.value = coalesce.(df_limpo.value, 0.0)
    df_limpo.international_reputation = coalesce.(df_limpo.international_reputation, 1.0)
    
    # 7. Salvar o resultado
    mkpath("data/processed")
    CSV.write(OUTPUT_PATH, df_limpo)
    
    println("✅ Sucesso! Total de jogadores: $(nrow(df_limpo))")
    println("📌 Coluna 'age' gerada para o cálculo de evolução futuro.")
    
    return df_limpo
end

# Executa
carregar_e_limpar_dados()