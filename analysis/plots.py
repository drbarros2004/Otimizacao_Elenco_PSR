# analysis/plots.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configurações de Estilo Globais
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 150

def load_player_window_audit(path="../data/processed/player_window_audit.csv"):
    """Carrega os dados e valida a existência do arquivo."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Arquivo {path} não encontrado. Verifique se o pipeline em Julia foi executado.")
    return pd.read_csv(path)

def plot_ovr_evolution(df, targets, output_dir="viz/"):
    """Gera o gráfico de evolução técnica por categoria."""
    print("📈 Gerando gráfico de evolução de OVR...")
    
    # Filtra apenas os jogadores que REALMENTE existem no CSV
    available_players = df['name'].unique()
    valid_targets = {k: v for k, v in targets.items() if k in available_players}
    
    if not valid_targets:
        print(f"⚠️ Nenhum dos jogadores alvo {list(targets.keys())} foi encontrado no CSV.")
        print(f"Dica: Os primeiros nomes no CSV são: {available_players[:5]}")
        return

    df_plot = df[df['name'].isin(valid_targets.keys())].copy()
    df_plot['Categoria'] = df_plot['name'].map(valid_targets)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    palette = {"Jovem": "#2ecc71", "Auge": "#f1c40f", "Veterano": "#e74c3c"}

    sns.lineplot(
        data=df_plot, x="window", y="ovr", hue="Categoria", style="Categoria",
        markers=True, dashes=False, linewidth=3, markersize=12, palette=palette, ax=ax
    )

    # Anotações apenas para os jogadores encontrados
    for nome, cat in valid_targets.items():
        player_data = df_plot[df_plot['name'] == nome].sort_values('window')
        
        if player_data.empty:
            continue
            
        for x, y in zip(player_data['window'], player_data['ovr']):
            ax.text(x, y + 0.5, f"{int(y)}", ha='center', weight='bold')
        
        # Aqui era onde dava o erro! Agora está seguro.
        last = player_data.iloc[-1]
        ax.text(last['window']+0.1, last['ovr'], f"{nome}\n({cat})", va='center', fontweight='bold')

    ax.set_title("Metodologia de Evolução: Dinâmica por Faixa Etária", fontsize=18, pad=20, fontweight='bold')
    ax.set_ylabel("Overall Rating (OVR)", fontsize=14)
    ax.set_xlabel("Janelas de Transferência", fontsize=14)
    ax.set_xticks(df_plot['window'].unique())

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "evolution_plot.png"))
    plt.close()

def plot_cost_comparison(df, target_players, output_dir="viz/"):
    print("💰 Gerando gráfico comparativo com breakdown de custos...")
    
    df_now = df[(df['window'] == 0) & (df['name'].isin(target_players))].copy()
    if df_now.empty:
        return

    df_melted = df_now.melt(
        id_vars=['name', 'league', 'league_mult', 'ir_bonus', 'rep_multiplier'],
        value_vars=['market_value_eur', 'acquisition_cost_eur'],
        var_name='Tipo', value_name='Valor'
    )

    plt.figure(figsize=(13, 8))
    ax = sns.barplot(
        data=df_melted, x="name", y="Valor", hue="Tipo",
        order=target_players,
        palette=["#3498db", "#e74c3c"]
    )

    # 1. Labels de Valor (Milhões)
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'€{height/1e6:.1f}M', 
                (p.get_x() + p.get_width() / 2., height), 
                ha='center', va='bottom', fontsize=10, fontweight='bold', xytext=(0, 5),
                textcoords='offset points')

    # 2. Caixa de Texto com Decomposição (O "Pulo do Gato")
    for i, name in enumerate(target_players):
        p_info = df_now[df_now['name'] == name].iloc[0]
        
        # Montamos a string de explicação
        label_text = (
            f"Total: {p_info['rep_multiplier']}x\n"
            f"───────────\n"
            f"Liga: {p_info['league_mult']}x\n"
            f"Fama (IR): +{int(p_info['ir_bonus']*100)}%\n"
            f"({p_info['league']})"
        )
        
        # Posicionamos a caixa levemente acima da base para não sumir
        plt.text(i, ax.get_ylim()[1] * 0.05, label_text, 
                 ha='center', fontsize=9, fontweight='bold',
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='#bdc3c7', boxstyle='round,pad=0.5'))

    plt.title("Análise de Custo: Valor de Vitrine vs. Realidade de Mercado", fontsize=18, fontweight='bold', pad=25)
    plt.ylabel("Valor em Euros (€)", fontsize=13)
    plt.xlabel("Jogador Selecionado", fontsize=13)
    plt.legend(title="Legenda", loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cost_breakdown_plot.png"))
    plt.close()

def main():
    try:
        df = load_player_window_audit()
        
        # DICA: Verifique se esses nomes estão EXATAMENTE assim no seu CSV
        # Se você fez scraping novo, talvez os nomes tenham mudado ligeiramente.
        alvos = {
            "Breno de Souza Bidon": "Jovem",
            "Samuel Dias Lino": "Auge",
            "Jorge Luiz Frello Filho": "Veterano"
        }

        # Reaproveita os mesmos jogadores definidos em `alvos`
        alvos_custo = list(alvos.keys())
        
        plot_ovr_evolution(df, alvos)
        plot_cost_comparison(df, alvos_custo)
        
        print("✅ Processo concluído.")
        
    except Exception:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()