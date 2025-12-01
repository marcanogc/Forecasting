import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from datetime import datetime
from typing import List, Tuple
from pathlib import Path
import os

# tentativa de carregar joblib a partir do sklearn ou do pacote joblib
try:
    from sklearn.externals import joblib  # type: ignore
except Exception:
    try:
        import joblib  # type: ignore
    except Exception:
        joblib = None

# Obter o diretório base do projeto (pasta pai do diretório app)
BASE_DIR = Path(__file__).resolve().parent.parent

# App de simulação - Forecast Novembro 2025
st.set_page_config(page_title="Simulação Vendas — Novembro 2025", layout="wide",
                   initial_sidebar_state="expanded")

# Paleta
PALETTE_PRIMARY = "#667eea"
PALETTE_SECONDARY = "#764ba2"
sns.set_theme(style="whitegrid", palette=[PALETTE_PRIMARY, PALETTE_SECONDARY])

# Helpers
def format_eur(x):
    return f"€{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def format_units(x):
    return f"{int(round(x)):,}".replace(",", ".")

def safe_load_model(path: str):
    if joblib is None:
        raise ImportError("Não foi possível importar joblib (necessário para carregar o modelo).")
    return joblib.load(path)

def ensure_datetime(df: pd.DataFrame, col="fecha"):
    df = df.copy()
    if not np.issubdtype(df[col].dtype, np.datetime64):
        df[col] = pd.to_datetime(df[col])
    return df

def run_recursive_prediction(df_prod: pd.DataFrame, model, feature_names: List[str]) -> pd.DataFrame:
    """
    Recebe df_prod já filtrado e ordenado por fecha.
    Realiza predições recursivas dia a dia usando as colunas:
    unidades_vendidas_lag1 ... unidades_vendidas_lag7 e unidades_vendidas_mm7
    """
    df_work = df_prod.sort_values("fecha").reset_index(drop=True).copy()
    n_rows = df_work.shape[0]

    # validar colunas de lag e mm7
    required_lags = [f"unidades_vendidas_lag{i}" for i in range(1, 8)]
    for c in required_lags + ["unidades_vendidas_mm7"]:
        if c not in df_work.columns:
            raise KeyError(f"Coluna necessária ausente: {c}")

    preds = []
    ingresos = []

    # inicializar lags atuais a partir da primeira linha (dia 1)
    current_lags = [float(df_work.loc[0, f"unidades_vendidas_lag{i}"]) for i in range(1, 8)]
    current_mm7 = float(df_work.loc[0, "unidades_vendidas_mm7"])

    # iterar dias
    for i in range(n_rows):
        row = df_work.loc[i].copy()

        # injetar lags e mm7 atuais no row (no dia 1 já são os do arquivo)
        for j, c in enumerate(required_lags, start=1):
            row[c] = current_lags[j - 1]
        row["unidades_vendidas_mm7"] = current_mm7

        # preparar features na ordem do modelo
        try:
            X_row = row[feature_names].astype(float).values.reshape(1, -1)
        except Exception as e:
            raise RuntimeError(f"Erro ao montar features para o modelo: {e}")

        # predizer
        y_pred = model.predict(X_row)[0]
        if np.isnan(y_pred):
            y_pred = 0.0
        y_pred = float(max(0.0, y_pred))  # não aceitar negativo

        # calcular ingresos com precio_venta presente no row
        precio_venta = float(row.get("precio_venta", 0.0))
        ingreso = y_pred * precio_venta

        preds.append(y_pred)
        ingresos.append(ingreso)

        # atualizar lags para o próximo dia:
        # novo conjunto = [pred, lag1_antigo, lag2_antigo, ..., lag6_antigo]
        current_lags = [y_pred] + current_lags[:6]
        # atualizar mm7 = média das últimas 7 (current_lags)
        current_mm7 = float(np.mean(current_lags))

    df_out = df_work.copy()
    df_out["pred_unidades"] = np.round(preds, 0)
    df_out["pred_ingresos"] = ingresos
    # garantir tipos
    df_out["pred_unidades"] = df_out["pred_unidades"].astype(float)
    df_out["pred_ingresos"] = df_out["pred_ingresos"].astype(float)
    return df_out

# Sidebar - Controles
with st.sidebar:
    st.markdown("<h3 style='color: #764ba2;'>Controles de Simulação</h3>", unsafe_allow_html=True)
    st.write("")
    # Carregar dados e modelo (mensagens iniciais se falharão no botão)
    # Seletores posteriores dependem do arquivo de inferência
    st.markdown("---")
    st.info("Selecione o produto, ajuste o desconto e escolha o cenário de concorrência. Depois clique em 'Simular Vendas'.")

# Carregar dados e modelo com tratamento de erro
DATA_PATH = BASE_DIR / "data" / "processed" / "inferencia_df_transformado.csv"
MODEL_PATH = BASE_DIR / "models" / "modelo_final.joblib"

try:
    df_all = pd.read_csv(DATA_PATH)
    df_all = ensure_datetime(df_all, "fecha")
except Exception as e:
    st.error(f"Erro ao carregar dados de inferência: {e}")
    st.stop()

try:
    model = safe_load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Erro ao carregar o modelo: {e}")
    st.stop()

# obter nomes dos produtos
if "nombre" not in df_all.columns:
    st.error("Coluna 'nombre' não encontrada no arquivo de inferência.")
    st.stop()

product_names = sorted(df_all["nombre"].dropna().unique().tolist())

# Sidebar controls (continuação para widgets dependentes dos dados)
with st.sidebar:
    selected_product = st.selectbox("Produto", product_names)
    desconto_slider = st.slider("Ajuste de desconto (%)", -50, 50, 0, step=5)
    scenario = st.radio("Cenário de concorrência", ("Atual (0%)", "Concorrência -5%", "Concorrência +5%"))
    st.markdown("")
    simulate_btn = st.button("Simular Vendas")

# Mapear cenário para ajuste multiplicativo
scenario_map = {"Atual (0%)": 0.0, "Concorrência -5%": -0.05, "Concorrência +5%": 0.05}
scenario_adj = scenario_map.get(scenario, 0.0)

# Função para preparar DF do produto com os ajustes fixos (desconto e competição)
def prepare_product_df(df_all: pd.DataFrame, product_name: str,
                       discount_pct: float, comp_adj: float) -> pd.DataFrame:
    df_p = df_all[df_all["nombre"] == product_name].copy()
    if df_p.empty:
        raise ValueError("Não foram encontrados registros para o produto selecionado.")
    df_p = df_p.sort_values("fecha").reset_index(drop=True)

    # recalcular precio_venta a partir de precio_base e desconto
    # desconto_pct recebido como percentagem (ex. 10 => 10%)
    desc_frac = discount_pct / 100.0
    df_p["descuento_porcentaje"] = desc_frac
    df_p["precio_venta"] = df_p["precio_base"] * (1 - desc_frac)

    # ajustar preços de concorrência se colunas existirem (Amazon, Decathlon, Deporvillage)
    comp_cols = [c for c in ["Amazon", "Decathlon", "Deporvillage"] if c in df_p.columns]
    if comp_cols:
        for c in comp_cols:
            df_p[c] = df_p[c] * (1 + comp_adj)
        # recalcular precio_competencia como média das plataformas se existir
        df_p["precio_competencia"] = df_p[comp_cols].mean(axis=1)
    else:
        # se não houver colunas individuais, apenas aplicar ajuste relativo ao precio_competencia
        if "precio_competencia" in df_p.columns:
            df_p["precio_competencia"] = df_p["precio_competencia"] * (1 + comp_adj)

    # recalcular ratio_precio
    if "precio_competencia" in df_p.columns:
        # evitar divisão por zero
        df_p["ratio_precio"] = df_p["precio_venta"] / df_p["precio_competencia"].replace(0, np.nan)
        df_p["ratio_precio"] = df_p["ratio_precio"].fillna(1.0)
    else:
        # criar coluna com 1.0 caso não exista
        df_p["precio_competencia"] = df_p.get("precio_competencia", np.nan)
        df_p["ratio_precio"] = df_p.get("ratio_precio", 1.0)

    return df_p

# função auxiliar que executa simulação completa e retorna df com predições e KPIs
def simulate_for_product(df_all, product_name, discount_pct, comp_adj, model) -> Tuple[pd.DataFrame, dict]:
    df_p = prepare_product_df(df_all, product_name, discount_pct, comp_adj)

    # obter feature names do modelo
    if not hasattr(model, "feature_names_in_"):
        raise AttributeError("O modelo não expõe 'feature_names_in_'. Não é possível garantir as colunas de entrada.")
    feature_names = list(model.feature_names_in_)

    # validar que todas as features existam no df_p (exceto quando feature não estiver presente)
    missing = [f for f in feature_names if f not in df_p.columns]
    if missing:
        raise KeyError(f"Colunas de feature ausentes no dataframe: {missing}")

    # executar predições recursivas com spinner
    with st.spinner("Calculando previsões recursivas..."):
        df_pred = run_recursive_prediction(df_p, model, feature_names)

    # calcular KPIs
    total_unidades = float(df_pred["pred_unidades"].sum())
    total_ingresos = float(df_pred["pred_ingresos"].sum())
    precio_promedio = float(df_pred["precio_venta"].mean())
    descuento_promedio = float(df_pred["descuento_porcentaje"].mean())

    kpis = {
        "total_unidades": total_unidades,
        "total_ingresos": total_ingresos,
        "precio_promedio": precio_promedio,
        "descuento_promedio": descuento_promedio
    }
    return df_pred, kpis

# Executar simulação principal quando o botão for clicado
if simulate_btn:
    try:
        # simulação principal (cenário escolhido)
        df_result, kpis_main = simulate_for_product(df_all, selected_product, desconto_slider, scenario_adj, model)

        # também gerar comparativa para os 3 cenários mantendo o desconto do usuário
        scenario_list = [("Atual (0%)", 0.0), ("Concorrência -5%", -0.05), ("Concorrência +5%", 0.05)]
        comparativo = []
        for name, adj in scenario_list:
            _, k = simulate_for_product(df_all, selected_product, desconto_slider, adj, model)
            comparativo.append((name, k["total_unidades"], k["total_ingresos"]))

        # --- Zona principal do dashboard ---
        st.markdown(f"<h1 style='color:{PALETTE_SECONDARY};'>Simulação Novembro 2025 — {selected_product}</h1>", unsafe_allow_html=True)
        st.markdown("---")

        # KPIs em linha
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Unidades Totais", format_units(kpis_main["total_unidades"]))
        col2.metric("Receita Projetada", format_eur(kpis_main["total_ingresos"]))
        col3.metric("Preço Médio", format_eur(kpis_main["precio_promedio"]))
        col4.metric("Desconto Médio", f"{kpis_main['descuento_promedio']*100:.0f}%")

        st.markdown("---")

        # Gráfico de previsão diária
        st.subheader("Previsão diária (1–30 de Novembro)")
        df_plot = df_result.copy()
        df_plot["dia"] = df_plot["fecha"].dt.day
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.lineplot(data=df_plot, x="dia", y="pred_unidades", marker="o", ax=ax, color=PALETTE_PRIMARY)
        ax.set_xlabel("Dia de Novembro")
        ax.set_ylabel("Unidades Vendidas (prev.)")
        ax.set_xticks(range(1, 31))
        # marcar Black Friday: assumido 28 de novembro
        bf_day = 28
        if bf_day in df_plot["dia"].values:
            bf_val = float(df_plot.loc[df_plot["dia"] == bf_day, "pred_unidades"].values[0])
            ax.axvline(bf_day, color="grey", linestyle="--", alpha=0.6)
            ax.scatter([bf_day], [bf_val], color="red", zorder=5, s=80)
            ax.annotate("🛍️ Black Friday", xy=(bf_day, bf_val),
                        xytext=(bf_day+1, bf_val + max(1.0, bf_val*0.05)),
                        color="red")
        st.pyplot(fig)

        st.markdown("---")

        # Tabela detalhada
        st.subheader("Detalhamento diário")
        df_table = df_result[["fecha", "nombre_dia_semana", "precio_venta", "precio_competencia",
                              "descuento_porcentaje", "pred_unidades", "pred_ingresos"]].copy()
        df_table["fecha"] = df_table["fecha"].dt.date
        df_table.rename(columns={
            "nombre_dia_semana": "dia_semana",
            "precio_venta": "preço_venda (€)",
            "precio_competencia": "preço_concorrência (€)",
            "descuento_porcentaje": "desconto (%)",
            "pred_unidades": "unidades_prev",
            "pred_ingresos": "receita_prev (€)"
        }, inplace=True)
        # formatar colunas
        df_table["preço_venda (€)"] = df_table["preço_venda (€)"].apply(lambda x: format_eur(x))
        df_table["preço_concorrência (€)"] = df_table["preço_concorrência (€)"].apply(lambda x: format_eur(x) if pd.notna(x) else "")
        df_table["desconto (%)"] = (df_table["desconto (%)"] * 100).round(0).astype(int).astype(str) + "%"
        df_table["unidades_prev"] = df_table["unidades_prev"].round(0).astype(int)
        df_table["receita_prev (€)"] = df_table["receita_prev (€)"].apply(lambda x: format_eur(x))

        # adicionar destaque para Black Friday
        df_table["evento"] = ""
        bf_mask = pd.to_datetime(df_result["fecha"]).dt.day == 28
        if bf_mask.any():
            idx = df_table[bf_mask.values].index
            if len(idx) > 0:
                df_table.loc[idx[0], "evento"] = "🛍️ Black Friday"

        st.dataframe(df_table.style.set_properties(**{"text-align": "left"}), use_container_width=True)

        st.markdown("---")

        # Comparativa de cenários
        st.subheader("Comparação de cenários de concorrência")
        cols = st.columns(3)
        for (name, unidades, ingresos), c in zip(comparativo, cols):
            c.markdown(f"**{name}**")
            c.metric("Unidades Totais", format_units(unidades))
            c.metric("Receita", format_eur(ingresos))

        st.success("Simulação concluída ✅")
    except Exception as e:
        st.error(f"Erro durante a simulação: {e}")
else:
    st.markdown("<h2 style='color: #667eea;'>Pronto para simular</h2>", unsafe_allow_html=True)
    st.info("Escolha o produto e pressione 'Simular Vendas' para iniciar a previsão recursiva.")