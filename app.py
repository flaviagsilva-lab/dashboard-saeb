
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import unicodedata
from pathlib import Path

st.set_page_config(
    page_title="Painel Saeb • Rede Municipal de Ensino de Barueri",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

AZUL = "#0E5A70"
ROSA = "#C76D8A"
LILAS = "#8E6BBE"
AZUL_COMP = "#4F86C6"
VERDE_COMP = "#65A88A"
LARANJA = "#E58A2B"
LILAS_P = "#8064A2"
VERDE = "#2E8B57"
VERMELHO = "#B83A3A"
CINZA = "#6B7280"
FUNDO = "#F8FAFB"

st.markdown(
    f"""
    <style>
    .stApp {{background: {FUNDO};}}
    .block-container {{max-width: 1500px; padding-top: 1rem; padding-bottom: 3rem;}}
    header[data-testid="stHeader"] {{background: rgba(0,0,0,0);}}
    [data-testid="stSidebar"] {{display:none;}}
    .brand {{display:flex; align-items:center; gap:14px; padding:6px 0 2px 0;}}
    .brand-icon {{
        background:{AZUL}; color:white; font-weight:800; width:44px;height:44px;border-radius:10px;
        display:flex;align-items:center;justify-content:center;font-size:13px;letter-spacing:.5px;
    }}
    .brand-title {{font-size:21px;font-weight:750;color:#17212b;line-height:1.05;}}
    .brand-sub {{font-size:12px;color:{CINZA};margin-top:3px;}}
    .eyebrow {{font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:{CINZA};font-weight:700;}}
    .hero-title {{font-size:28px;font-weight:780;color:#17212b;margin:.15rem 0 .2rem 0;}}
    .hero-sub {{font-size:14px;color:{CINZA};max-width:900px;}}
    .section-title {{font-size:20px;font-weight:750;color:#17212b;margin-top:8px;}}
    .note {{background:#FFF7E5;border:1px solid #F5D79A;border-radius:10px;padding:10px 13px;font-size:13px;color:#6c5425;}}
    .info {{background:#EFF7F9;border:1px solid #C9E2E8;border-radius:10px;padding:10px 13px;font-size:13px;color:#335a64;}}
    .metric-card {{
        background:white;border:1px solid #E5E7EB;border-radius:14px;padding:17px 18px;min-height:118px;
        box-shadow:0 1px 2px rgba(0,0,0,.02);
    }}
    .metric-label {{font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:{CINZA};font-weight:700;}}
    .metric-value {{font-size:28px;font-weight:780;color:#17212b;margin-top:6px;}}
    .metric-foot {{font-size:12px;color:{CINZA};margin-top:4px;}}
    .stTabs [data-baseweb="tab-list"] {{gap:8px; flex-wrap:wrap;}}
    .stTabs [data-baseweb="tab"] {{background:white;border:1px solid #E5E7EB;border-radius:9px;padding:8px 14px;}}
    .stTabs [aria-selected="true"] {{background:{AZUL}!important;color:white!important;border-color:{AZUL}!important;}}
    div[data-testid="stPlotlyChart"] {{background:white;border:1px solid #E5E7EB;border-radius:14px;padding:7px;}}
    .footer {{font-size:11px;color:{CINZA};padding-top:25px;}}
    </style>
    """,
    unsafe_allow_html=True,
)

def _normalizar_serie(s: pd.Series) -> pd.Series:
    return (
        s.astype("string")
         .str.normalize("NFKD")
         .str.encode("ascii", errors="ignore")
         .str.decode("utf-8")
         .str.lower()
         .str.strip()
    )

@st.cache_data(show_spinner="Carregando dados educacionais...")
def carregar_bases():
    mun = pd.read_csv(DATA_DIR / "base_municipios.csv", encoding="utf-8-sig")
    esc = pd.read_csv(DATA_DIR / "base_escolas.csv", encoding="utf-8-sig")
    inv = pd.read_csv(DATA_DIR / "investimento_inep.csv", encoding="utf-8-sig")

    mun["Município_Busca"] = _normalizar_serie(mun["Município"])
    esc["Escola_Busca"] = _normalizar_serie(esc["Escola"])

    for df in (mun, esc, inv):
        df["Ano"] = pd.to_numeric(df["Ano"], errors="coerce").astype("Int64")

    num_cols = [
        "Matemática","Língua Portuguesa","N","P","Aprovação Geral",
        "1º","2º","3º","4º","5º","6º","7º","8º","9º",
        "IDEB","Meta IDEB","Diferença para Meta",
        "Nível Matemática","Nível Língua Portuguesa"
    ]
    for df in (mun, esc):
        for c in num_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # ----------------------------------------------------
        # Notas padronizadas do IDEB por componente (0 a 10)
        # Fórmula:
        # nota = (proficiência - limite_inferior) /
        #        (limite_superior - limite_inferior) * 10
        #
        # Anos iniciais:
        # LP: 49 a 324 | Matemática: 60 a 322
        # Anos finais:
        # LP: 100 a 400 | Matemática: 100 a 400
        #
        # O cálculo é vetorizado e feito uma única vez no cache.
        # ----------------------------------------------------
        mask_fi = df["Etapa"].eq("Fundamental I")
        mask_fii = df["Etapa"].eq("Fundamental II")

        df["Nota Padronizada LP"] = np.nan
        df["Nota Padronizada Matemática"] = np.nan

        df.loc[mask_fi, "Nota Padronizada LP"] = (
            (df.loc[mask_fi, "Língua Portuguesa"] - 49.0)
            / (324.0 - 49.0)
            * 10.0
        )
        df.loc[mask_fi, "Nota Padronizada Matemática"] = (
            (df.loc[mask_fi, "Matemática"] - 60.0)
            / (322.0 - 60.0)
            * 10.0
        )

        df.loc[mask_fii, "Nota Padronizada LP"] = (
            (df.loc[mask_fii, "Língua Portuguesa"] - 100.0)
            / (400.0 - 100.0)
            * 10.0
        )
        df.loc[mask_fii, "Nota Padronizada Matemática"] = (
            (df.loc[mask_fii, "Matemática"] - 100.0)
            / (400.0 - 100.0)
            * 10.0
        )

        df["Nota Padronizada LP"] = df["Nota Padronizada LP"].clip(0, 10)
        df["Nota Padronizada Matemática"] = (
            df["Nota Padronizada Matemática"].clip(0, 10)
        )

        # N é um indicador oficial já existente na base.
        # O dashboard apenas converte seu tipo; nunca recalcula nem substitui N.
        df["N"] = pd.to_numeric(df["N"], errors="coerce")

    inv["Investimento por Estudante"] = pd.to_numeric(
        inv["Investimento por Estudante"], errors="coerce"
    )
    return mun, esc, inv

municipios, escolas, investimento = carregar_bases()

ANOS = sorted(int(x) for x in municipios["Ano"].dropna().unique())
ETAPAS = ["Fundamental I", "Fundamental II"]
SERIES = {
    "Fundamental I": ["1º","2º","3º","4º","5º"],
    "Fundamental II": ["6º","7º","8º","9º"],
}

def normalizar_texto(texto):
    """Normaliza somente o texto digitado pelo usuário."""
    if texto is None:
        return ""
    return (
        unicodedata.normalize("NFKD", str(texto))
        .encode("ascii", "ignore")
        .decode("utf-8")
        .lower()
        .strip()
    )

def filtrar_nomes_busca(df, coluna_nome, coluna_busca, termo="", filtro=None):
    """
    Usa a coluna *_Busca já criada no carregamento cacheado.
    Não remove acentos da base novamente a cada pesquisa.
    """
    base = df
    if filtro is not None:
        base = base.loc[filtro]

    pares = (
        base[[coluna_nome, coluna_busca]]
        .dropna(subset=[coluna_nome])
        .drop_duplicates()
    )

    busca = normalizar_texto(termo)
    if busca:
        pares = pares[
            pares[coluna_busca].astype("string").str.contains(
                busca, na=False, regex=False
            )
        ]

    return sorted(pares[coluna_nome].astype(str).unique().tolist())

def fmt(v, casas=2, sufixo=""):
    if pd.isna(v):
        return "—"
    return f"{float(v):.{casas}f}{sufixo}".replace(".", ",")

def escolher_rede(nome, etapa):
    redes = municipios.loc[
        (municipios["Município"] == nome) & (municipios["Etapa"] == etapa), "Rede"
    ].dropna().unique().tolist()
    for preferida in ["Municipal", "Pública", "Estadual", "Federal"]:
        if preferida in redes:
            return preferida
    return redes[0] if redes else None

def dados_municipio(nome, etapa, ano_ini=None, ano_fim=None, rede=None):
    x = municipios[
        (municipios["Município"] == nome) & (municipios["Etapa"] == etapa)
    ].copy()
    rede = rede or escolher_rede(nome, etapa)
    if rede is not None:
        x = x[x["Rede"] == rede]
    if ano_ini is not None:
        x = x[x["Ano"] >= ano_ini]
    if ano_fim is not None:
        x = x[x["Ano"] <= ano_fim]
    return x.sort_values("Ano")

def dados_escola(nome, etapa, ano_ini=None, ano_fim=None):
    x = escolas[
        (escolas["Escola"] == nome) & (escolas["Etapa"] == etapa)
    ].copy()
    if ano_ini is not None:
        x = x[x["Ano"] >= ano_ini]
    if ano_fim is not None:
        x = x[x["Ano"] <= ano_fim]
    return x.sort_values("Ano")

def ultima_linha(df, ano=None):
    if ano is not None:
        x = df[df["Ano"] == ano]
        if not x.empty:
            return x.iloc[-1]
    x = df.dropna(subset=["Ano"]).sort_values("Ano")
    return x.iloc[-1] if not x.empty else None

def estilo_fig(fig, titulo=None, ytitle=None, height=460):
    fig.update_layout(
        title=dict(text=titulo or "", x=.01, xanchor="left", font=dict(size=18)),
        height=height,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=35, r=35, t=70, b=40),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.10, x=0),
        font=dict(family="Arial, sans-serif", color="#26313b"),
    )
    fig.update_xaxes(showgrid=False, linecolor="#DDE3E8", tickmode="array", tickvals=ANOS)
    fig.update_yaxes(gridcolor="#EDF1F4", zeroline=False, title=ytitle)
    return fig

def grafico_ideb_meta(df, titulo):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Ano"], y=df["IDEB"], mode="lines+markers+text",
        text=[fmt(v,1) if pd.notna(v) else "" for v in df["IDEB"]],
        textposition="top center", name="IDEB",
        line=dict(color=AZUL, width=3), marker=dict(size=8)
    ))
    if df["Meta IDEB"].notna().any():
        fig.add_trace(go.Scatter(
            x=df["Ano"], y=df["Meta IDEB"], mode="lines+markers",
            name="Meta IDEB", line=dict(color=VERDE_COMP, width=2, dash="dot")
        ))
    return estilo_fig(fig, titulo, "IDEB", 475)

def grafico_lp_mat(df, titulo, modo="linhas"):
    fig = go.Figure()
    if modo == "linhas":
        fig.add_trace(go.Scatter(
            x=df["Ano"], y=df["Língua Portuguesa"], mode="lines+markers",
            name="Língua Portuguesa",
            line=dict(color=ROSA, width=3), marker=dict(size=8)
        ))
        adicionar_rotulos_acima(
            fig, df["Ano"], df["Língua Portuguesa"],
            casas=1, cor=ROSA, yshift=16
        )

        fig.add_trace(go.Scatter(
            x=df["Ano"], y=df["Matemática"], mode="lines+markers",
            name="Matemática",
            line=dict(color=LILAS, width=3), marker=dict(size=8)
        ))
        adicionar_rotulos_acima(
            fig, df["Ano"], df["Matemática"],
            casas=1, cor=LILAS, yshift=32
        )
    else:
        fig.add_trace(go.Bar(x=df["Ano"], y=df["Língua Portuguesa"], name="Língua Portuguesa", marker_color=ROSA))
        fig.add_trace(go.Bar(x=df["Ano"], y=df["Matemática"], name="Matemática", marker_color=LILAS))
        fig.update_layout(barmode="group")
    return estilo_fig(fig, titulo, "Proficiência SAEB", 480)

def grafico_lp_mat_n(df, titulo):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["Ano"], y=df["Língua Portuguesa"], name="Língua Portuguesa", marker_color=ROSA))
    fig.add_trace(go.Bar(x=df["Ano"], y=df["Matemática"], name="Matemática", marker_color=LILAS))
    fig.add_trace(go.Scatter(
        x=df["Ano"], y=df["N"], name="Nota Média Padronizada (N)",
        mode="lines+markers+text", yaxis="y2",
        text=[fmt(v,2) if pd.notna(v) else "" for v in df["N"]],
        textposition="top center", line=dict(color=LARANJA, width=3), marker=dict(size=8)
    ))
    estilo_fig(fig, titulo, "Proficiência SAEB", 505)
    fig.update_layout(
        barmode="group",
        yaxis2=dict(title="Nota Média Padronizada (N)", overlaying="y", side="right", showgrid=False, rangemode="tozero")
    )
    return fig

def grafico_aprovacao_series_p(df, etapa, titulo):
    fig = go.Figure()
    paleta = ["#3F7C91","#5A91A4","#76A6B7","#91BBC8","#ADCED6"]
    for i, serie in enumerate(SERIES[etapa]):
        if df[serie].notna().any():
            fig.add_trace(go.Bar(x=df["Ano"], y=df[serie], name=serie, marker_color=paleta[i % len(paleta)]))
    fig.add_trace(go.Scatter(
        x=df["Ano"], y=df["P"], name="Indicador de Rendimento (P)",
        mode="lines+markers+text", yaxis="y2",
        text=[fmt(v,3) if pd.notna(v) else "" for v in df["P"]],
        textposition="top center", line=dict(color=LILAS_P, width=3), marker=dict(size=8)
    ))
    estilo_fig(fig, titulo, "Taxa de aprovação (%)", 520)
    fig.update_layout(
        barmode="group",
        yaxis=dict(title="Taxa de aprovação (%)", range=[0,105], gridcolor="#EDF1F4"),
        yaxis2=dict(title="Indicador de Rendimento (P)", overlaying="y", side="right", showgrid=False)
    )
    return fig

def grafico_aprovacao_linhas(df, etapa, titulo):
    fig = go.Figure()
    for serie in SERIES[etapa]:
        if df[serie].notna().any():
            fig.add_trace(go.Scatter(x=df["Ano"], y=df[serie], mode="lines+markers", name=serie))
    if df["Aprovação Geral"].notna().any():
        fig.add_trace(go.Scatter(
            x=df["Ano"], y=df["Aprovação Geral"], mode="lines+markers", name="Aprovação geral",
            line=dict(color="#17212b", width=3)
        ))
    return estilo_fig(fig, titulo, "Taxa de aprovação (%)", 500)

def grafico_np_ideb(df, titulo):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["Ano"], y=df["N"], name="Nota Média Padronizada (N)", marker_color="#E5A8BA"))
    fig.add_trace(go.Scatter(
        x=df["Ano"], y=df["IDEB"], name="IDEB", mode="lines+markers+text",
        text=[fmt(v,1) if pd.notna(v) else "" for v in df["IDEB"]],
        textposition="top center", line=dict(color=LARANJA, width=3), marker=dict(size=8)
    ))
    fig.add_trace(go.Scatter(
        x=df["Ano"], y=df["P"], name="Indicador de Rendimento (P)",
        mode="lines+markers+text", yaxis="y2",
        text=[fmt(v,3) if pd.notna(v) else "" for v in df["P"]],
        textposition="top center", line=dict(color=LILAS_P, width=3), marker=dict(size=8)
    ))
    estilo_fig(fig, titulo, "N / IDEB", 510)
    fig.update_layout(yaxis2=dict(title="Indicador de Rendimento (P)", overlaying="y", side="right", showgrid=False))
    return fig

def grafico_comparacao_lp_mat(df1, nome1, df2, nome2, titulo):
    fig = go.Figure()
    traces = [
        (df1, "Língua Portuguesa", f"{nome1} — LP", ROSA, 14),
        (df1, "Matemática", f"{nome1} — Matemática", LILAS, 26),
        (df2, "Língua Portuguesa", f"{nome2} — LP", AZUL_COMP, 38),
        (df2, "Matemática", f"{nome2} — Matemática", VERDE_COMP, 50),
    ]

    for d, col, nome, cor, deslocamento in traces:
        fig.add_trace(go.Scatter(
            x=d["Ano"],
            y=d[col],
            mode="lines+markers",
            name=nome,
            line=dict(color=cor, width=3),
            marker=dict(size=8)
        ))
        adicionar_rotulos_acima(
            fig, d["Ano"], d[col],
            casas=1, cor=cor, yshift=deslocamento
        )

    fig = estilo_fig(fig, titulo, "Proficiência SAEB", 570)
    fig.update_layout(margin=dict(l=35, r=35, t=100, b=45))
    return fig


def grafico_comparacao_indicador(datasets, indicador, titulo):
    fig = go.Figure()
    cores = [AZUL, AZUL_COMP, VERDE_COMP, ROSA, LILAS, LARANJA]

    casas = 1 if indicador in ["IDEB", "Aprovação Geral"] else 2

    for i, (nome, df) in enumerate(datasets):
        cor = cores[i % len(cores)]
        fig.add_trace(go.Scatter(
            x=df["Ano"],
            y=df[indicador],
            mode="lines+markers",
            name=nome,
            line=dict(color=cor, width=3),
            marker=dict(size=8)
        ))
        adicionar_rotulos_acima(
            fig,
            df["Ano"],
            df[indicador],
            casas=casas,
            cor=cor,
            yshift=14 + (i * 12)
        )

    fig = estilo_fig(fig, titulo, indicador, 545)
    fig.update_layout(margin=dict(l=35, r=35, t=100, b=45))
    return fig


def grafico_comparacao_etapas(df_fi, df_fii, indicador, titulo, casas=2, metas=False):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_fi["Ano"], y=df_fi[indicador],
        mode="lines+markers",
        name=f"Fundamental I — {indicador}",
        line=dict(color=AZUL, width=3),
        marker=dict(size=8)
    ))
    adicionar_rotulos_acima(
        fig, df_fi["Ano"], df_fi[indicador],
        casas=casas, cor=AZUL, yshift=16
    )

    fig.add_trace(go.Scatter(
        x=df_fii["Ano"], y=df_fii[indicador],
        mode="lines+markers",
        name=f"Fundamental II — {indicador}",
        line=dict(color=LILAS, width=3),
        marker=dict(size=8)
    ))
    adicionar_rotulos_acima(
        fig, df_fii["Ano"], df_fii[indicador],
        casas=casas, cor=LILAS, yshift=32
    )

    if metas and "Meta IDEB" in df_fi.columns and "Meta IDEB" in df_fii.columns:
        if df_fi["Meta IDEB"].notna().any():
            fig.add_trace(go.Scatter(
                x=df_fi["Ano"], y=df_fi["Meta IDEB"],
                mode="lines+markers",
                name="Meta — Fundamental I",
                line=dict(color=VERDE_COMP, width=2, dash="dot")
            ))
        if df_fii["Meta IDEB"].notna().any():
            fig.add_trace(go.Scatter(
                x=df_fii["Ano"], y=df_fii["Meta IDEB"],
                mode="lines+markers",
                name="Meta — Fundamental II",
                line=dict(color=LARANJA, width=2, dash="dot")
            ))

    ytitle = "Taxa de aprovação (%)" if indicador == "Aprovação Geral" else indicador
    fig = estilo_fig(fig, titulo, ytitle, 530)
    fig.update_layout(margin=dict(l=35, r=35, t=100, b=45))

    if indicador == "Aprovação Geral":
        fig.update_yaxes(range=[0, 105])

    return fig



def grafico_notas_padronizadas_componentes(df, titulo):
    """
    LP e Matemática já na mesma escala 0–10.
    A linha N mostra a média padronizada usada no IDEB.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["Ano"],
        y=df["Nota Padronizada LP"],
        mode="lines+markers+text",
        name="LP padronizada",
        text=[
            fmt(v, 2) if pd.notna(v) else ""
            for v in df["Nota Padronizada LP"]
        ],
        textposition="top center",
        line=dict(color=ROSA, width=3),
        marker=dict(size=8)
    ))

    fig.add_trace(go.Scatter(
        x=df["Ano"],
        y=df["Nota Padronizada Matemática"],
        mode="lines+markers+text",
        name="Matemática padronizada",
        text=[
            fmt(v, 2) if pd.notna(v) else ""
            for v in df["Nota Padronizada Matemática"]
        ],
        textposition="top center",
        line=dict(color=LILAS, width=3),
        marker=dict(size=8)
    ))

    fig.add_trace(go.Scatter(
        x=df["Ano"],
        y=df["N"],
        mode="lines+markers",
        name="N — média padronizada",
        line=dict(color=LARANJA, width=2, dash="dot"),
        marker=dict(size=7)
    ))

    estilo_fig(
        fig,
        titulo,
        "Nota padronizada (0–10)",
        480
    )
    fig.update_yaxes(range=[0, 10])

    return fig


def grafico_composicao_ideb_temporal(df, titulo):
    """
    Para uma MESMA unidade ao longo do tempo:
    N e IDEB juntos; P em gráfico próprio.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["Ano"],
        y=df["N"],
        mode="lines+markers+text",
        name="N",
        text=[fmt(v,2) if pd.notna(v) else "" for v in df["N"]],
        textposition="top center",
        line=dict(color=ROSA, width=3),
        marker=dict(size=8)
    ))

    fig.add_trace(go.Scatter(
        x=df["Ano"],
        y=df["IDEB"],
        mode="lines+markers+text",
        name="IDEB",
        text=[fmt(v,1) if pd.notna(v) else "" for v in df["IDEB"]],
        textposition="top center",
        line=dict(color=LARANJA, width=3),
        marker=dict(size=8)
    ))

    return estilo_fig(
        fig,
        titulo,
        "Nota Média Padronizada (N) / IDEB",
        460
    )

def grafico_p_temporal(df, titulo):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["Ano"],
        y=df["P"],
        mode="lines+markers+text",
        name="P",
        text=[fmt(v,3) if pd.notna(v) else "" for v in df["P"]],
        textposition="top center",
        line=dict(color=LILAS_P, width=3),
        marker=dict(size=8)
    ))

    return estilo_fig(
        fig,
        titulo,
        "Indicador de Rendimento (P)",
        390
    )

def tabela_composicao_unidades(rows):
    """
    Comparação transversal: cada município/escola é uma observação
    independente. Não cria linhas entre unidades.
    """
    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).copy()

    for col in [
        "Nota Padronizada LP",
        "Nota Padronizada Matemática",
        "N",
        "P",
        "IDEB"
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    return out

def cards_composicao_ideb(trans, coluna_nome):
    """
    Exibe N × P → IDEB por unidade, sem sugerir sequência entre elas.
    """
    if trans.empty:
        st.info("Não há dados suficientes para a composição do IDEB.")
        return

    for _, r in trans.iterrows():
        st.markdown(
            f'<div class="metric-card" style="margin-bottom:10px;">'
            f'<div class="metric-label">{r[coluna_nome]}</div>'
            f'<div class="metric-value" style="font-size:19px;">'
            f'LP {fmt(r.get("Nota Padronizada LP"),2)} + '
            f'MAT {fmt(r.get("Nota Padronizada Matemática"),2)} '
            f'→ N {fmt(r.get("N"),2)}</div>'
            f'<div style="font-size:18px;font-weight:750;margin-top:7px;">'
            f'N {fmt(r.get("N"),2)} × P {fmt(r.get("P"),3)} '
            f'→ IDEB {fmt(r.get("IDEB"),1)}</div>'
            f'<div class="metric-foot">LP e Matemática estão padronizadas na escala 0–10; '
            f'N é a média das duas notas padronizadas.</div>'
            f'</div>',
            unsafe_allow_html=True
        )


RANKING_INDICADORES = {
    "IDEB": {
        "coluna": "IDEB",
        "casas": 1,
        "nivel": None,
        "padrao": None,
        "rotulo": "IDEB",
    },
    "Língua Portuguesa (SAEB)": {
        "coluna": "Língua Portuguesa",
        "casas": 1,
        "nivel": "Nível Língua Portuguesa",
        "padrao": "Padrão Língua Portuguesa",
        "rotulo": "Língua Portuguesa",
    },
    "Matemática (SAEB)": {
        "coluna": "Matemática",
        "casas": 1,
        "nivel": "Nível Matemática",
        "padrao": "Padrão Matemática",
        "rotulo": "Matemática",
    },
    "Nota Padronizada LP (0–10)": {
        "coluna": "Nota Padronizada LP",
        "casas": 2,
        "nivel": None,
        "padrao": None,
        "rotulo": "Nota padronizada LP",
    },
    "Nota Padronizada Matemática (0–10)": {
        "coluna": "Nota Padronizada Matemática",
        "casas": 2,
        "nivel": None,
        "padrao": None,
        "rotulo": "Nota padronizada Matemática",
    },
    "Nota Média Padronizada (N)": {
        "coluna": "N",
        "casas": 2,
        "nivel": None,
        "padrao": None,
        "rotulo": "N",
    },
}


def _ano_anterior_disponivel(df, etapa, ano):
    anos = sorted(
        int(a) for a in df.loc[
            (df["Etapa"] == etapa) &
            (pd.to_numeric(df["Ano"], errors="coerce") < int(ano)),
            "Ano"
        ].dropna().unique()
    )
    return anos[-1] if anos else None


def _ordenar_ranking_com_desempate(
    atual,
    historico,
    chave,
    etapa,
    ano,
    coluna_indicador,
    filtro_historico=None
):
    """
    Gera posições únicas.

    Desempate:
    1) indicador atual;
    2) N atual;
    3) indicador da edição imediatamente anterior disponível;
    4) N da edição anterior;
    5) ordem alfabética.

    N é sempre lido da base; nunca é recalculado.
    """
    x = atual.copy()

    x[coluna_indicador] = pd.to_numeric(x[coluna_indicador], errors="coerce")
    if "N" in x.columns:
        x["N"] = pd.to_numeric(x["N"], errors="coerce")
    else:
        x["N"] = np.nan

    ano_ant = _ano_anterior_disponivel(historico, etapa, ano)

    x["_IndicadorAnterior"] = np.nan
    x["_NAnterior"] = np.nan

    if ano_ant is not None:
        ant = historico[
            (historico["Etapa"] == etapa) &
            (historico["Ano"] == ano_ant)
        ].copy()

        if filtro_historico is not None:
            ant = filtro_historico(ant)

        ant[coluna_indicador] = pd.to_numeric(
            ant[coluna_indicador], errors="coerce"
        )
        ant["N"] = pd.to_numeric(ant["N"], errors="coerce")

        ant = (
            ant.sort_values(
                [chave, coluna_indicador, "N"],
                ascending=[True, False, False],
                na_position="last"
            )
            .drop_duplicates(chave, keep="first")
            [[chave, coluna_indicador, "N"]]
            .rename(columns={
                coluna_indicador: "_IndicadorAnterior",
                "N": "_NAnterior"
            })
        )

        x = x.drop(columns=["_IndicadorAnterior", "_NAnterior"]).merge(
            ant, on=chave, how="left"
        )

    # A ordenação por várias colunas é suportada diretamente pelo pandas.
    # NaNs ficam por último em cada critério de desempate.
    x = x.sort_values(
        [coluna_indicador, "N", "_IndicadorAnterior", "_NAnterior", chave],
        ascending=[False, False, False, False, True],
        na_position="last"
    ).reset_index(drop=True)

    # Posição sequencial e única: 1, 2, 3...
    x["Posição"] = pd.Series(
        range(1, len(x) + 1), index=x.index, dtype="Int64"
    )

    return x


def ranking_base_municipios_indicador(etapa, ano, indicador_label, rede):
    cfg = RANKING_INDICADORES[indicador_label]
    coluna = cfg["coluna"]

    filtro = (
        (municipios["Etapa"] == etapa) &
        (municipios["Ano"] == ano) &
        (municipios["Rede"] == rede)
    )
    x = municipios[filtro].copy()

    x[coluna] = pd.to_numeric(x[coluna], errors="coerce")
    x["IDEB"] = pd.to_numeric(x["IDEB"], errors="coerce")
    x["N"] = pd.to_numeric(x["N"], errors="coerce")
    x = x.dropna(subset=[coluna])

    x = (
        x.sort_values(
            ["Município", coluna, "N"],
            ascending=[True, False, False],
            na_position="last"
        )
        .drop_duplicates("Município", keep="first")
        .copy()
    )

    return _ordenar_ranking_com_desempate(
        atual=x,
        historico=municipios,
        chave="Município",
        etapa=etapa,
        ano=ano,
        coluna_indicador=coluna,
        filtro_historico=lambda d: d[d["Rede"] == rede].copy()
    )


def ranking_base_escolas_indicador(
    etapa,
    ano,
    indicador_label,
    incluir_itbs=True
):
    cfg = RANKING_INDICADORES[indicador_label]
    coluna = cfg["coluna"]

    x = escolas[
        (escolas["Etapa"] == etapa) &
        (escolas["Ano"] == ano)
    ].copy()

    if not incluir_itbs:
        x = x.loc[~mascara_itb(x["Escola"])].copy()

    x[coluna] = pd.to_numeric(x[coluna], errors="coerce")
    x["IDEB"] = pd.to_numeric(x["IDEB"], errors="coerce")
    x["N"] = pd.to_numeric(x["N"], errors="coerce")
    x = x.dropna(subset=[coluna])

    x = (
        x.sort_values(
            ["Escola", coluna, "N"],
            ascending=[True, False, False],
            na_position="last"
        )
        .drop_duplicates("Escola", keep="first")
        .copy()
    )

    def filtro_ant(d):
        if not incluir_itbs:
            return d.loc[~mascara_itb(d["Escola"])].copy()
        return d

    return _ordenar_ranking_com_desempate(
        atual=x,
        historico=escolas,
        chave="Escola",
        etapa=etapa,
        ano=ano,
        coluna_indicador=coluna,
        filtro_historico=filtro_ant
    )


def comp_ranking_municipios_indicador(
    etapa, indicador_label, ano_ini, ano_fim, rede_comparacao
):
    cfg = RANKING_INDICADORES[indicador_label]
    coluna = cfg["coluna"]
    nivel_col = cfg["nivel"]
    padrao_col = cfg["padrao"]

    ini = ranking_base_municipios_indicador(
        etapa, ano_ini, indicador_label, rede_comparacao
    )
    fim = ranking_base_municipios_indicador(
        etapa, ano_fim, indicador_label, rede_comparacao
    )

    cols_ini = ["Município", "Rede", coluna, "Posição"]
    cols_fim = ["Município", "Rede", coluna, "Posição"]

    # IDEB deve aparecer também como informação auxiliar, mas sem
    # duplicar a coluna quando o próprio ranking já é por IDEB.
    if coluna != "IDEB":
        cols_ini.append("IDEB")
        cols_fim.append("IDEB")

    if nivel_col:
        cols_ini.append(nivel_col)
        cols_fim.append(nivel_col)
    if padrao_col:
        cols_ini.append(padrao_col)
        cols_fim.append(padrao_col)

    ini = ini[cols_ini].copy()
    fim = fim[cols_fim].copy()

    ini = ini.rename(columns={
        "Rede": "Rede Inicial",
        coluna: "Resultado Inicial",
        **({"IDEB": "IDEB Inicial"} if coluna != "IDEB" else {}),
        "Posição": "Posição Inicial",
        **({nivel_col: "Nível Inicial"} if nivel_col else {}),
        **({padrao_col: "Padrão Inicial"} if padrao_col else {}),
    })

    fim = fim.rename(columns={
        "Rede": "Rede Atual",
        coluna: "Resultado Atual",
        **({"IDEB": "IDEB Atual"} if coluna != "IDEB" else {}),
        "Posição": "Posição Atual",
        **({nivel_col: "Nível Atual"} if nivel_col else {}),
        **({padrao_col: "Padrão Atual"} if padrao_col else {}),
    })

    if coluna == "IDEB":
        ini["IDEB Inicial"] = ini["Resultado Inicial"]
        fim["IDEB Atual"] = fim["Resultado Atual"]

    comp = fim.merge(ini, on="Município", how="left")
    comp["Variação de Posição"] = comp["Posição Inicial"] - comp["Posição Atual"]
    comp["Movimento"] = comp["Variação de Posição"].apply(texto_movimento)

    if "Nível Atual" not in comp.columns:
        comp["Nível Atual"] = pd.NA
        comp["Nível Inicial"] = pd.NA
    if "Padrão Atual" not in comp.columns:
        comp["Padrão Atual"] = pd.NA
        comp["Padrão Inicial"] = pd.NA

    return comp.sort_values(["Posição Atual", "Município"]).reset_index(drop=True)

def comp_ranking_escolas_indicador(
    etapa,
    indicador_label,
    ano_ini,
    ano_fim,
    incluir_itbs=True
):
    cfg = RANKING_INDICADORES[indicador_label]
    coluna = cfg["coluna"]
    nivel_col = cfg["nivel"]
    padrao_col = cfg["padrao"]

    ini = ranking_base_escolas_indicador(
        etapa,
        ano_ini,
        indicador_label,
        incluir_itbs=incluir_itbs
    )
    fim = ranking_base_escolas_indicador(
        etapa,
        ano_fim,
        indicador_label,
        incluir_itbs=incluir_itbs
    )

    cols_ini = ["Escola", coluna, "Posição"]
    cols_fim = ["Escola", coluna, "Posição"]

    if coluna != "IDEB":
        cols_ini.append("IDEB")
        cols_fim.append("IDEB")

    if nivel_col:
        cols_ini.append(nivel_col)
        cols_fim.append(nivel_col)
    if padrao_col:
        cols_ini.append(padrao_col)
        cols_fim.append(padrao_col)

    ini = ini[cols_ini].copy()
    fim = fim[cols_fim].copy()

    ini = ini.rename(columns={
        coluna: "Resultado Inicial",
        **({"IDEB": "IDEB Inicial"} if coluna != "IDEB" else {}),
        "Posição": "Posição Inicial",
        **({nivel_col: "Nível Inicial"} if nivel_col else {}),
        **({padrao_col: "Padrão Inicial"} if padrao_col else {}),
    })

    fim = fim.rename(columns={
        coluna: "Resultado Atual",
        **({"IDEB": "IDEB Atual"} if coluna != "IDEB" else {}),
        "Posição": "Posição Atual",
        **({nivel_col: "Nível Atual"} if nivel_col else {}),
        **({padrao_col: "Padrão Atual"} if padrao_col else {}),
    })

    if coluna == "IDEB":
        ini["IDEB Inicial"] = ini["Resultado Inicial"]
        fim["IDEB Atual"] = fim["Resultado Atual"]

    comp = fim.merge(ini, on="Escola", how="left")
    comp["Variação de Posição"] = comp["Posição Inicial"] - comp["Posição Atual"]
    comp["Movimento"] = comp["Variação de Posição"].apply(texto_movimento)

    if "Nível Atual" not in comp.columns:
        comp["Nível Atual"] = pd.NA
        comp["Nível Inicial"] = pd.NA
    if "Padrão Atual" not in comp.columns:
        comp["Padrão Atual"] = pd.NA
        comp["Padrão Inicial"] = pd.NA

    return comp.sort_values(["Posição Atual", "Escola"]).reset_index(drop=True)

def adicionar_barueri_indicador(
    comp, etapa, indicador_label, ano_ini, ano_fim, rede_comparacao
):
    """
    Barueri permanece Municipal e entra como referência no universo
    da rede selecionada.
    """
    cfg = RANKING_INDICADORES[indicador_label]
    coluna = cfg["coluna"]
    nivel_col = cfg["nivel"]
    padrao_col = cfg["padrao"]

    def linha_barueri(ano):
        x = municipios[
            (municipios["Município"] == "Barueri") &
            (municipios["Rede"] == "Municipal") &
            (municipios["Etapa"] == etapa) &
            (municipios["Ano"] == ano)
        ].copy()
        if x.empty:
            return None
        x[coluna] = pd.to_numeric(x[coluna], errors="coerce")
        x = x.dropna(subset=[coluna])
        if x.empty:
            return None
        return x.sort_values(coluna, ascending=False).iloc[0]

    bi = linha_barueri(ano_ini)
    bf = linha_barueri(ano_fim)
    if bf is None:
        return comp

    rank_ini = ranking_base_municipios_indicador(
        etapa, ano_ini, indicador_label, rede_comparacao
    )
    rank_fim = ranking_base_municipios_indicador(
        etapa, ano_fim, indicador_label, rede_comparacao
    )

    def posicao_barueri_no_recorte(rank_rede, linha_b, ano):
        if linha_b is None:
            return pd.NA

        # Monta temporariamente o universo da rede + Barueri Municipal.
        temp = rank_rede.copy()
        temp = temp[temp["Município"] != "Barueri"].copy()

        bar = linha_b.to_dict()
        bar["Município"] = "Barueri"
        bar["Rede"] = "Municipal"
        temp = pd.concat([temp, pd.DataFrame([bar])], ignore_index=True)

        # O helper busca o histórico na rede selecionada; para Barueri,
        # corrigimos explicitamente os critérios anteriores com a Rede Municipal.
        temp = _ordenar_ranking_com_desempate(
            atual=temp,
            historico=municipios,
            chave="Município",
            etapa=etapa,
            ano=ano,
            coluna_indicador=coluna,
            filtro_historico=lambda d: d[d["Rede"] == rede_comparacao].copy()
        )

        ano_ant = _ano_anterior_disponivel(municipios, etapa, ano)
        if ano_ant is not None:
            b_ant = municipios[
                (municipios["Município"] == "Barueri") &
                (municipios["Rede"] == "Municipal") &
                (municipios["Etapa"] == etapa) &
                (municipios["Ano"] == ano_ant)
            ].copy()
            if not b_ant.empty:
                b_ant[coluna] = pd.to_numeric(b_ant[coluna], errors="coerce")
                b_ant["N"] = pd.to_numeric(b_ant["N"], errors="coerce")
                br = b_ant.sort_values(
                    [coluna, "N"], ascending=[False, False], na_position="last"
                ).iloc[0]
                temp.loc[temp["Município"] == "Barueri", "_IndicadorAnterior"] = br.get(coluna)
                temp.loc[temp["Município"] == "Barueri", "_NAnterior"] = br.get("N")

                temp = temp.sort_values(
                    [coluna, "N", "_IndicadorAnterior", "_NAnterior", "Município"],
                    ascending=[False, False, False, False, True],
                    na_position="last"
                ).reset_index(drop=True)
                temp["Posição"] = pd.Series(
                    range(1, len(temp) + 1), index=temp.index, dtype="Int64"
                )

        return int(
            temp.loc[temp["Município"] == "Barueri", "Posição"].iloc[0]
        )

    pos_ini = posicao_barueri_no_recorte(rank_ini, bi, ano_ini)
    pos_fim = posicao_barueri_no_recorte(rank_fim, bf, ano_fim)

    variacao = (
        pos_ini - pos_fim
        if pd.notna(pos_ini) and pd.notna(pos_fim)
        else pd.NA
    )

    linha = {
        "Município": "Barueri",
        "Rede Atual": "Municipal (referência)",
        "Rede Inicial": "Municipal (referência)",
        "Resultado Atual": bf.get(coluna),
        "Resultado Inicial": bi.get(coluna) if bi is not None else pd.NA,
        "IDEB Atual": bf.get("IDEB"),
        "IDEB Inicial": bi.get("IDEB") if bi is not None else pd.NA,
        "Posição Atual": pos_fim,
        "Posição Inicial": pos_ini,
        "Variação de Posição": variacao,
        "Movimento": texto_movimento(variacao),
        "Nível Atual": bf.get(nivel_col) if nivel_col else pd.NA,
        "Nível Inicial": bi.get(nivel_col) if (nivel_col and bi is not None) else pd.NA,
        "Padrão Atual": bf.get(padrao_col) if padrao_col else pd.NA,
        "Padrão Inicial": bi.get(padrao_col) if (padrao_col and bi is not None) else pd.NA,
        "_Referencia": True,
    }

    out = comp.copy()
    out["_Referencia"] = False
    out = out[out["Município"] != "Barueri"]
    out = pd.concat([out, pd.DataFrame([linha])], ignore_index=True)
    return out.sort_values(["Posição Atual", "Município"]).reset_index(drop=True)

def ranking_selecionados_municipios(
    nomes, etapa, indicador_label, ano_ini, ano_fim, rede_comparacao
):
    """
    Ranking somente entre os municípios selecionados.
    Barueri é sempre Municipal.

    Desempate:
    indicador atual > N atual > indicador da edição anterior >
    N da edição anterior > ordem alfabética.
    """
    nomes_finais = []
    for nome in ["Barueri"] + list(nomes):
        if nome not in nomes_finais:
            nomes_finais.append(nome)

    cfg = RANKING_INDICADORES[indicador_label]
    coluna = cfg["coluna"]
    nivel_col = cfg["nivel"]
    padrao_col = cfg["padrao"]

    def montar_ano(ano):
        rows = []
        for nome in nomes_finais:
            rede = "Municipal" if nome == "Barueri" else rede_comparacao
            x = municipios[
                (municipios["Município"] == nome) &
                (municipios["Rede"] == rede) &
                (municipios["Etapa"] == etapa) &
                (municipios["Ano"] == ano)
            ].copy()

            if x.empty:
                continue

            x[coluna] = pd.to_numeric(x[coluna], errors="coerce")
            x["N"] = pd.to_numeric(x["N"], errors="coerce")
            x = x.dropna(subset=[coluna])

            if x.empty:
                continue

            r = x.sort_values(
                [coluna, "N"],
                ascending=[False, False],
                na_position="last"
            ).iloc[0]

            rows.append({
                "Município": nome,
                "Rede": "Municipal (referência)" if nome == "Barueri" else rede,
                "Resultado": r.get(coluna),
                "N_desempate": r.get("N"),
                "IDEB": r.get("IDEB"),
                "Nível": r.get(nivel_col) if nivel_col else pd.NA,
                "Padrão": r.get(padrao_col) if padrao_col else pd.NA,
            })

        out = pd.DataFrame(rows)
        if out.empty:
            return out

        # edição anterior ao ano que está sendo classificado
        ano_ant = _ano_anterior_disponivel(municipios, etapa, ano)
        ant_ind = {}
        ant_n = {}

        if ano_ant is not None:
            for nome in out["Município"]:
                rede = "Municipal" if nome == "Barueri" else rede_comparacao
                a = municipios[
                    (municipios["Município"] == nome) &
                    (municipios["Rede"] == rede) &
                    (municipios["Etapa"] == etapa) &
                    (municipios["Ano"] == ano_ant)
                ].copy()

                if not a.empty:
                    a[coluna] = pd.to_numeric(a[coluna], errors="coerce")
                    a["N"] = pd.to_numeric(a["N"], errors="coerce")
                    a = a.sort_values(
                        [coluna, "N"],
                        ascending=[False, False],
                        na_position="last"
                    )
                    ant_ind[nome] = a.iloc[0].get(coluna)
                    ant_n[nome] = a.iloc[0].get("N")

        out["_IndicadorAnterior"] = out["Município"].map(ant_ind)
        out["_NAnterior"] = out["Município"].map(ant_n)

        out = out.sort_values(
            ["Resultado", "N_desempate", "_IndicadorAnterior", "_NAnterior", "Município"],
            ascending=[False, False, False, False, True],
            na_position="last"
        ).reset_index(drop=True)

        out["Posição"] = pd.Series(
            range(1, len(out) + 1), index=out.index, dtype="Int64"
        )
        return out

    ini = montar_ano(ano_ini)
    fim = montar_ano(ano_fim)
    if fim.empty:
        return pd.DataFrame()

    ini = ini.rename(columns={
        "Rede": "Rede Inicial",
        "Resultado": "Resultado Inicial",
        "IDEB": "IDEB Inicial",
        "Nível": "Nível Inicial",
        "Padrão": "Padrão Inicial",
        "Posição": "Posição Inicial",
    })
    fim = fim.rename(columns={
        "Rede": "Rede Atual",
        "Resultado": "Resultado Atual",
        "IDEB": "IDEB Atual",
        "Nível": "Nível Atual",
        "Padrão": "Padrão Atual",
        "Posição": "Posição Atual",
    })

    comp = fim.merge(
        ini[[
            "Município", "Rede Inicial", "Resultado Inicial", "IDEB Inicial",
            "Nível Inicial", "Padrão Inicial", "Posição Inicial"
        ]],
        on="Município",
        how="left"
    )

    comp["Variação de Posição"] = comp["Posição Inicial"] - comp["Posição Atual"]
    comp["Movimento"] = comp["Variação de Posição"].apply(texto_movimento)
    comp["_Referencia"] = comp["Município"].eq("Barueri")
    return comp.sort_values(["Posição Atual", "Município"]).reset_index(drop=True)



# Cores fixas de indicadores que NÃO possuem padrão pedagógico.
COR_RANK_IDEB = "#1F6F78"   # azul-petróleo
COR_RANK_N = "#8A63A8"      # lilás


def estilo_indicador_ranking(indicador_label, padrao=None):
    """
    IDEB e N: cor identifica o indicador, não nível de aprendizagem.
    LP/Matemática: cor continua identificando o padrão pedagógico.
    """
    nome = str(indicador_label).strip().lower()

    if nome == "ideb":
        return COR_RANK_IDEB, None

    if nome in {
        "n",
        "nota média padronizada (n)",
        "nota media padronizada (n)"
    }:
        return COR_RANK_N, None

    # Para proficiências SAEB, mantém a classificação já construída.
    return cor_padrao(padrao), padrao



def texto_classificacao_ranking(indicador_label, nivel=None, padrao=None):
    """
    Só proficiências SAEB de LP/Matemática exibem nível/padrão.
    IDEB e N retornam texto vazio.
    """
    nome = str(indicador_label).strip().lower()
    if nome == "ideb" or nome in {
        "n",
        "nota média padronizada (n)",
        "nota media padronizada (n)"
    }:
        return ""

    partes = []
    if pd.notna(nivel) and str(nivel).strip() not in {"", "<NA>", "nan"}:
        partes.append(str(nivel))
    if pd.notna(padrao) and str(padrao).strip() not in {"", "<NA>", "nan"}:
        partes.append(str(padrao))
    return " — ".join(partes)


def cor_movimento(valor):
    if pd.isna(valor):
        return "#6B7280"
    if valor > 0:
        return "#2E8B57"
    if valor < 0:
        return "#B83A3A"
    return "#6B7280"

def tabela_visual_ranking(
    comp,
    entidade,
    indicador_label,
    quantidade="Todos",
    destaque_barueri=False
):
    """
    Ranking com aparência de tabela: cada unidade ocupa uma linha,
    e as informações ficam organizadas em colunas.
    """
    chave = "Município" if entidade == "Município" else "Escola"
    cfg = RANKING_INDICADORES[indicador_label]

    base = comp.copy()

    if quantidade != "Todos":
        if entidade == "Município" and "_Referencia" in base.columns:
            ref = base[base["_Referencia"] == True]
            outros = base[base["_Referencia"] != True].head(int(quantidade))
            base = pd.concat([outros, ref], ignore_index=True)
        else:
            base = base.head(int(quantidade))

    base = base.drop_duplicates(chave).sort_values(["Posição Atual", chave])

    for _, r in base.iterrows():
        movimento = r.get("Movimento", "—")
        cor_mov = cor_movimento(r.get("Variação de Posição"))

        nivel_txt = ""
        if pd.notna(r.get("Nível Atual")):
            nivel_txt = f'Nível {int(float(r.get("Nível Atual")))}'
            if pd.notna(r.get("Padrão Atual")):
                nivel_txt += f' • {r.get("Padrão Atual")}'

        cor_nivel = "#E5E7EB"
        if pd.notna(r.get("Nível Atual")):
            cor_nivel = cores_niveis(
                st.session_state.get("rank_m_etapa")
                if entidade == "Município"
                else st.session_state.get("rank_e_etapa")
            ).get(int(float(r.get("Nível Atual"))), "#E5E7EB")

        destaque = (
            "border:2px solid #0E5A70;"
            if destaque_barueri and r[chave] == "Barueri"
            else "border:1px solid #E5E7EB;"
        )

        st.markdown(
            f'<div style="{destaque}background:white;border-radius:12px;'
            f'padding:12px 14px;margin-bottom:8px;display:grid;'
            f'grid-template-columns:72px minmax(190px,2fr) 115px 115px 115px minmax(150px,1.4fr);'
            f'gap:12px;align-items:center;">'
            f'<div><div class="metric-label">Posição</div>'
            f'<div style="font-size:23px;font-weight:800;">{int(r["Posição Atual"])}º</div></div>'
            f'<div><div class="metric-label">{entidade}</div>'
            f'<div style="font-size:15px;font-weight:750;">{r[chave]}</div>'
            f'<div style="font-size:11px;color:#6B7280;">{r.get("Rede Atual","")}</div></div>'
            f'<div><div class="metric-label">{cfg["rotulo"]}</div>'
            f'<div style="font-size:17px;font-weight:750;">{fmt(r.get("Resultado Atual"),cfg["casas"])}</div></div>'
            f'<div><div class="metric-label">IDEB</div>'
            f'<div style="font-size:17px;font-weight:750;">{fmt(r.get("IDEB Atual"),1)}</div></div>'
            f'<div><div class="metric-label">Movimento</div>'
            f'<div style="font-size:16px;font-weight:800;color:{cor_mov};">{movimento}</div>'
            f'<div style="font-size:11px;color:#6B7280;">de {fmt(r.get("Posição Inicial"),0)}º</div></div>'
            f'<div><div class="metric-label">Classificação</div>'
            f'<div style="font-size:13px;font-weight:700;">{nivel_txt or "—"}</div>'
            f'<div style="height:5px;border-radius:4px;background:{cor_nivel};margin-top:6px;"></div></div>'
            f'</div>',
            unsafe_allow_html=True
        )



def classificacao_visual_ranking(indicador_label, valor, nivel_saeb=None, padrao_saeb=None):
    if pd.isna(valor):
        return "Sem resultado", "#CBD5E1"

    if indicador_label in ["Língua Portuguesa (SAEB)", "Matemática (SAEB)"]:
        if pd.notna(nivel_saeb):
            n = int(float(nivel_saeb))
            etapa = st.session_state.get("rank_m_etapa") or st.session_state.get("rank_e_etapa") or "Fundamental I"
            cor = cores_niveis(etapa).get(n, "#CBD5E1")
            rotulo = f"Nível {n}"
            if pd.notna(padrao_saeb) and str(padrao_saeb).strip():
                rotulo += f" • {padrao_saeb}"
            return rotulo, cor
        return "Sem nível", "#CBD5E1"

    if indicador_label == "IDEB":
        v = float(valor)
        if v >= 6.0:
            return "Avançado", "#16A34A"
        if v >= 5.0:
            return "Adequado", "#F4B400"
        if v >= 4.0:
            return "Básico", "#2583E9"
        if v >= 3.0:
            return "Insuficiente", "#F97316"
        return "Crítico", "#E31B23"

    return "Nota padronizada", "#0E5A70"


def ranking_visual_com_barras(comp, entidade, indicador_label, quantidade="Todos", destaque_barueri=False):
    chave = "Município" if entidade == "Município" else "Escola"
    cfg = RANKING_INDICADORES[indicador_label]
    base = comp.copy()

    if quantidade != "Todos":
        if entidade == "Município" and "_Referencia" in base.columns:
            ref = base[base["_Referencia"] == True]
            outros = base[base["_Referencia"] != True].head(int(quantidade))
            base = pd.concat([outros, ref], ignore_index=True)
        else:
            base = base.head(int(quantidade))

    base = base.drop_duplicates(chave).sort_values(["Posição Atual", chave]).copy()

    if base.empty:
        st.info("Não há resultados para os filtros selecionados.")
        return

    atual_num = pd.to_numeric(base["Resultado Atual"], errors="coerce")
    inicial_num = pd.to_numeric(base["Resultado Inicial"], errors="coerce")

    if indicador_label == "IDEB":
        escala_min, escala_max = 0.0, 10.0
    else:
        valores = pd.concat([atual_num, inicial_num]).dropna()
        if valores.empty:
            escala_min, escala_max = 0.0, 1.0
        elif indicador_label in ["Língua Portuguesa (SAEB)", "Matemática (SAEB)"]:
            escala_min = max(0.0, float(valores.min()) - 15)
            escala_max = float(valores.max()) + 15
        else:
            escala_min = 0.0
            escala_max = max(10.0, float(valores.max()))

    def pct(v):
        if pd.isna(v) or escala_max <= escala_min:
            return 0
        return max(2, min(100, ((float(v)-escala_min)/(escala_max-escala_min))*100))

    rotulo = cfg["rotulo"]

    header = (
        '<div style="display:grid;grid-template-columns:72px minmax(210px,1.35fr) '
        'minmax(290px,2fr) 105px 115px 115px 130px;gap:14px;padding:9px 14px;'
        'border-bottom:1px solid #D8E0E8;color:#52606D;font-size:11px;font-weight:800;">'
        '<div>POSIÇÃO</div>'
        f'<div>{entidade.upper()}</div>'
        f'<div>{rotulo.upper()} — RESULTADO ATUAL</div>'
        '<div>POS. INICIAL</div><div>RESULT. INICIAL</div>'
        '<div>VARIAÇÃO</div><div>MOVIMENTO</div></div>'
    )
    st.markdown(header, unsafe_allow_html=True)

    for _, r in base.iterrows():
        atual = r.get("Resultado Atual")
        inicial = r.get("Resultado Inicial")
        pa = r.get("Posição Atual")
        pi = r.get("Posição Inicial")
        vp = r.get("Variação de Posição")

        classe, cor = classificacao_visual_ranking(
            indicador_label, atual, r.get("Nível Atual"), r.get("Padrão Atual")
        )

        if pd.isna(vp):
            mov, cor_mov = "—", "#6B7280"
        elif vp > 0:
            mov, cor_mov = f"↑ {int(vp)}", "#169B62"
        elif vp < 0:
            mov, cor_mov = f"↓ {abs(int(vp))}", "#D92D20"
        else:
            mov, cor_mov = "= 0", "#6B7280"

        if pd.notna(atual) and pd.notna(inicial):
            dv = float(atual) - float(inicial)
            sinal = "+" if dv > 0 else ""
            var_txt = f"{sinal}{fmt(dv,cfg['casas'])}"
            cor_var = "#169B62" if dv > 0 else "#D92D20" if dv < 0 else "#374151"
        else:
            var_txt, cor_var = "—", "#6B7280"

        rede = str(r.get("Rede Atual", "") or "") if entidade == "Município" else ""
        pa_txt = "—" if pd.isna(pa) else f"{int(pa)}º"
        pi_txt = "—" if pd.isna(pi) else f"{int(pi)}º"
        border = "2px solid #0E5A70" if destaque_barueri and r[chave] == "Barueri" else "1px solid #E3E8EF"

        row = (
            f'<div style="border:{border};background:white;border-radius:12px;padding:13px 14px;'
            'margin:7px 0;display:grid;grid-template-columns:72px minmax(210px,1.35fr) '
            'minmax(290px,2fr) 105px 115px 115px 130px;gap:14px;align-items:center;">'
            f'<div style="font-size:23px;font-weight:800;">{pa_txt}</div>'
            f'<div><div style="font-size:15px;font-weight:800;">{r[chave]}</div>'
            f'<div style="font-size:11px;color:#6B7280;margin-top:3px;">{rede}</div></div>'
            '<div>'
            '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">'
            f'<span style="background:{cor};color:white;padding:3px 8px;border-radius:5px;font-size:11px;font-weight:800;">{classe}</span>'
            f'<span style="font-size:18px;font-weight:800;">{fmt(atual,cfg["casas"])}</span></div>'
            '<div style="height:10px;background:#E9EDF2;border-radius:6px;overflow:hidden;">'
            f'<div style="height:100%;width:{pct(atual):.1f}%;background:{cor};border-radius:6px;"></div></div></div>'
            f'<div style="font-size:16px;font-weight:700;">{pi_txt}</div>'
            f'<div style="font-size:16px;font-weight:700;">{fmt(inicial,cfg["casas"])}</div>'
            f'<div style="font-size:16px;font-weight:800;color:{cor_var};">{var_txt}</div>'
            f'<div><div style="font-size:17px;font-weight:850;color:{cor_mov};">{mov}</div>'
            f'<div style="font-size:11px;color:#6B7280;">{"de "+pi_txt if pi_txt != "—" else ""}</div></div></div>'
        )
        st.markdown(row, unsafe_allow_html=True)

    if indicador_label == "IDEB":
        legend = (
            '<div style="margin-top:14px;padding:13px 16px;background:white;border:1px solid #E3E8EF;border-radius:12px;">'
            '<div style="font-size:12px;font-weight:800;margin-bottom:10px;">Classificação visual utilizada no ranking do IDEB</div>'
            '<div style="display:flex;gap:24px;flex-wrap:wrap;font-size:12px;">'
            '<span><b style="color:#16A34A;">■</b> Avançado ≥ 6,0</span>'
            '<span><b style="color:#F4B400;">■</b> Adequado 5,0–5,9</span>'
            '<span><b style="color:#2583E9;">■</b> Básico 4,0–4,9</span>'
            '<span><b style="color:#F97316;">■</b> Insuficiente 3,0–3,9</span>'
            '<span><b style="color:#E31B23;">■</b> Crítico &lt; 3,0</span>'
            '</div></div>'
        )
        st.markdown(legend, unsafe_allow_html=True)
    elif indicador_label in ["Língua Portuguesa (SAEB)", "Matemática (SAEB)"]:
        st.caption("As cores correspondem aos níveis de proficiência do Saeb definidos para a etapa e o componente selecionados.")

def opcoes_municipios_busca_ranking(
    etapa,
    rede,
    termo="",
    selecionados=None
):
    """
    Busca sobre Município_Busca, coluna normalizada uma única vez
    dentro de carregar_bases() com @st.cache_data.
    """
    selecionados = selecionados or []

    base = municipios[
        (municipios["Etapa"] == etapa) &
        (municipios["Rede"] == rede) &
        (municipios["Município"] != "Barueri")
    ][["Município", "Município_Busca"]].drop_duplicates()

    termo_norm = normalizar_texto(termo)

    if termo_norm:
        encontrados = base[
            base["Município_Busca"]
            .astype("string")
            .str.contains(
                termo_norm,
                na=False,
                regex=False
            )
        ]["Município"].astype(str).tolist()
    else:
        encontrados = base["Município"].astype(str).tolist()

    # Mantém municípios já selecionados nas opções enquanto
    # uma nova busca é feita.
    return sorted(
        set(encontrados).union(set(selecionados))
    )


def mascara_itb(series):
    """
    Identifica ITBs sem alterar a base original.
    Usa normalização vetorizada apenas sobre a Series recebida.
    """
    s = (
        series.astype("string")
        .str.normalize("NFKD")
        .str.encode("ascii", errors="ignore")
        .str.decode("utf-8")
        .str.lower()
        .str.strip()
    )
    return s.str.startswith("itb", na=False)

def filtrar_itbs_dataframe(df, incluir_itbs=True):
    if incluir_itbs or "Escola" not in df.columns:
        return df
    return df.loc[~mascara_itb(df["Escola"])].copy()

def preparar_ranking_download(comp, entidade, indicador_label, quantidade="Todos"):
    chave = "Município" if entidade == "Município" else "Escola"
    cfg = RANKING_INDICADORES[indicador_label]
    base = comp.copy()

    if quantidade != "Todos":
        if entidade == "Município" and "_Referencia" in base.columns:
            ref = base[base["_Referencia"] == True]
            outros = base[base["_Referencia"] != True].head(int(quantidade))
            base = pd.concat([outros, ref], ignore_index=True)
        else:
            base = base.head(int(quantidade))

    base = base.drop_duplicates(chave).sort_values(["Posição Atual", chave]).copy()

    cols = [chave]
    if entidade == "Município" and "Rede Atual" in base.columns:
        cols.append("Rede Atual")

    cols += [
        "Posição Atual", "Resultado Atual",
        "Posição Inicial", "Resultado Inicial",
        "Variação de Posição", "Movimento"
    ]

    if "Nível Atual" in base.columns:
        cols.append("Nível Atual")
    if "Padrão Atual" in base.columns:
        cols.append("Padrão Atual")

    cols = [c for c in cols if c in base.columns]
    out = base[cols].copy()
    out = out.rename(columns={
        "Resultado Atual": cfg["rotulo"] + " Atual",
        "Resultado Inicial": cfg["rotulo"] + " Inicial",
    })
    return out

def botao_download_ranking(comp, entidade, indicador_label, quantidade, sufixo):
    tabela = preparar_ranking_download(
        comp, entidade, indicador_label, quantidade
    )
    csv = tabela.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig")

    st.download_button(
        "⬇️ Baixar dados do ranking (CSV)",
        data=csv,
        file_name=f"ranking_{sufixo}.csv",
        mime="text/csv",
        key=f"download_csv_{sufixo}",
        use_container_width=False
    )

def grafico_ranking_download(comp, entidade, indicador_label, quantidade="Todos"):
    """
    Versão gráfica do ranking para exportação em PNG pelo menu nativo do Plotly.
    As barras usam as mesmas cores/classificações do ranking visual.
    """
    chave = "Município" if entidade == "Município" else "Escola"
    cfg = RANKING_INDICADORES[indicador_label]
    base = comp.copy()

    if quantidade != "Todos":
        if entidade == "Município" and "_Referencia" in base.columns:
            ref = base[base["_Referencia"] == True]
            outros = base[base["_Referencia"] != True].head(int(quantidade))
            base = pd.concat([outros, ref], ignore_index=True)
        else:
            base = base.head(int(quantidade))

    base = base.drop_duplicates(chave).sort_values("Posição Atual", ascending=False).copy()

    cores = []
    classes = []
    for _, r in base.iterrows():
        classe, cor = classificacao_visual_ranking(
            indicador_label,
            r.get("Resultado Atual"),
            r.get("Nível Atual"),
            r.get("Padrão Atual")
        )
        cores.append(cor)
        classes.append(classe)

    base["_Classe"] = classes

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=base["Resultado Atual"],
        y=base[chave],
        orientation="h",
        marker_color=cores,
        text=[
            f'{fmt(v,cfg["casas"])} • {cl}'
            for v, cl in zip(base["Resultado Atual"], base["_Classe"])
        ],
        textposition="outside",
        customdata=np.column_stack([
            base["Posição Atual"].fillna(np.nan),
            base["Posição Inicial"].fillna(np.nan),
            base["Variação de Posição"].fillna(np.nan),
        ]),
        hovertemplate=(
            "<b>%{y}</b><br>"
            + cfg["rotulo"] + ": %{x}<br>"
            "Posição atual: %{customdata[0]}<br>"
            "Posição inicial: %{customdata[1]}<br>"
            "Variação de posição: %{customdata[2]}"
            "<extra></extra>"
        )
    ))

    fig.update_layout(
        title=dict(
            text=f"Ranking — {cfg['rotulo']}",
            x=.01, xanchor="left"
        ),
        height=max(430, 42 * len(base) + 110),
        margin=dict(l=190, r=150, t=65, b=45),
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis_title=cfg["rotulo"],
        yaxis_title="",
        showlegend=False
    )
    fig.update_xaxes(gridcolor="#EDF1F4")
    return fig





def cores_niveis(etapa):
    return CORES_NIVEIS_FI if etapa == "Fundamental I" else CORES_NIVEIS_FII


def colunas_disciplina(disciplina):
    if disciplina == "Matemática":
        return "Matemática", "Nível Matemática", "Padrão Matemática"
    return "Língua Portuguesa", "Nível Língua Portuguesa", "Padrão Língua Portuguesa"


def texto_movimento(valor):
    if pd.isna(valor):
        return "—"
    valor = int(valor)
    if valor > 0:
        return f"↑ {valor}"
    if valor < 0:
        return f"↓ {abs(valor)}"
    return "= 0"


def texto_mudanca_nivel(valor):
    if pd.isna(valor):
        return "—"
    valor = int(valor)
    if valor > 0:
        return f"↑ {valor} nível" if valor == 1 else f"↑ {valor} níveis"
    if valor < 0:
        n = abs(valor)
        return f"↓ {n} nível" if n == 1 else f"↓ {n} níveis"
    return "manteve"


def cor_padrao(padrao):
    return {
        "Abaixo do básico": "#D32F2F",
        "Básico": "#FFA726",
        "Adequado": "#66BB6A",
        "Avançado": "#1B5E20",
    }.get(padrao, "#BDBDBD")


def painel_padrao_disciplina(disciplina, etapa):
    """
    Exibe a escala de interpretação usada no projeto para a disciplina e etapa.
    Não mostra código de cor; mostra faixa, padrão e significado.
    """
    st.markdown(
        f'<div class="section-title">{disciplina} — {etapa}</div>',
        unsafe_allow_html=True
    )

    colunas = st.columns(4)

    for col, (padrao, faixa) in zip(
        colunas,
        PADROES_DESEMPENHO[disciplina][etapa]
    ):
        cor = cor_padrao(padrao)
        significado = EXPLICACAO_PADROES[padrao]

        col.markdown(
            f'<div class="metric-card" style="border-top:5px solid {cor};min-height:210px;">'
            f'<div class="metric-label">{padrao}</div>'
            f'<div class="metric-value" style="font-size:20px;">{faixa}</div>'
            f'<div class="metric-foot" style="font-size:13px;line-height:1.45;">'
            f'{significado}</div>'
            f'</div>',
            unsafe_allow_html=True
        )


def localizar_padrao(valor, disciplina, etapa):
    if pd.isna(valor):
        return None

    valor = float(valor)

    if disciplina == "Língua Portuguesa":
        if etapa == "Fundamental I":
            if valor <= 179:
                return "Abaixo do básico"
            if valor <= 219:
                return "Básico"
            if valor <= 259:
                return "Adequado"
            return "Avançado"

        if valor <= 224:
            return "Abaixo do básico"
        if valor <= 274:
            return "Básico"
        if valor <= 324:
            return "Adequado"
        return "Avançado"

    if etapa == "Fundamental I":
        if valor <= 204:
            return "Abaixo do básico"
        if valor <= 244:
            return "Básico"
        if valor <= 284:
            return "Adequado"
        return "Avançado"

    if valor <= 224:
        return "Abaixo do básico"
    if valor <= 299:
        return "Básico"
    if valor <= 349:
        return "Adequado"
    return "Avançado"


def card_situacao_atual(disciplina, valor, nivel, etapa, padrao_base=None):
    padrao = padrao_base
    if pd.isna(padrao) or str(padrao).strip() in {"", "<NA>", "nan"}:
        padrao = localizar_padrao(valor, disciplina, etapa)

    cor = cor_padrao(padrao)

    valor_txt = fmt(valor, 2)
    nivel_txt = "—" if pd.isna(nivel) else f"Nível {int(float(nivel))}"

    return (
        f'<div class="metric-card" style="border-top:5px solid {cor};min-height:160px;">'
        f'<div class="metric-label">{disciplina}</div>'
        f'<div class="metric-value">{valor_txt}</div>'
        f'<div class="metric-foot"><b>{nivel_txt}</b> • {padrao or "—"}</div>'
        f'</div>'
    )


def legenda_niveis_html(etapa):
    """
    Gera a legenda em HTML sem indentação inicial.
    Isso evita que o Markdown interprete o HTML como bloco de código.
    """
    cores = cores_niveis(etapa)
    blocos = []

    for nivel, faixa in FAIXAS_NIVEIS[etapa]:
        blocos.append(
            f'<div style="display:flex;align-items:center;gap:8px;'
            f'border:1px solid #E5E7EB;background:white;'
            f'border-radius:9px;padding:7px 9px;">'
            f'<span style="width:14px;height:14px;border-radius:3px;'
            f'background:{cores[nivel]};display:inline-block;"></span>'
            f'<span style="font-weight:700;font-size:12px;">Nível {nivel}</span>'
            f'<span style="font-size:11px;color:#6B7280;">{faixa}</span>'
            f'</div>'
        )

    return (
        '<div style="display:grid;'
        'grid-template-columns:repeat(auto-fit,minmax(170px,1fr));'
        'gap:7px;margin:8px 0 14px 0;">'
        + "".join(blocos)
        + '</div>'
    )


def ranking_municipios_ano(etapa, ano, disciplina, rede_comparacao):
    """
    Ranking municipal com filtro EXPLÍCITO de rede.

    Importante:
    - os municípios comparados pertencem somente à rede selecionada;
    - não há prioridade automática entre redes;
    - Barueri (Municipal) é tratada separadamente como referência.
    """
    coluna, coluna_nivel, coluna_padrao = colunas_disciplina(disciplina)

    x = municipios[
        (municipios["Etapa"] == etapa) &
        (municipios["Ano"] == ano) &
        (municipios["Rede"] == rede_comparacao)
    ].copy()

    x[coluna] = pd.to_numeric(x[coluna], errors="coerce")
    x = x.dropna(subset=[coluna])

    # Mantém um registro por município dentro da MESMA rede.
    x = (
        x.sort_values(
            ["Município", coluna],
            ascending=[True, False]
        )
        .drop_duplicates("Município", keep="first")
        .copy()
    )

    x["Posição"] = (
        x[coluna]
        .rank(method="min", ascending=False)
        .astype("Int64")
    )

    return (
        x[
            [
                "Município",
                "Rede",
                coluna,
                coluna_nivel,
                coluna_padrao,
                "Posição",
            ]
        ]
        .sort_values(["Posição", "Município"])
        .reset_index(drop=True)
    )


def referencia_barueri_ano(etapa, ano, disciplina):
    """
    Retorna Barueri SEMPRE como Rede Municipal,
    independentemente da rede usada para comparar os demais municípios.
    """
    coluna, coluna_nivel, coluna_padrao = colunas_disciplina(disciplina)

    x = municipios[
        (municipios["Município"] == "Barueri") &
        (municipios["Rede"] == "Municipal") &
        (municipios["Etapa"] == etapa) &
        (municipios["Ano"] == ano)
    ].copy()

    if x.empty:
        return None

    x[coluna] = pd.to_numeric(x[coluna], errors="coerce")
    x = x.dropna(subset=[coluna])

    if x.empty:
        return None

    r = x.sort_values(coluna, ascending=False).iloc[0]

    return {
        "Município": "Barueri",
        "Rede": "Municipal",
        "Pontuação": r.get(coluna),
        "Nível": r.get(coluna_nivel),
        "Padrão": r.get(coluna_padrao),
    }


def posicao_referencia_no_universo(valor_referencia, ranking_rede, disciplina):
    """
    Calcula em qual posição a pontuação de Barueri Municipal ficaria
    dentro do universo da rede selecionada.
    """
    if valor_referencia is None or pd.isna(valor_referencia):
        return pd.NA

    coluna, _, _ = colunas_disciplina(disciplina)

    valores = pd.to_numeric(
        ranking_rede[coluna],
        errors="coerce"
    ).dropna()

    if valores.empty:
        return pd.NA

    # rank method=min: 1 + quantidade de valores estritamente maiores.
    return int((valores > float(valor_referencia)).sum() + 1)


def ranking_escolas_ano(etapa, ano, disciplina):
    coluna, coluna_nivel, coluna_padrao = colunas_disciplina(disciplina)

    x = escolas[
        (escolas["Etapa"] == etapa) &
        (escolas["Ano"] == ano)
    ].copy()

    x[coluna] = pd.to_numeric(x[coluna], errors="coerce")
    x = x.dropna(subset=[coluna])

    # Caso alguma unidade apareça duplicada no mesmo ano/etapa,
    # preserva o registro com valor válido e maior proficiência.
    x = (
        x.sort_values(
            ["Escola", coluna],
            ascending=[True, False]
        )
        .drop_duplicates("Escola", keep="first")
        .copy()
    )

    x["Posição"] = (
        x[coluna]
        .rank(method="min", ascending=False)
        .astype("Int64")
    )

    return (
        x[
            [
                "Escola",
                coluna,
                coluna_nivel,
                coluna_padrao,
                "Posição",
            ]
        ]
        .sort_values(["Posição", "Escola"])
        .reset_index(drop=True)
    )


def comparar_rankings(
    entidade,
    etapa,
    disciplina,
    ano_ini,
    ano_fim,
    rede_comparacao=None
):
    coluna, coluna_nivel, coluna_padrao = colunas_disciplina(disciplina)

    if entidade == "Município":
        if rede_comparacao is None:
            raise ValueError("rede_comparacao é obrigatória para ranking de municípios.")

        ini = ranking_municipios_ano(
            etapa,
            ano_ini,
            disciplina,
            rede_comparacao
        )
        fim = ranking_municipios_ano(
            etapa,
            ano_fim,
            disciplina,
            rede_comparacao
        )
        chave = "Município"

        ini = ini[
            [chave, "Rede", coluna, coluna_nivel, coluna_padrao, "Posição"]
        ].copy()
        fim = fim[
            [chave, "Rede", coluna, coluna_nivel, coluna_padrao, "Posição"]
        ].copy()

    else:
        ini = ranking_escolas_ano(etapa, ano_ini, disciplina)
        fim = ranking_escolas_ano(etapa, ano_fim, disciplina)

        if not incluir_itbs:
            ini = ini.loc[~mascara_itb(ini["Escola"])].copy()
            fim = fim.loc[~mascara_itb(fim["Escola"])].copy()

            # Recalcula as posições dentro do universo sem ITBs.
            coluna_disc, _, _ = colunas_disciplina(disciplina)
            ini["Posição"] = (
                pd.to_numeric(ini[coluna_disc], errors="coerce")
                .rank(method="min", ascending=False)
                .astype("Int64")
            )
            fim["Posição"] = (
                pd.to_numeric(fim[coluna_disc], errors="coerce")
                .rank(method="min", ascending=False)
                .astype("Int64")
            )
        chave = "Escola"

        ini = ini[
            [chave, coluna, coluna_nivel, coluna_padrao, "Posição"]
        ].copy()
        fim = fim[
            [chave, coluna, coluna_nivel, coluna_padrao, "Posição"]
        ].copy()

    ren_ini = {
        coluna: "Pontuação Inicial",
        coluna_nivel: "Nível Inicial",
        coluna_padrao: "Padrão Inicial",
        "Posição": "Posição Inicial",
    }

    ren_fim = {
        coluna: "Pontuação Atual",
        coluna_nivel: "Nível Atual",
        coluna_padrao: "Padrão Atual",
        "Posição": "Posição Atual",
    }

    if entidade == "Município":
        ren_ini["Rede"] = "Rede Inicial"
        ren_fim["Rede"] = "Rede Atual"

    ini = ini.rename(columns=ren_ini)
    fim = fim.rename(columns=ren_fim)

    comp = fim.merge(
        ini,
        on=chave,
        how="left"
    )

    comp["Variação de Posição"] = (
        comp["Posição Inicial"] - comp["Posição Atual"]
    )

    comp["Variação de Nível"] = (
        pd.to_numeric(comp["Nível Atual"], errors="coerce") -
        pd.to_numeric(comp["Nível Inicial"], errors="coerce")
    )

    comp["Movimento"] = comp["Variação de Posição"].apply(texto_movimento)
    comp["Mudança de Nível"] = comp["Variação de Nível"].apply(texto_mudanca_nivel)

    comp["Nível Atual Texto"] = comp["Nível Atual"].apply(
        lambda x: "—" if pd.isna(x) else f"Nível {int(x)}"
    )

    comp["Nível Inicial Texto"] = comp["Nível Inicial"].apply(
        lambda x: "—" if pd.isna(x) else f"Nível {int(x)}"
    )

    return comp.sort_values(
        ["Posição Atual", chave]
    ).reset_index(drop=True)


def adicionar_barueri_referencia(
    comp,
    etapa,
    disciplina,
    ano_ini,
    ano_fim,
    rede_comparacao
):
    """
    Adiciona Barueri Municipal à tabela comparativa como referência.
    Sua posição é calculada dentro do universo da rede escolhida.
    """
    bar_ini = referencia_barueri_ano(
        etapa,
        ano_ini,
        disciplina
    )
    bar_fim = referencia_barueri_ano(
        etapa,
        ano_fim,
        disciplina
    )

    if bar_fim is None:
        return comp

    rank_ini = ranking_municipios_ano(
        etapa,
        ano_ini,
        disciplina,
        rede_comparacao
    )
    rank_fim = ranking_municipios_ano(
        etapa,
        ano_fim,
        disciplina,
        rede_comparacao
    )

    pos_ini = (
        posicao_referencia_no_universo(
            bar_ini["Pontuação"],
            rank_ini,
            disciplina
        )
        if bar_ini is not None
        else pd.NA
    )

    pos_fim = posicao_referencia_no_universo(
        bar_fim["Pontuação"],
        rank_fim,
        disciplina
    )

    nivel_ini = (
        bar_ini["Nível"]
        if bar_ini is not None
        else pd.NA
    )

    nivel_fim = bar_fim["Nível"]

    variacao_pos = (
        pos_ini - pos_fim
        if pd.notna(pos_ini) and pd.notna(pos_fim)
        else pd.NA
    )

    variacao_nivel = (
        float(nivel_fim) - float(nivel_ini)
        if pd.notna(nivel_ini) and pd.notna(nivel_fim)
        else pd.NA
    )

    linha = {
        "Município": "Barueri",
        "Rede Atual": "Municipal (referência)",
        "Pontuação Atual": bar_fim["Pontuação"],
        "Nível Atual": nivel_fim,
        "Padrão Atual": bar_fim["Padrão"],
        "Posição Atual": pos_fim,
        "Rede Inicial": "Municipal (referência)",
        "Pontuação Inicial": (
            bar_ini["Pontuação"]
            if bar_ini is not None
            else pd.NA
        ),
        "Nível Inicial": nivel_ini,
        "Padrão Inicial": (
            bar_ini["Padrão"]
            if bar_ini is not None
            else pd.NA
        ),
        "Posição Inicial": pos_ini,
        "Variação de Posição": variacao_pos,
        "Variação de Nível": variacao_nivel,
        "Movimento": texto_movimento(variacao_pos),
        "Mudança de Nível": texto_mudanca_nivel(variacao_nivel),
        "Nível Atual Texto": (
            "—" if pd.isna(nivel_fim)
            else f"Nível {int(float(nivel_fim))}"
        ),
        "Nível Inicial Texto": (
            "—" if pd.isna(nivel_ini)
            else f"Nível {int(float(nivel_ini))}"
        ),
        "_Referencia": True,
    }

    comp2 = comp.copy()
    comp2["_Referencia"] = False

    # Evita duplicar Barueri se a rede selecionada também for Municipal.
    comp2 = comp2[
        comp2["Município"] != "Barueri"
    ]

    comp2 = pd.concat(
        [comp2, pd.DataFrame([linha])],
        ignore_index=True
    )

    return comp2.sort_values(
        ["Posição Atual", "Município"]
    ).reset_index(drop=True)


def grafico_ranking_movimento(
    comp,
    entidade,
    etapa,
    disciplina,
    ano_ini,
    ano_fim,
    quantidade=20,
    destaque=None,
    rede_comparacao=None
):
    chave = "Município" if entidade == "Município" else "Escola"
    cores = cores_niveis(etapa)

    base_plot = comp.copy()

    if quantidade == "Todos":
        plot = base_plot.copy()
    else:
        if entidade == "Município" and "_Referencia" in base_plot.columns:
            referencia = base_plot[
                base_plot["_Referencia"] == True
            ].copy()

            universo = base_plot[
                base_plot["_Referencia"] != True
            ].copy()

            plot = universo.head(int(quantidade)).copy()

            if not referencia.empty:
                plot = pd.concat(
                    [plot, referencia],
                    ignore_index=True
                )
        else:
            plot = base_plot.head(int(quantidade)).copy()

    if entidade == "Escola" and destaque and destaque != "Nenhuma":
        if destaque in base_plot[chave].values and destaque not in plot[chave].values:
            plot = pd.concat(
                [plot, base_plot[base_plot[chave] == destaque]],
                ignore_index=True
            )

    plot = (
        plot.drop_duplicates(chave)
        .sort_values("Posição Atual", ascending=False)
        .copy()
    )

    plot["_Cor"] = plot["Nível Atual"].apply(
        lambda x: "#BDBDBD"
        if pd.isna(x)
        else cores.get(int(x), "#BDBDBD")
    )

    plot["_Texto"] = plot.apply(
        lambda r:
        f'{int(r["Posição Atual"])}º | {r["Movimento"]} | {r["Nível Atual Texto"]}',
        axis=1
    )

    custom = np.column_stack([
        plot["Pontuação Atual"].fillna(np.nan),
        plot["Posição Inicial"].fillna(np.nan),
        plot["Posição Atual"].fillna(np.nan),
        plot["Movimento"].fillna("—"),
        plot["Nível Inicial Texto"].fillna("—"),
        plot["Nível Atual Texto"].fillna("—"),
        plot["Mudança de Nível"].fillna("—"),
        plot["Padrão Atual"].fillna("—"),
        (
            plot["Rede Atual"].fillna("—")
            if "Rede Atual" in plot.columns
            else pd.Series(["—"] * len(plot))
        ),
    ])

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=plot["Pontuação Atual"],
            y=plot[chave],
            orientation="h",
            marker_color=plot["_Cor"],
            text=plot["_Texto"],
            textposition="outside",
            customdata=custom,
            hovertemplate=(
                "<b>%{y}</b><br>"
                f"{disciplina}: %{{customdata[0]:.1f}}<br>"
                f"Posição em {ano_ini}: %{{customdata[1]}}<br>"
                f"Posição em {ano_fim}: %{{customdata[2]}}<br>"
                "Movimento: %{customdata[3]}<br>"
                "Nível inicial: %{customdata[4]}<br>"
                "Nível atual: %{customdata[5]}<br>"
                "Mudança de nível: %{customdata[6]}<br>"
                "Padrão atual: %{customdata[7]}<br>"
                "Rede: %{customdata[8]}"
                "<extra></extra>"
            )
        )
    )

    titulo_entidade = (
        "municípios"
        if entidade == "Município"
        else "escolas"
    )

    subtitulo_rede = (
        f" • Rede de comparação: {rede_comparacao}"
        if entidade == "Município" and rede_comparacao
        else ""
    )

    fig.update_layout(
        title=dict(
            text=(
                f"Ranking de {titulo_entidade} — {disciplina} — "
                f"{etapa} — {ano_fim}{subtitulo_rede}"
            ),
            x=.01,
            xanchor="left",
            font=dict(size=18)
        ),
        height=max(520, len(plot) * 34),
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=40, r=180, t=85, b=45),
        showlegend=False,
        xaxis_title=f"Proficiência em {disciplina}",
        yaxis_title="",
        font=dict(
            family="Arial, sans-serif",
            color="#26313b"
        ),
    )

    fig.update_xaxes(
        gridcolor="#EDF1F4",
        zeroline=False
    )

    fig.update_yaxes(
        showgrid=False
    )

    return fig


def tabela_ranking_exibicao(comp, entidade):
    chave = "Município" if entidade == "Município" else "Escola"

    cols = [
        chave,
        "Posição Inicial",
        "Posição Atual",
        "Movimento",
        "Pontuação Inicial",
        "Pontuação Atual",
        "Nível Inicial Texto",
        "Nível Atual Texto",
        "Mudança de Nível",
        "Padrão Atual",
    ]

    if entidade == "Município":
        cols.insert(1, "Rede Atual")

    out = comp[cols].copy()

    out = out.rename(columns={
        "Nível Inicial Texto": "Nível inicial",
        "Nível Atual Texto": "Nível atual",
        "Padrão Atual": "Padrão atual",
    })

    return out


def media_rede_recorte(etapa, ano, indicador, rede):
    """
    Média simples dos municípios com resultado publicado na combinação
    etapa + ano + indicador + rede. É um recorte analítico do painel,
    não uma média oficial ponderada por matrícula.
    """
    x = municipios[
        (municipios["Etapa"] == etapa) &
        (municipios["Ano"] == ano) &
        (municipios["Rede"] == rede)
    ].copy()

    x[indicador] = pd.to_numeric(x[indicador], errors="coerce")
    x = x.dropna(subset=[indicador])

    # Um valor por município dentro da mesma rede.
    x = (
        x.sort_values(["Município", indicador], ascending=[True, False])
        .drop_duplicates("Município", keep="first")
    )

    if x.empty:
        return np.nan, 0

    return float(x[indicador].mean()), int(x["Município"].nunique())


def historico_rede_barueri(etapa):
    x = dados_municipio("Barueri", etapa, rede="Municipal").copy()
    cols = ["Ano","IDEB","Aprovação Geral","Língua Portuguesa","Matemática","Meta IDEB"]
    return x[[c for c in cols if c in x.columns]].sort_values("Ano")


def fig_escala_saeb(row, etapa, disciplina, anos_referencia=None):
    """
    Escala visual por níveis. O resultado fica abaixo da barra,
    evitando cobrir os níveis da escala.
    """
    coluna, coluna_nivel, _ = colunas_disciplina(disciplina)
    valor = row.get(coluna)
    nivel = row.get(coluna_nivel)

    faixas = FAIXAS_NIVEIS[etapa]
    niveis = [n for n, _ in faixas]
    maxn = max(niveis) if niveis else 1

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=[1] * len(niveis),
        y=["Escala"] * len(niveis),
        orientation="h",
        marker_color=[
            cores_niveis(etapa).get(n, "#BDBDBD")
            for n in niveis
        ],
        text=[str(n) for n in niveis],
        textposition="inside",
        insidetextanchor="middle",
        hovertext=[
            f"Nível {n} • {faixa}"
            for n, faixa in faixas
        ],
        hovertemplate="%{hovertext}<extra></extra>",
        showlegend=False
    ))

    if pd.notna(valor) and pd.notna(nivel):
        n = int(float(nivel))
        xpos = (n + .5) / (maxn + 1)

        # Linha discreta aponta para a posição; o rótulo fica abaixo.
        fig.add_shape(
            type="line",
            x0=xpos, x1=xpos,
            y0=.38, y1=.60,
            xref="paper", yref="paper",
            line=dict(color="#0E5A70", width=3)
        )

        fig.add_annotation(
            x=xpos,
            y=.03,
            xref="paper",
            yref="paper",
            text=f"<b>{fmt(valor,2)}</b> • Nível {n}",
            showarrow=False,
            bgcolor="#0E5A70",
            bordercolor="#0E5A70",
            borderpad=6,
            font=dict(color="white", size=11)
        )

    fig.update_layout(
        barmode="stack",
        height=210,
        margin=dict(l=20, r=20, t=40, b=55),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        paper_bgcolor="white",
        plot_bgcolor="white",
        title=dict(
            text=f"{disciplina} — {etapa}",
            x=.01,
            xanchor="left",
            font=dict(size=16)
        )
    )

    return fig


def resultado_evolucao_barueri(etapa, disciplina, anos=(2019, 2023, 2025)):
    coluna, coluna_nivel, _ = colunas_disciplina(disciplina)
    base = dados_municipio("Barueri", etapa, rede="Municipal")
    out = []
    for ano in anos:
        r = ultima_linha(base, ano)
        if r is not None and pd.notna(r.get(coluna)):
            out.append({
                "Ano": ano,
                "Valor": float(r.get(coluna)),
                "Nível": r.get(coluna_nivel)
            })
    return pd.DataFrame(out)


def fig_matriz_nivel_tendencia(
    df,
    entidade,
    disciplina,
    ano_ini,
    ano_fim,
    etapa,
    incluir_itbs=True
):
    chave = "Município" if entidade == "Município" else "Escola"
    coluna, coluna_nivel, _ = colunas_disciplina(disciplina)

    if entidade == "Município":
        ini = ranking_municipios_ano(etapa, ano_ini, disciplina, "Municipal")
        fim = ranking_municipios_ano(etapa, ano_fim, disciplina, "Municipal")
    else:
        ini = ranking_escolas_ano(etapa, ano_ini, disciplina)
        fim = ranking_escolas_ano(etapa, ano_fim, disciplina)

    ini = ini[[chave, coluna]].rename(columns={coluna:"Valor Inicial"})
    fim = fim[[chave, coluna, coluna_nivel]].rename(
        columns={coluna:"Valor Atual", coluna_nivel:"Nível Atual"}
    )
    comp = fim.merge(ini, on=chave, how="left")
    comp["Variação"] = comp["Valor Atual"] - comp["Valor Inicial"]

    def classe(v):
        if pd.isna(v):
            return "Sem comparação"
        if v > 0.5:
            return "Avanço"
        if v < -0.5:
            return "Recuo"
        return "Estabilidade"

    comp["Tendência"] = comp["Variação"].apply(classe)
    mapa_cores = {
        "Avanço":"#2E8B57",
        "Estabilidade":"#7A8490",
        "Recuo":"#C44747",
        "Sem comparação":"#B8C0C8",
    }

    fig = go.Figure()
    for cat in ["Avanço","Estabilidade","Recuo","Sem comparação"]:
        s = comp[comp["Tendência"] == cat]
        if s.empty:
            continue
        fig.add_trace(go.Scatter(
            x=s["Valor Atual"],
            y=s["Variação"],
            mode="markers",
            name=cat,
            marker=dict(size=9, color=mapa_cores[cat]),
            text=s[chave],
            customdata=np.column_stack([
                s["Nível Atual"].fillna(np.nan),
                s["Valor Inicial"].fillna(np.nan),
                s["Valor Atual"].fillna(np.nan)
            ]),
            hovertemplate=(
                "<b>%{text}</b><br>"
                f"{disciplina} {ano_ini}: %{{customdata[1]:.2f}}<br>"
                f"{disciplina} {ano_fim}: %{{customdata[2]:.2f}}<br>"
                "Variação: %{y:.2f}<br>"
                "Nível atual: %{customdata[0]}"
                "<extra></extra>"
            )
        ))

    fig.add_hline(y=0, line_dash="dash", line_color="#8A94A0")
    mediana = comp["Valor Atual"].median()
    if pd.notna(mediana):
        fig.add_vline(x=mediana, line_dash="dash", line_color="#7FA1B0")

    fig.update_layout(
        title=dict(
            text=f"Matriz nível × tendência — {disciplina} — {etapa}",
            x=.01, xanchor="left", font=dict(size=18)
        ),
        height=520,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=55,r=30,t=70,b=50),
        xaxis_title=f"{disciplina} {ano_fim} (resultado atual)",
        yaxis_title=f"Variação {ano_ini} → {ano_fim}",
        legend=dict(orientation="h", y=1.08, x=0),
    )
    fig.update_xaxes(gridcolor="#EDF1F4")
    fig.update_yaxes(gridcolor="#EDF1F4")
    return fig, comp


def cards_movimento(comp):
    total = len(comp)
    avanc = int((comp["Tendência"] == "Avanço").sum())
    est = int((comp["Tendência"] == "Estabilidade").sum())
    rec = int((comp["Tendência"] == "Recuo").sum())
    sem = int((comp["Tendência"] == "Sem comparação").sum())

    cols = st.columns(4)
    dados = [
        ("↑ Avanço", avanc, "#D9F1E2"),
        ("— Estabilidade", est, "#E7ECF2"),
        ("↓ Recuo", rec, "#F7D8D5"),
        ("Sem comparação", sem, "#E4EDF2"),
    ]
    for col, (rot, val, cor) in zip(cols, dados):
        col.markdown(
            f'<div class="metric-card" style="background:{cor};">'
            f'<div class="metric-label">{rot}</div>'
            f'<div class="metric-value">{val}</div>'
            f'<div class="metric-foot">de {total} unidades consideradas</div>'
            f'</div>',
            unsafe_allow_html=True
        )


def adicionar_rotulos_acima(fig, x, y, casas=1, cor="#26313b", yshift=14):
    """
    Adiciona os valores sempre acima dos pontos.
    Usa yshift positivo e fundo branco translúcido para evitar
    que o número fique sobre a própria linha ou sobre outra série.
    """
    for xv, yv in zip(list(x), list(y)):
        if pd.isna(yv):
            continue
        fig.add_annotation(
            x=xv,
            y=float(yv),
            text=fmt(yv, casas),
            showarrow=False,
            yshift=yshift,
            xanchor="center",
            yanchor="bottom",
            font=dict(size=11, color=cor),
            bgcolor="rgba(255,255,255,0.82)",
            borderpad=1
        )


def grafico_comparativo_referencias_historico(df_barueri, etapa):
    """
    Evolução do IDEB no estilo visual do QEdu.

    Fundamental I:
    - São Paulo e Brasil: série temporária transcrita da visualização QEdu
      usada como referência no projeto.

    Fundamental II:
    - São Paulo e Brasil: série histórica de referência MEC/Inep.
      2025 usa os resultados divulgados em agosto de 2026.
    """
    cores = {
        "Barueri": "#22305B",
        "São Paulo": "#F2B800",
        "Brasil": "#58B7CF",
    }

    if etapa == "Fundamental I":
        anos = [2007, 2009, 2011, 2013, 2015, 2017, 2019, 2021, 2023, 2025]
        sp = [5.0, 5.5, 5.6, 6.1, 6.4, 6.6, 6.7, 6.3, 6.5, 6.6]
        brasil = [4.2, 4.6, 5.0, 5.2, 5.5, 5.8, 5.9, 5.8, 6.0, 6.3]
        y_min, y_max = 4.0, 7.2
        fonte = (
            "São Paulo e Brasil: referências temporárias da série exibida pelo QEdu; "
            "Barueri: base do Inep utilizada neste projeto."
        )
    else:
        anos = [2005, 2007, 2009, 2011, 2013, 2015, 2017, 2019, 2021, 2023, 2025]

        # Referência estadual total de São Paulo.
        # Série 2005–2023 reconstruída a partir dos resultados MEC/Inep;
        # 2025 = 5,5 divulgado pelo Inep em agosto de 2026.
        sp = [4.2, 4.4, 4.5, 4.7, 4.7, 5.0, 5.2, 5.4, 5.5, 5.4, 5.5]

        # Brasil total — anos finais.
        brasil = [3.5, 3.8, 4.0, 4.1, 4.2, 4.5, 4.7, 4.9, 5.1, 5.0, 5.3]

        y_min, y_max = 3.2, 6.2
        fonte = (
            "São Paulo e Brasil: série histórica MEC/Inep; "
            "Barueri: Rede Municipal, base do Inep utilizada neste projeto."
        )

    bar_map = (
        df_barueri[["Ano", "IDEB"]]
        .dropna(subset=["Ano"])
        .assign(Ano=lambda x: x["Ano"].astype(int))
        .set_index("Ano")["IDEB"]
        .to_dict()
    )
    barueri = [bar_map.get(a, np.nan) for a in anos]

    fig = go.Figure()

    series = [
        ("Barueri", barueri, cores["Barueri"], 16),
        ("São Paulo", sp, cores["São Paulo"], 30),
        ("Brasil", brasil, cores["Brasil"], 44),
    ]

    # Sem texto embutido na própria trace:
    # os valores são adicionados depois como annotations, sempre acima.
    for nome, valores, cor, deslocamento in series:
        fig.add_trace(go.Scatter(
            x=anos,
            y=valores,
            mode="lines+markers",
            name=nome,
            line=dict(color=cor, width=3),
            marker=dict(size=8, color=cor),
            hovertemplate=(
                f"<b>{nome}</b><br>Ano: %{{x}}<br>IDEB: %{{y:.1f}}<extra></extra>"
            )
        ))
        adicionar_rotulos_acima(
            fig,
            anos,
            valores,
            casas=1,
            cor=cor,
            yshift=deslocamento
        )

    fig.update_layout(
        title=dict(
            text=f"Evolução do IDEB — Barueri × São Paulo × Brasil — {etapa}",
            x=.01,
            xanchor="left"
        ),
        height=520,
        margin=dict(l=55, r=35, t=95, b=65),
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.22,
            xanchor="center",
            x=.5
        ),
        xaxis=dict(
            title="",
            tickmode="array",
            tickvals=anos,
            gridcolor="#F3F5F7"
        ),
        yaxis=dict(
            title="IDEB",
            range=[y_min, y_max],
            dtick=.4,
            gridcolor="#E7EBEF",
            zeroline=False
        )
    )

    return fig, fonte



def grafico_executivo_ideb_rede(fi, fii):
    fig = go.Figure()

    series = [
        (fi, "Fundamental I", "#0E5A70", 16),
        (fii, "Fundamental II", "#7C3AED", 32),
    ]

    for df, etapa, cor, deslocamento in series:
        d = df.copy().sort_values("Ano")

        fig.add_trace(go.Scatter(
            x=d["Ano"],
            y=d["IDEB"],
            mode="lines+markers",
            name=f"IDEB — {etapa}",
            line=dict(color=cor, width=3),
            marker=dict(size=8)
        ))

        adicionar_rotulos_acima(
            fig, d["Ano"], d["IDEB"],
            casas=1, cor=cor, yshift=deslocamento
        )

        if "Meta IDEB" in d.columns and d["Meta IDEB"].notna().any():
            fig.add_trace(go.Scatter(
                x=d["Ano"],
                y=d["Meta IDEB"],
                mode="lines+markers",
                name=f"Meta — {etapa}",
                line=dict(color=cor, width=1.8, dash="dot"),
                marker=dict(size=6),
                opacity=.72
            ))

    estilo_fig(
        fig,
        "Evolução do IDEB e metas — Rede Municipal de Barueri",
        "IDEB",
        540
    )
    fig.update_layout(margin=dict(l=35, r=35, t=95, b=45))
    fig.update_yaxes(rangemode="tozero")
    return fig


def card_executivo_variacao(row_atual, row_anterior, etapa):
    indicadores = [
        ("IDEB", "IDEB", 1, ""),
        ("Língua Portuguesa", "Língua Portuguesa", 1, ""),
        ("Matemática", "Matemática", 1, ""),
        ("Aprovação", "Aprovação Geral", 1, "%"),
    ]
    st.markdown(f'<div style="font-size:16px;font-weight:850;margin:12px 0 8px;">{etapa}</div>',
                unsafe_allow_html=True)
    cols = st.columns(4)
    for col, (rotulo, campo, casas, suf) in zip(cols, indicadores):
        atual = pd.to_numeric(pd.Series([row_atual.get(campo)]), errors="coerce").iloc[0]
        ant = pd.to_numeric(pd.Series([row_anterior.get(campo)]), errors="coerce").iloc[0]
        if pd.notna(atual) and pd.notna(ant):
            delta = float(atual) - float(ant)
            seta = "↑" if delta > 0 else "↓" if delta < 0 else "→"
            cor = "#169B62" if delta > 0 else "#D92D20" if delta < 0 else "#6B7280"
            delta_txt = f"{seta} {abs(delta):.{casas}f}{suf}".replace(".", ",")
        else:
            cor, delta_txt = "#6B7280", "sem comparação"
        col.markdown(
            f'<div class="metric-card"><div class="metric-label">{rotulo}</div>'
            f'<div class="metric-value">{fmt(atual,casas,suf)}</div>'
            f'<div class="metric-foot" style="color:{cor};font-weight:750;">'
            f'{delta_txt} desde a aplicação anterior</div></div>',
            unsafe_allow_html=True
        )


def card_ranking_barueri_executivo(etapa, ano_ini, ano_fim):
    comp = comp_ranking_municipios_indicador(etapa, "IDEB", ano_ini, ano_fim, "Municipal")
    universo = ranking_base_municipios_indicador(
        etapa, ano_fim, "IDEB", "Municipal"
    )["Município"].nunique()
    r = comp[comp["Município"] == "Barueri"]
    if r.empty:
        st.info(f"Sem posição comparável para {etapa}.")
        return
    r = r.iloc[0]
    pa, pi, vp = r.get("Posição Atual"), r.get("Posição Inicial"), r.get("Variação de Posição")
    if pd.isna(vp):
        mov, cor = "sem comparação", "#6B7280"
    elif vp > 0:
        mov, cor = f"↑ {int(vp)} posições", "#169B62"
    elif vp < 0:
        mov, cor = f"↓ {abs(int(vp))} posições", "#D92D20"
    else:
        mov, cor = "→ mesma posição", "#6B7280"
    anterior = f" • posição anterior: {int(pi)}º" if pd.notna(pi) else ""
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">Ranking IDEB • {etapa}</div>'
        f'<div class="metric-value">{int(pa)}º</div>'
        f'<div class="metric-foot">entre {universo} municípios com resultado na Rede Municipal em {ano_fim}</div>'
        f'<div style="margin-top:7px;font-size:12px;font-weight:800;color:{cor};">'
        f'{mov} desde {ano_ini}{anterior}</div></div>',
        unsafe_allow_html=True
    )

def cards_rede(row):
    itens = [
        ("IDEB", row.get("IDEB"), 1, ""),
        ("Fluxo • Aprovação", row.get("Aprovação Geral"), 1, "%"),
        ("Língua Portuguesa • SAEB", row.get("Língua Portuguesa"), 2, ""),
        ("Matemática • SAEB", row.get("Matemática"), 2, ""),
    ]
    cols = st.columns(4)
    for c, (label, val, casas, suf) in zip(cols, itens):
        c.markdown(
            f"""<div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{fmt(val,casas,suf)}</div>
            <div class="metric-foot">valor disponível para o ano selecionado</div>
            </div>""",
            unsafe_allow_html=True
        )

st.markdown(
    '<div class="brand"><div class="brand-icon">SAEB</div><div><div class="brand-title">'
    'Painel Saeb • Rede Municipal de Ensino de Barueri</div><div class="brand-sub">'
    'Série histórica 2005–2025 • IDEB • Saeb • Fluxo escolar</div></div></div>',
    unsafe_allow_html=True
)

pagina = st.radio(
    "Navegação principal",
    ["Visão Geral","Municípios","Escolas","Metodologia e dados"],
    horizontal=True,
    label_visibility="collapsed",
    key="nav_principal"
)

sub_municipios = None
if pagina == "Municípios":
    sub_municipios = st.segmented_control(
        "Área municipal",
        ["Aprendizagem e rankings", "Comparar municípios"],
        default="Aprendizagem e rankings",
        selection_mode="single",
        key="subnav_municipios"
    ) or "Aprendizagem e rankings"

if pagina == "Visão Geral":
    st.markdown('<div class="eyebrow">Visão geral • Rede Municipal de Barueri</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">Panorama executivo — IDEB e Saeb</div>', unsafe_allow_html=True)
    st.caption(
        "Resumo da rede com resultados recentes, mudança entre as duas últimas aplicações, "
        "posição no ranking municipal, níveis Saeb e trajetória histórica."
    )

    fi = dados_municipio("Barueri", "Fundamental I", rede="Municipal").sort_values("Ano")
    fii = dados_municipio("Barueri", "Fundamental II", rede="Municipal").sort_values("Ano")
    anos_fi = sorted(int(a) for a in fi["Ano"].dropna().unique())
    anos_fii = sorted(int(a) for a in fii["Ano"].dropna().unique())

    if len(anos_fi) >= 2 and len(anos_fii) >= 2:
        atual_fi, anterior_fi = anos_fi[-1], anos_fi[-2]
        atual_fii, anterior_fii = anos_fii[-1], anos_fii[-2]
        rfi, rfi_ant = ultima_linha(fi, atual_fi), ultima_linha(fi, anterior_fi)
        rfii, rfii_ant = ultima_linha(fii, atual_fii), ultima_linha(fii, anterior_fii)

        st.markdown('<div class="section-title">Resultados mais recentes</div>', unsafe_allow_html=True)
        st.caption(f"Fundamental I: {anterior_fi} → {atual_fi} • Fundamental II: {anterior_fii} → {atual_fii}")
        card_executivo_variacao(rfi, rfi_ant, "Fundamental I")
        card_executivo_variacao(rfii, rfii_ant, "Fundamental II")

        st.markdown('<div class="section-title">Posição de Barueri no IDEB</div>', unsafe_allow_html=True)
        st.caption("Ranking entre municípios com resultado disponível na Rede Municipal.")
        c1, c2 = st.columns(2)
        with c1:
            card_ranking_barueri_executivo("Fundamental I", anterior_fi, atual_fi)
        with c2:
            card_ranking_barueri_executivo("Fundamental II", anterior_fii, atual_fii)

        # Um único seletor controla as duas seções abaixo.
        etapa_visao = st.segmented_control(
            "Etapa para os gráficos da visão geral",
            ["Fundamental I", "Fundamental II", "Ambos"],
            default="Ambos",
            selection_mode="single",
            key="visao_geral_etapa"
        ) or "Ambos"

        etapas_exibir = (
            ["Fundamental I", "Fundamental II"]
            if etapa_visao == "Ambos"
            else [etapa_visao]
        )

        st.markdown(
            '<div class="section-title">Onde Barueri está na escala Saeb</div>',
            unsafe_allow_html=True
        )
        st.caption(
            "A interpretação detalhada dos níveis fica em Metodologia e dados. "
            "A etapa selecionada acima vale também para a comparação com São Paulo e Brasil."
        )

        for etapa_atual in etapas_exibir:
            row_atual = rfi if etapa_atual == "Fundamental I" else rfii

            if etapa_visao == "Ambos":
                st.markdown(
                    f'<div style="font-size:15px;font-weight:800;margin:12px 0 6px;">'
                    f'{etapa_atual}</div>',
                    unsafe_allow_html=True
                )

            st.plotly_chart(
                fig_escala_saeb(row_atual, etapa_atual, "Língua Portuguesa"),
                use_container_width=True
            )
            st.plotly_chart(
                fig_escala_saeb(row_atual, etapa_atual, "Matemática"),
                use_container_width=True
            )

        st.markdown(
            '<div class="section-title">Barueri × Estado de São Paulo × Brasil</div>',
            unsafe_allow_html=True
        )
        st.caption(
            "Comparação histórica em linhas. Todos os valores numéricos ficam acima "
            "dos pontos, com deslocamentos diferentes para reduzir sobreposições."
        )

        for etapa_atual in etapas_exibir:
            base_barueri = fi if etapa_atual == "Fundamental I" else fii
            fig_ref, fonte_ref = grafico_comparativo_referencias_historico(
                base_barueri,
                etapa_atual
            )

            st.plotly_chart(
                fig_ref,
                use_container_width=True,
                config={
                    "displaylogo": False,
                    "toImageButtonOptions": {"format": "png", "scale": 2}
                }
            )
            st.caption(fonte_ref + " Este gráfico não participa dos rankings municipais.")

        st.markdown('<div class="section-title">Trajetória histórica</div>', unsafe_allow_html=True)
        st.plotly_chart(
            grafico_executivo_ideb_rede(fi, fii),
            use_container_width=True,
            config={"displaylogo": False, "toImageButtonOptions": {"format": "png", "scale": 2}}
        )
        st.caption("As metas aparecem somente nos anos em que estão disponíveis na base.")
    else:
        st.info("Não há duas aplicações disponíveis para montar a síntese executiva.")


elif pagina == "Escolas":
    st.markdown('<div class="eyebrow">Escolas</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">Painel de consulta às unidades escolares</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">A busca ignora acentos. Ex.: digitar "jose" encontra "José".</div>', unsafe_allow_html=True)

    @st.fragment
    def painel_escolas():
        etapa = st.selectbox(
            "Etapa",
            ["Fundamental I", "Fundamental II", "Ambos"],
            key="esc_etapa"
        )

        busca_escola = st.text_input(
            "Buscar escola",
            placeholder="Digite parte do nome — não é necessário usar acentos",
            key="busca_escola_principal"
        )

        if etapa == "Ambos":
            nomes_fi = set(
                escolas.loc[
                    escolas["Etapa"] == "Fundamental I",
                    "Escola"
                ].dropna().astype(str)
            )
            nomes_fii = set(
                escolas.loc[
                    escolas["Etapa"] == "Fundamental II",
                    "Escola"
                ].dropna().astype(str)
            )
            nomes_ambas = nomes_fi.intersection(nomes_fii)

            filtro_escola = escolas["Escola"].isin(nomes_ambas)
        else:
            filtro_escola = (escolas["Etapa"] == etapa)

        lista = filtrar_nomes_busca(
            escolas,
            "Escola",
            "Escola_Busca",
            busca_escola,
            filtro=filtro_escola
        )

        if not lista:
            st.info("Nenhuma escola encontrada para essa busca.")
            return

        with st.form("form_escola"):
            escola = st.selectbox(
                "Escola",
                lista,
                key="escola_principal"
            )
            st.form_submit_button("Aplicar filtros", type="primary")

        if etapa == "Ambos":
            fi_escola = dados_escola(escola, "Fundamental I")
            fii_escola = dados_escola(escola, "Fundamental II")

            anos_ambos = sorted(
                set(int(a) for a in fi_escola["Ano"].dropna().unique())
                | set(int(a) for a in fii_escola["Ano"].dropna().unique())
            )

            if not anos_ambos:
                st.info("Não há anos disponíveis para comparar as duas etapas.")
                return

            intervalo_ambos = st.select_slider(
                "Período",
                options=anos_ambos,
                value=(anos_ambos[0], anos_ambos[-1]),
                key="periodo_escola_ambos"
            )

            fi_escola = fi_escola[
                fi_escola["Ano"].between(intervalo_ambos[0], intervalo_ambos[1])
            ].copy()
            fii_escola = fii_escola[
                fii_escola["Ano"].between(intervalo_ambos[0], intervalo_ambos[1])
            ].copy()

            st.caption(
                "Modo Ambos: a mesma escola é acompanhada nas duas etapas. "
                "As proficiências Saeb preservam as escalas próprias de cada etapa."
            )

            abas_ambos = st.tabs([
                "IDEB",
                "Língua Portuguesa",
                "Matemática",
                "Notas padronizadas",
                "Aprovação",
                "N e P"
            ])

            with abas_ambos[0]:
                st.plotly_chart(
                    grafico_comparacao_etapas(
                        fi_escola, fii_escola, "IDEB",
                        f"IDEB — Fundamental I × Fundamental II — {escola}",
                        casas=1, metas=True
                    ),
                    use_container_width=True
                )

            with abas_ambos[1]:
                st.plotly_chart(
                    grafico_comparacao_etapas(
                        fi_escola, fii_escola, "Língua Portuguesa",
                        f"Língua Portuguesa — Fundamental I × Fundamental II — {escola}",
                        casas=1
                    ),
                    use_container_width=True
                )

            with abas_ambos[2]:
                st.plotly_chart(
                    grafico_comparacao_etapas(
                        fi_escola, fii_escola, "Matemática",
                        f"Matemática — Fundamental I × Fundamental II — {escola}",
                        casas=1
                    ),
                    use_container_width=True
                )

            with abas_ambos[3]:
                st.plotly_chart(
                    grafico_comparacao_etapas(
                        fi_escola, fii_escola, "Nota Padronizada LP",
                        f"LP padronizada (0–10) — FI × FII — {escola}",
                        casas=2
                    ),
                    use_container_width=True
                )
                st.plotly_chart(
                    grafico_comparacao_etapas(
                        fi_escola, fii_escola, "Nota Padronizada Matemática",
                        f"Matemática padronizada (0–10) — FI × FII — {escola}",
                        casas=2
                    ),
                    use_container_width=True
                )

            with abas_ambos[4]:
                st.plotly_chart(
                    grafico_comparacao_etapas(
                        fi_escola, fii_escola, "Aprovação Geral",
                        f"Aprovação — Fundamental I × Fundamental II — {escola}",
                        casas=1
                    ),
                    use_container_width=True
                )

            with abas_ambos[5]:
                st.plotly_chart(
                    grafico_comparacao_etapas(
                        fi_escola, fii_escola, "N",
                        f"Nota Média Padronizada (N) — FI × FII — {escola}",
                        casas=2
                    ),
                    use_container_width=True
                )
                st.plotly_chart(
                    grafico_comparacao_etapas(
                        fi_escola, fii_escola, "P",
                        f"Indicador de Rendimento (P) — FI × FII — {escola}",
                        casas=3
                    ),
                    use_container_width=True
                )

            return

        anos_e = sorted(
            int(a) for a in escolas.loc[
                (escolas["Etapa"] == etapa) &
                (escolas["Escola"] == escola),
                "Ano"
            ].dropna().unique()
        )

        if not anos_e:
            st.info("Não há anos disponíveis para a escola selecionada.")
            return

        intervalo = st.select_slider(
            "Período",
            options=anos_e,
            value=(anos_e[0], anos_e[-1]),
            key="periodo_escola"
        )

        d = dados_escola(escola, etapa, intervalo[0], intervalo[1])

        row = ultima_linha(d)
        if row is not None:
            cols = st.columns(4)
            for c, (rot, val, casas) in zip(cols, [
                ("IDEB",row.get("IDEB"),1),
                ("Língua Portuguesa",row.get("Língua Portuguesa"),2),
                ("Matemática",row.get("Matemática"),2),
                ("Aprovação",row.get("Aprovação Geral"),1)
            ]):
                c.markdown(
                    f'<div class="metric-card"><div class="metric-label">{rot}</div>'
                    f'<div class="metric-value">{fmt(val,casas,"%" if rot=="Aprovação" else "")}</div>'
                    f'<div class="metric-foot">último valor disponível</div></div>',
                    unsafe_allow_html=True
                )

        t1,t2,t3,t4 = st.tabs([
            "Desempenho",
            "Fluxo e rendimento",
            "Comparar escolas",
            "Ranking de escolas"
        ])

        with t1:
            st.plotly_chart(
                grafico_ideb_meta(d, f"IDEB e metas — {escola}"),
                use_container_width=True
            )
            st.plotly_chart(
                grafico_lp_mat(d, f"LP × Matemática — {escola}"),
                use_container_width=True
            )
            st.plotly_chart(
                grafico_lp_mat_n(d, f"LP + Matemática × N — {escola}"),
                use_container_width=True
            )

        with t2:
            st.plotly_chart(
                grafico_aprovacao_linhas(
                    d, etapa, f"Aprovação por série — {escola}"
                ),
                use_container_width=True
            )
            st.plotly_chart(
                grafico_aprovacao_series_p(
                    d, etapa, f"Aprovação por série × P — {escola}"
                ),
                use_container_width=True
            )

        with t3:
            c1, c2 = st.columns(2)

            with c1:
                busca_e1 = st.text_input(
                    "Buscar Escola 1",
                    placeholder="Ex.: jose",
                    key="busca_e1"
                )
                lista_e1 = filtrar_nomes_busca(
                    escolas,
                    "Escola",
                    "Escola_Busca",
                    busca_e1,
                    filtro=(escolas["Etapa"] == etapa)
                )

            with c2:
                busca_e2 = st.text_input(
                    "Buscar Escola 2",
                    placeholder="Ex.: conceicao",
                    key="busca_e2"
                )
                lista_e2 = filtrar_nomes_busca(
                    escolas,
                    "Escola",
                    "Escola_Busca",
                    busca_e2,
                    filtro=(escolas["Etapa"] == etapa)
                )

            if not lista_e1 or not lista_e2:
                st.info("Use as buscas acima até encontrar as duas escolas.")
            else:
                with st.form("form_comp_escolas"):
                    c1,c2,c3 = st.columns([2,2,1])
                    with c1:
                        e1 = st.selectbox(
                            "Escola 1",
                            lista_e1,
                            key="comp_escola_1"
                        )
                    with c2:
                        e2 = st.selectbox(
                            "Escola 2",
                            lista_e2,
                            key="comp_escola_2"
                        )
                    with c3:
                        indicador = st.selectbox(
                            "Indicador",
                            ["IDEB","Língua Portuguesa","Matemática","N","P","Aprovação Geral"],
                            key="comp_escola_indicador"
                        )
                    st.form_submit_button("Comparar", type="primary")

                d1 = dados_escola(
                    e1, etapa, intervalo[0], intervalo[1]
                )
                d2 = dados_escola(
                    e2, etapa, intervalo[0], intervalo[1]
                )

                if indicador in ["Língua Portuguesa","Matemática"]:
                    st.plotly_chart(
                        grafico_comparacao_lp_mat(
                            d1,e1,d2,e2,
                            f"LP e Matemática — {e1} × {e2}"
                        ),
                        use_container_width=True
                    )
                else:
                    st.plotly_chart(
                        grafico_comparacao_indicador(
                            [(e1,d1),(e2,d2)],
                            indicador,
                            f"{indicador} — comparação entre escolas"
                        ),
                        use_container_width=True
                    )

        with t4:
            st.caption(
                "O ranking pertence ao nível Escolas. "
                "A posição é recalculada de acordo com a etapa, período e universo selecionados."
            )

            c1, c2 = st.columns([1.5, 1])
            with c1:
                indicador_e = st.selectbox(
                    "Indicador do ranking",
                    list(RANKING_INDICADORES.keys()),
                    key="rank_e_indicador_escolas"
                )
            with c2:
                incluir_itbs_e = st.checkbox(
                    "Incluir ITBs",
                    value=True,
                    key="rank_e_incluir_itbs_escolas",
                    help="Os ITBs continuam na base e nas consultas individuais."
                )

            anos_rank_e = sorted(
                int(a)
                for a in escolas.loc[
                    escolas["Etapa"] == etapa,
                    "Ano"
                ].dropna().unique()
            )

            if len(anos_rank_e) < 2:
                st.info("Não há edições suficientes para calcular movimento no ranking.")
            else:
                c1, c2, c3 = st.columns([1,1,1])
                with c1:
                    ano_ini_e = st.selectbox(
                        "Ano inicial",
                        anos_rank_e[:-1],
                        index=max(0, len(anos_rank_e)-2),
                        key="rank_e_ini_escolas"
                    )

                finais_rank_e = [a for a in anos_rank_e if a > ano_ini_e]

                with c2:
                    ano_fim_e = st.selectbox(
                        "Ano final",
                        finais_rank_e,
                        index=len(finais_rank_e)-1,
                        key="rank_e_fim_escolas"
                    )

                with c3:
                    quantidade_e = st.selectbox(
                        "Quantidade exibida",
                        [10, 20, 30, 50, "Todos"],
                        index=1,
                        key="rank_e_qtd_escolas"
                    )

                comp_e = comp_ranking_escolas_indicador(
                    etapa,
                    indicador_e,
                    ano_ini_e,
                    ano_fim_e,
                    incluir_itbs=incluir_itbs_e
                )

                comparaveis_e = comp_e["Posição Inicial"].notna().sum() if not comp_e.empty else 0

                st.caption(
                    f"Universo atual: {comp_e['Escola'].nunique() if not comp_e.empty else 0} escolas • "
                    f"{comparaveis_e} com resultado comparável entre {ano_ini_e} e {ano_fim_e} • "
                    + ("ITBs incluídos." if incluir_itbs_e else "ITBs excluídos e posições recalculadas.")
                )

                if comp_e.empty:
                    st.info("Não há dados suficientes para montar o ranking selecionado.")
                else:
                    ranking_visual_com_barras(
                        comp_e,
                        "Escola",
                        indicador_e,
                        quantidade=quantidade_e,
                        destaque_barueri=False
                    )

                    botao_download_ranking(
                        comp_e, "Escola", indicador_e, quantidade_e,
                        f"escolas_{etapa}_{indicador_e}_{ano_fim_e}"
                    )

                    with st.expander("📊 Versão gráfica para visualizar ou baixar em PNG"):
                        st.plotly_chart(
                            grafico_ranking_download(
                                comp_e, "Escola", indicador_e, quantidade_e
                            ),
                            use_container_width=True,
                            config={
                                "displaylogo": False,
                                "toImageButtonOptions": {"format": "png", "scale": 2}
                            }
                        )

                    if indicador_e in ["Língua Portuguesa (SAEB)", "Matemática (SAEB)"]:
                        disciplina_e = (
                            "Língua Portuguesa"
                            if indicador_e.startswith("Língua")
                            else "Matemática"
                        )
                        st.markdown("#### Matriz nível × tendência")
                        fig_e, comp_matriz_e = fig_matriz_nivel_tendencia(
                            pd.DataFrame(),
                            "Escola",
                            disciplina_e,
                            ano_ini_e,
                            ano_fim_e,
                            etapa,
                            incluir_itbs=incluir_itbs_e
                        )
                        cards_movimento(comp_matriz_e)
                        st.plotly_chart(fig_e, use_container_width=True)

    painel_escolas()

elif pagina == "Municípios" and sub_municipios == "Aprendizagem e rankings":
    st.markdown('<div class="eyebrow">Municípios • Aprendizagem e rankings</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">Diagnóstico, níveis de proficiência e rankings</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">Acompanhe a aprendizagem em Língua Portuguesa e Matemática, '
        'a posição relativa dos municípios e a mudança de nível entre duas edições.</div>',
        unsafe_allow_html=True
    )

    @st.fragment
    def painel_aprendizagem():
        abas_apr = st.tabs([
            "Visão geral",
            "Organização dos níveis",
            "Ranking de municípios",
        ])

        # ====================================================
        # VISÃO GERAL
        # ====================================================
        with abas_apr[0]:
            modo = st.radio(
                "Visão",
                ["Por rede", "Por escola"],
                horizontal=True,
                label_visibility="collapsed",
                key="apr_visao"
            )

            etapa = st.segmented_control(
                "Etapa",
                ETAPAS,
                default="Fundamental I",
                selection_mode="single",
                key="apr_etapa_visao"
            ) or "Fundamental I"

            if modo == "Por rede":
                base = dados_municipio("Barueri", etapa, rede="Municipal")
                anos_disp = sorted(int(a) for a in base["Ano"].dropna().unique())
                ano = st.selectbox(
                    "Ano de referência",
                    anos_disp,
                    index=len(anos_disp)-1,
                    key="apr_ano_rede"
                )
                row = ultima_linha(base, ano)

                if row is not None:
                    cards_rede(row)

                    d_lp = resultado_evolucao_barueri(etapa, "Língua Portuguesa")
                    d_mat = resultado_evolucao_barueri(etapa, "Matemática")

                    c1,c2 = st.columns(2)
                    with c1:
                        st.plotly_chart(fig_escala_saeb(row, etapa, "Língua Portuguesa"), use_container_width=True)
                    with c2:
                        st.plotly_chart(fig_escala_saeb(row, etapa, "Matemática"), use_container_width=True)

                    st.markdown('<div class="section-title">O que este resultado indica?</div>', unsafe_allow_html=True)
                    c1,c2 = st.columns(2)

                    lp_padrao = localizar_padrao(row.get("Língua Portuguesa"), "Língua Portuguesa", etapa)
                    mat_padrao = localizar_padrao(row.get("Matemática"), "Matemática", etapa)

                    c1.markdown(
                        f'<div class="metric-card" style="min-height:230px;">'
                        f'<div class="metric-label">Língua Portuguesa</div>'
                        f'<div class="metric-value" style="font-size:20px;">'
                        f'{fmt(row.get("Língua Portuguesa"),2)} • '
                        f'Nível {fmt(row.get("Nível Língua Portuguesa"),0)} • {lp_padrao}</div>'
                        f'<div class="metric-foot" style="font-size:13px;line-height:1.45;">'
                        f'{EXPLICACAO_PADROES.get(lp_padrao,"")}</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                    c2.markdown(
                        f'<div class="metric-card" style="min-height:230px;">'
                        f'<div class="metric-label">Matemática</div>'
                        f'<div class="metric-value" style="font-size:20px;">'
                        f'{fmt(row.get("Matemática"),2)} • '
                        f'Nível {fmt(row.get("Nível Matemática"),0)} • {mat_padrao}</div>'
                        f'<div class="metric-foot" style="font-size:13px;line-height:1.45;">'
                        f'{EXPLICACAO_PADROES.get(mat_padrao,"")}</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                st.plotly_chart(
                    grafico_lp_mat(base, f"Série histórica das proficiências — {etapa}"),
                    use_container_width=True
                )

            else:
                busca_apr_escola = st.text_input(
                    "Buscar escola",
                    placeholder="Digite parte do nome — não é necessário usar acentos",
                    key="busca_aprendizagem_escola"
                )

                lista = filtrar_nomes_busca(
                    escolas,
                    "Escola",
                    "Escola_Busca",
                    busca_apr_escola,
                    filtro=(escolas["Etapa"] == etapa)
                )

                if not lista:
                    st.info("Nenhuma escola encontrada para essa busca.")
                else:
                    escola = st.selectbox(
                        "Selecione uma escola",
                        lista,
                        key="apr_escola"
                    )
                    d = dados_escola(escola, etapa)
                    st.plotly_chart(
                        grafico_lp_mat(d, f"Série histórica das proficiências — {escola}"),
                        use_container_width=True
                    )
        # ORGANIZAÇÃO DOS NÍVEIS
        # ====================================================
        with abas_apr[1]:
            st.markdown(
                '<div class="section-title">Escalas de proficiência do SAEB usadas no painel</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                '<div class="hero-sub">Língua Portuguesa e Matemática possuem faixas de '
                'interpretação diferentes. Por isso, a classificação deve ser lida sempre '
                'junto da disciplina e da etapa selecionadas.</div>',
                unsafe_allow_html=True
            )

            etapa_nivel = st.segmented_control(
                "Etapa",
                ETAPAS,
                default="Fundamental I",
                selection_mode="single",
                key="apr_etapa_niveis"
            ) or "Fundamental I"

            st.markdown(
                '<div class="info"><b>Como ler:</b> o <b>nível numérico</b> indica a faixa '
                'detalhada da escala de proficiência. O <b>padrão de desempenho</b> '
                '(Abaixo do básico, Básico, Adequado ou Avançado) resume a interpretação '
                'do resultado. As faixas do padrão são diferentes em Língua Portuguesa e '
                'Matemática.</div>',
                unsafe_allow_html=True
            )

            disc_tabs = st.tabs([
                "Língua Portuguesa",
                "Matemática",
                "Níveis numéricos"
            ])

            with disc_tabs[0]:
                painel_padrao_disciplina(
                    "Língua Portuguesa",
                    etapa_nivel
                )

                base_atual = dados_municipio(
                    "Barueri",
                    etapa_nivel
                )
                if not base_atual.empty:
                    ano_atual = int(base_atual["Ano"].max())
                    atual = ultima_linha(base_atual, ano_atual)

                    if atual is not None:
                        st.markdown(
                            '<div class="section-title">Onde Barueri se encontra</div>',
                            unsafe_allow_html=True
                        )
                        st.markdown(
                            card_situacao_atual(
                                "Língua Portuguesa",
                                atual.get("Língua Portuguesa"),
                                atual.get("Nível Língua Portuguesa"),
                                etapa_nivel,
                                atual.get("Padrão Língua Portuguesa")
                            ),
                            unsafe_allow_html=True
                        )
                        st.caption(
                            f"Último resultado disponível na base para {etapa_nivel}: {ano_atual}."
                        )

            with disc_tabs[1]:
                painel_padrao_disciplina(
                    "Matemática",
                    etapa_nivel
                )

                base_atual = dados_municipio(
                    "Barueri",
                    etapa_nivel
                )
                if not base_atual.empty:
                    ano_atual = int(base_atual["Ano"].max())
                    atual = ultima_linha(base_atual, ano_atual)

                    if atual is not None:
                        st.markdown(
                            '<div class="section-title">Onde Barueri se encontra</div>',
                            unsafe_allow_html=True
                        )
                        st.markdown(
                            card_situacao_atual(
                                "Matemática",
                                atual.get("Matemática"),
                                atual.get("Nível Matemática"),
                                etapa_nivel,
                                atual.get("Padrão Matemática")
                            ),
                            unsafe_allow_html=True
                        )
                        st.caption(
                            f"Último resultado disponível na base para {etapa_nivel}: {ano_atual}."
                        )

            with disc_tabs[2]:
                st.markdown(
                    '<div class="section-title">Níveis numéricos da escala</div>',
                    unsafe_allow_html=True
                )
                st.markdown(
                    '<div class="hero-sub">Esta escala detalhada é utilizada nas cores dos '
                    'rankings. A cor identifica o nível; o código da cor não é exibido ao '
                    'usuário.</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    legenda_niveis_html(etapa_nivel),
                    unsafe_allow_html=True
                )

                linhas = []
                for nivel, faixa in FAIXAS_NIVEIS[etapa_nivel]:
                    linhas.append({
                        "Nível": f"Nível {nivel}",
                        "Faixa de proficiência": faixa,
                    })

                st.dataframe(
                    pd.DataFrame(linhas),
                    hide_index=True,
                    use_container_width=True
                )

            st.markdown(
                '<div class="info"><b>Ranking × nível:</b> posição no ranking e nível de '
                'proficiência são informações diferentes. Uma rede ou escola pode melhorar '
                'a pontuação e o nível e, ainda assim, perder posições se outras avançarem '
                'mais no mesmo período.</div>',
                unsafe_allow_html=True
            )

        # RANKING DE MUNICÍPIOS
        # ====================================================
        with abas_apr[2]:
            st.markdown(
                '<div class="section-title">Ranking entre municípios</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                '<div class="hero-sub">Barueri permanece sempre como Rede Municipal. '
                'Você pode analisar o ranking geral da rede escolhida ou recalcular o ranking '
                'somente entre municípios selecionados.</div>',
                unsafe_allow_html=True
            )

            modo_rank_m = st.radio(
                "Tipo de ranking",
                ["Ranking geral", "Municípios selecionados"],
                horizontal=True,
                key="rank_m_modo"
            )

            # ------------------------------------------------
            # 1º passo: definir o universo da comparação.
            # Etapa, indicador e rede ficam fora do form para que
            # a troca de rede atualize imediatamente o universo.
            # ------------------------------------------------
            c1,c2,c3 = st.columns([1.2,1.7,1.3])

            with c1:
                etapa_m = st.selectbox(
                    "Etapa",
                    ETAPAS,
                    key="rank_m_etapa"
                )

            with c2:
                indicador_m = st.selectbox(
                    "Indicador do ranking",
                    list(RANKING_INDICADORES.keys()),
                    key="rank_m_indicador"
                )

            redes_disponiveis = [
                r for r in ["Municipal", "Pública", "Estadual", "Federal"]
                if (
                    (municipios["Etapa"] == etapa_m) &
                    (municipios["Rede"] == r)
                ).any()
            ]

            with c3:
                rede_m = st.selectbox(
                    "Rede de comparação",
                    redes_disponiveis,
                    key="rank_m_rede"
                )

            anos_m = sorted(
                int(a)
                for a in municipios.loc[
                    (municipios["Etapa"] == etapa_m) &
                    (municipios["Rede"] == rede_m),
                    "Ano"
                ].dropna().unique()
            )

            if len(anos_m) < 2:
                st.info("Não há duas edições disponíveis para este recorte.")
                return

            # ------------------------------------------------
            # 2º passo: período e quantidade.
            # ------------------------------------------------
            with st.form("form_ranking_municipios_v299"):
                c4,c5,c6 = st.columns([1,1,1.2])

                with c4:
                    ano_ini_m = st.selectbox(
                        "Ano inicial",
                        anos_m[:-1],
                        index=0,
                        key="rank_m_ini"
                    )

                finais_m = [a for a in anos_m if a > ano_ini_m]

                with c5:
                    ano_fim_m = st.selectbox(
                        "Ano final",
                        finais_m,
                        index=len(finais_m)-1,
                        key="rank_m_fim"
                    )

                with c6:
                    quantidade_m = st.selectbox(
                        "Quantidade exibida",
                        [10, 20, 30, 50, "Todos"],
                        index=1,
                        key="rank_m_qtd"
                    )

                st.form_submit_button("Aplicar ranking", type="primary")

            coluna_rank_m = RANKING_INDICADORES[indicador_m]["coluna"]

            base_rede_ano = municipios.loc[
                (municipios["Etapa"] == etapa_m) &
                (municipios["Rede"] == rede_m) &
                (municipios["Ano"] == ano_fim_m)
            ].copy()

            municipios_na_rede = base_rede_ano["Município"].dropna().nunique()

            valores_validos = pd.to_numeric(
                base_rede_ano[coluna_rank_m],
                errors="coerce"
            )
            municipios_com_resultado = (
                base_rede_ano.loc[
                    valores_validos.notna(),
                    "Município"
                ]
                .dropna()
                .nunique()
            )

            st.caption(
                f"Rede {rede_m} • {etapa_m} • {ano_fim_m}: "
                f"{municipios_na_rede} municípios possuem registro nessa rede; "
                f"{municipios_com_resultado} têm resultado válido para {indicador_m}. "
                "O ranking é calculado somente depois desse filtro. "
                "Barueri permanece como referência da Rede Municipal."
            )

            if modo_rank_m == "Ranking geral":
                comp_m = comp_ranking_municipios_indicador(
                    etapa_m,
                    indicador_m,
                    ano_ini_m,
                    ano_fim_m,
                    rede_m
                )
                comp_m = adicionar_barueri_indicador(
                    comp_m,
                    etapa_m,
                    indicador_m,
                    ano_ini_m,
                    ano_fim_m,
                    rede_m
                )

                universo = comp_m[
                    comp_m.get("_Referencia", False) != True
                ] if "_Referencia" in comp_m.columns else comp_m

                st.markdown(
                    f'<div class="info"><b>Rede selecionada:</b> {rede_m}. '
                    f'<b>Universo efetivo do ranking:</b> '
                    f'{universo["Município"].nunique()} municípios com resultado válido '
                    f'para {indicador_m} em {ano_fim_m} — {etapa_m}. '
                    f'<br><b>Barueri:</b> permanece como referência da Rede Municipal, '
                    f'mesmo quando a comparação escolhida é Pública, Estadual ou Federal. '
                    f'Barueri não é contabilizada como integrante de uma rede que não possui.</div>',
                    unsafe_allow_html=True
                )

                ranking_visual_com_barras(
                    comp_m,
                    "Município",
                    indicador_m,
                    quantidade=quantidade_m,
                    destaque_barueri=True
                )

                botao_download_ranking(
                    comp_m, "Município", indicador_m, quantidade_m,
                    f"municipios_{etapa_m}_{rede_m}_{indicador_m}_{ano_fim_m}"
                )
                with st.expander("📊 Versão gráfica para visualizar ou baixar em PNG"):
                    st.plotly_chart(
                        grafico_ranking_download(
                            comp_m, "Município", indicador_m, quantidade_m
                        ),
                        use_container_width=True,
                        config={"displaylogo": False, "toImageButtonOptions": {"format": "png", "scale": 2}}
                    )

            else:
                busca_rank_m = st.text_input(
                    "Buscar município",
                    placeholder='Digite sem acento, por exemplo: "sao"',
                    key="rank_m_busca_texto"
                )

                selecionados_atuais = st.session_state.get(
                    "rank_m_selecionados",
                    []
                )

                opcoes_m = opcoes_municipios_busca_ranking(
                    etapa_m,
                    rede_m,
                    busca_rank_m,
                    selecionados_atuais
                )

                selecionados_m = st.multiselect(
                    "Selecione os municípios",
                    opcoes_m,
                    max_selections=10,
                    key="rank_m_selecionados",
                    placeholder="Selecione até 10 municípios"
                )

                if not selecionados_m:
                    st.info(
                        "Selecione pelo menos um município. "
                        "Barueri será incluída automaticamente como Rede Municipal."
                    )
                else:
                    comp_sel = ranking_selecionados_municipios(
                        selecionados_m,
                        etapa_m,
                        indicador_m,
                        ano_ini_m,
                        ano_fim_m,
                        rede_m
                    )

                    st.markdown(
                        f'<div class="info"><b>Ranking restrito ao grupo selecionado.</b> '
                        f'Primeiro é aplicado o filtro da Rede {rede_m}; depois as posições '
                        f'são recalculadas apenas entre os municípios escolhidos. '
                        f'Barueri permanece como referência da Rede Municipal.</div>',
                        unsafe_allow_html=True
                    )

                    ranking_visual_com_barras(
                        comp_sel,
                        "Município",
                        indicador_m,
                        quantidade="Todos",
                        destaque_barueri=True
                    )

                    botao_download_ranking(
                        comp_sel, "Município", indicador_m, "Todos",
                        f"municipios_selecionados_{etapa_m}_{rede_m}_{indicador_m}_{ano_fim_m}"
                    )
                    with st.expander("📊 Versão gráfica para visualizar ou baixar em PNG"):
                        st.plotly_chart(
                            grafico_ranking_download(
                                comp_sel, "Município", indicador_m, "Todos"
                            ),
                            use_container_width=True,
                            config={"displaylogo": False, "toImageButtonOptions": {"format": "png", "scale": 2}}
                        )

            # A matriz nível × tendência só faz sentido para proficiências SAEB.
            if indicador_m in ["Língua Portuguesa (SAEB)", "Matemática (SAEB)"]:
                disciplina_m = (
                    "Língua Portuguesa"
                    if indicador_m.startswith("Língua")
                    else "Matemática"
                )
                st.markdown(
                    '<div class="section-title">Matriz nível × tendência</div>',
                    unsafe_allow_html=True
                )
                fig_m, comp_matriz = fig_matriz_nivel_tendencia(
                    pd.DataFrame(),
                    "Município",
                    disciplina_m,
                    ano_ini_m,
                    ano_fim_m,
                    etapa_m
                )
                cards_movimento(comp_matriz)
                st.plotly_chart(fig_m, use_container_width=True)


    painel_aprendizagem()

elif pagina == "Municípios" and sub_municipios == "Comparar municípios":
    st.markdown('<div class="eyebrow">Municípios • Comparações</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">Barueri no contexto regional e estadual</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">A busca por município ignora acentos. Barueri permanece como referência.</div>', unsafe_allow_html=True)

    @st.fragment
    def painel_territorio():
        etapa = st.segmented_control(
            "Etapa",
            ETAPAS,
            default="Fundamental I",
            selection_mode="single",
            key="ter_etapa"
        ) or "Fundamental I"

        base_bar = dados_municipio("Barueri", etapa)
        anos_disp = sorted(
            int(a) for a in base_bar["Ano"].dropna().unique()
        )

        rede_territorio = st.selectbox(
            "Rede de comparação",
            [
                r for r in ["Municipal", "Pública", "Estadual", "Federal"]
                if (
                    (municipios["Etapa"] == etapa) &
                    (municipios["Rede"] == r)
                ).any()
            ],
            key="ter_rede_comparacao"
        )

        busca_municipio = st.text_input(
            "Buscar município",
            placeholder='Ex.: "sao" encontra "São Paulo"',
            key="busca_municipio_territorio"
        )

        lista_filtrada = filtrar_nomes_busca(
            municipios,
            "Município",
            "Município_Busca",
            busca_municipio,
            filtro=(
                (municipios["Etapa"] == etapa) &
                (municipios["Rede"] == rede_territorio) &
                (municipios["Município"] != "Barueri")
            )
        )

        # Mantém no multiselect municípios já escolhidos, mesmo quando
        # o usuário muda o texto da busca para localizar outro.
        ja_selecionados = st.session_state.get(
            "ter_municipios_selecionados", []
        )
        opcoes_municipios = sorted(
            set(lista_filtrada).union(ja_selecionados)
        )

        with st.form("form_comparacao_municipio"):
            c1,c2 = st.columns(2)
            with c1:
                ano_ini = st.selectbox(
                    "Ano inicial",
                    anos_disp,
                    index=0,
                    key="ter_ano_ini"
                )
            finais = [a for a in anos_disp if a >= ano_ini]
            with c2:
                ano_fim = st.selectbox(
                    "Ano final",
                    finais,
                    index=len(finais)-1,
                    key="ter_ano_fim"
                )

            outros = st.multiselect(
                "Comparar Barueri com",
                opcoes_municipios,
                max_selections=5,
                placeholder="Selecione até 5 municípios",
                key="ter_municipios_selecionados"
            )

            st.form_submit_button(
                "Aplicar comparação",
                type="primary"
            )

        bar = dados_municipio(
            "Barueri",
            etapa,
            ano_ini,
            ano_fim,
            rede="Municipal"
        )

        def dados_municipio_rede_selecionada(nome):
            x = municipios[
                (municipios["Município"] == nome) &
                (municipios["Etapa"] == etapa) &
                (municipios["Rede"] == rede_territorio) &
                (municipios["Ano"].between(ano_ini, ano_fim))
            ].copy()
            return x.sort_values("Ano")

        total_rede_territorio = municipios.loc[
            (municipios["Etapa"] == etapa) &
            (municipios["Rede"] == rede_territorio) &
            (municipios["Ano"] == ano_fim),
            "Município"
        ].nunique()

        st.markdown(
            f'<div class="info"><b>Referência:</b> Barueri — Rede Municipal. '
            f'<b>Comparação:</b> Rede {rede_territorio} — {etapa}. '
            f'<b>Municípios com registro em {ano_fim}:</b> {total_rede_territorio}.</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-title">Modelo comparativo do Colab — LP e Matemática</div>',
            unsafe_allow_html=True
        )

        if outros:
            comp = outros[0]
            dc = dados_municipio_rede_selecionada(comp)
            st.plotly_chart(
                grafico_comparacao_lp_mat(
                    bar,
                    "Barueri",
                    dc,
                    comp,
                    f"LP e Matemática — Barueri × {comp} — {etapa}"
                ),
                use_container_width=True
            )
        else:
            st.markdown(
                '<div class="info">Barueri permanece visível mesmo sem outro município selecionado. '
                'Use a busca acima para localizar um município, mesmo digitando sem acentos.</div>',
                unsafe_allow_html=True
            )
            st.plotly_chart(
                grafico_lp_mat(
                    bar,
                    f"LP e Matemática — Barueri — {etapa}"
                ),
                use_container_width=True
            )

        st.markdown(
            '<div class="section-title">Comparações por indicador</div>',
            unsafe_allow_html=True
        )

        indicador = st.selectbox(
            "Indicador",
            [
                "IDEB",
                "N",
                "P",
                "Aprovação Geral",
                "Língua Portuguesa",
                "Matemática"
            ],
            key="ter_ind"
        )

        datasets = [("Barueri",bar)]
        for nome in outros:
            datasets.append(
                (
                    f"{nome} — {rede_territorio}",
                    dados_municipio_rede_selecionada(nome)
                )
            )

        st.plotly_chart(
            grafico_comparacao_indicador(
                datasets,
                indicador,
                f"{indicador} — Barueri e municípios selecionados"
            ),
            use_container_width=True
        )

        st.markdown(
            '<div class="section-title">Composição do IDEB por município</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="hero-sub">Os municípios são observações independentes. '
            'Por isso, não há linhas ligando uma cidade à outra. '
            'A leitura é feita individualmente como N × P → IDEB.</div>',
            unsafe_allow_html=True
        )

        ano_ref = st.selectbox(
            "Ano da composição",
            finais,
            index=len(finais)-1,
            key="np_ano"
        )

        nomes = ["Barueri"] + outros
        rows = []

        for nome in nomes:
            if nome == "Barueri":
                d_np = dados_municipio(
                    "Barueri",
                    etapa,
                    ano_ref,
                    ano_ref,
                    rede="Municipal"
                )
                nome_exibido = "Barueri — Municipal"
            else:
                d_np = municipios[
                    (municipios["Município"] == nome) &
                    (municipios["Etapa"] == etapa) &
                    (municipios["Rede"] == rede_territorio) &
                    (municipios["Ano"] == ano_ref)
                ].copy()
                nome_exibido = f"{nome} — {rede_territorio}"

            if not d_np.empty:
                r_np = d_np.iloc[-1]
                rows.append({
                    "Município": nome_exibido,
                    "Nota Padronizada LP": r_np.get("Nota Padronizada LP"),
                    "Nota Padronizada Matemática": r_np.get("Nota Padronizada Matemática"),
                    "N": r_np.get("N"),
                    "P": r_np.get("P"),
                    "IDEB": r_np.get("IDEB")
                })

        trans = tabela_composicao_unidades(rows)

        if trans.empty:
            st.info("Não há dados suficientes para a composição do IDEB no ano selecionado.")
        else:
            cards_composicao_ideb(
                trans,
                "Município"
            )

            tabela_np = trans.copy()
            tabela_np["Nota Padronizada LP"] = tabela_np["Nota Padronizada LP"].map(
                lambda v: fmt(v,2) if pd.notna(v) else "—"
            )
            tabela_np["Nota Padronizada Matemática"] = (
                tabela_np["Nota Padronizada Matemática"].map(
                    lambda v: fmt(v,2) if pd.notna(v) else "—"
                )
            )
            tabela_np["N"] = tabela_np["N"].map(
                lambda v: fmt(v,2) if pd.notna(v) else "—"
            )
            tabela_np["P"] = tabela_np["P"].map(
                lambda v: fmt(v,3) if pd.notna(v) else "—"
            )
            tabela_np["IDEB"] = tabela_np["IDEB"].map(
                lambda v: fmt(v,1) if pd.notna(v) else "—"
            )

            st.dataframe(
                tabela_np,
                hide_index=True,
                use_container_width=True
            )


    painel_territorio()


elif pagina == "Metodologia e dados":
    st.markdown('<div class="eyebrow">Metodologia e dados</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">Como interpretar os indicadores do painel</div>', unsafe_allow_html=True)

    tabs_met = st.tabs(["Escalas Saeb", "IDEB e notas padronizadas", "Critérios de comparação", "Fontes e dados"])

    with tabs_met[0]:
        st.markdown("#### Escalas de proficiência e padrões de desempenho")
        st.write(
            "A nota Saeb aparece em pontos. O nível numérico localiza o resultado na escala; "
            "o padrão de desempenho traduz pedagogicamente essa posição em "
            "Abaixo do básico, Básico, Adequado ou Avançado."
        )

        etapa_met = st.segmented_control(
            "Etapa", ETAPAS, default="Fundamental I",
            selection_mode="single", key="met_etapa"
        ) or "Fundamental I"

        st.markdown("##### Língua Portuguesa")
        painel_padrao_disciplina("Língua Portuguesa", etapa_met)

        st.markdown("##### Matemática")
        painel_padrao_disciplina("Matemática", etapa_met)

        # Relação entre nível numérico e padrão. Algumas faixas de nível atravessam
        # o ponto de corte entre dois padrões; isso é indicado explicitamente.
        MAPA_PADRAO_NIVEL = {
            "Fundamental I": {
                "Língua Portuguesa": {
                    0:"Abaixo do básico", 1:"Abaixo do básico", 2:"Abaixo do básico",
                    3:"Abaixo do básico → Básico", 4:"Básico → Adequado",
                    5:"Adequado", 6:"Adequado → Avançado",
                    7:"Avançado", 8:"Avançado", 9:"Avançado", 10:"Avançado"
                },
                "Matemática": {
                    0:"Abaixo do básico", 1:"Abaixo do básico", 2:"Abaixo do básico",
                    3:"Abaixo do básico", 4:"Abaixo do básico → Básico",
                    5:"Básico → Adequado", 6:"Adequado",
                    7:"Adequado → Avançado", 8:"Avançado", 9:"Avançado", 10:"Avançado"
                }
            },
            "Fundamental II": {
                "Língua Portuguesa": {
                    0:"Abaixo do básico", 1:"Abaixo do básico",
                    2:"Básico", 3:"Básico", 4:"Adequado", 5:"Adequado",
                    6:"Avançado", 7:"Avançado", 8:"Avançado", 9:"Avançado"
                },
                "Matemática": {
                    0:"Abaixo do básico", 1:"Abaixo do básico",
                    2:"Básico", 3:"Básico", 4:"Básico", 5:"Adequado",
                    6:"Adequado", 7:"Avançado", 8:"Avançado", 9:"Avançado"
                }
            }
        }

        linhas_niveis = []
        for nivel, faixa in FAIXAS_NIVEIS[etapa_met]:
            linhas_niveis.append({
                "Nível": f"Nível {nivel}",
                "Faixa de proficiência": faixa,
                "Categoria — Língua Portuguesa": MAPA_PADRAO_NIVEL[etapa_met]["Língua Portuguesa"][nivel],
                "Categoria — Matemática": MAPA_PADRAO_NIVEL[etapa_met]["Matemática"][nivel],
            })

        st.markdown("##### Relação entre nível e categoria")
        st.dataframe(
            pd.DataFrame(linhas_niveis),
            hide_index=True,
            use_container_width=True
        )

        st.info(
            "Nível numérico e padrão de desempenho não têm sempre os mesmos pontos de corte. "
            "Quando uma faixa atravessa a fronteira entre duas categorias, o painel mostra a transição "
            "(por exemplo, Básico → Adequado), em vez de atribuir uma categoria incorreta ao nível inteiro."
        )

    with tabs_met[1]:
        st.markdown("#### Da proficiência ao IDEB")
        st.write(
            "O painel mantém as proficiências originais e acrescenta, apenas para visualização, "
            "as notas padronizadas separadas de Língua Portuguesa e Matemática na escala 0–10."
        )
        st.markdown("**N e P são lidos diretamente da base oficial; N × P compõe o IDEB.**")
        st.info(
            "O dashboard não recalcula N. Quando N não estiver publicado na linha correspondente, "
            "o painel mantém a ausência do dado. As notas padronizadas separadas de LP e Matemática "
            "não substituem o N oficial."
        )

    with tabs_met[2]:
        st.markdown("#### Critérios de comparação")
        st.write("Municípios são comparados com municípios e escolas com escolas. As redes permanecem separadas quando necessário.")
        st.write("Nos rankings, posição inicial e final usam o mesmo universo. Ao excluir ITBs, as posições das escolas são recalculadas.")
        st.write("Ausências são mantidas como ausência; o painel não estima resultados.")

    with tabs_met[3]:
        st.markdown("#### Bases do projeto")
        st.write("O painel utiliza as bases municipais e escolares tratadas previamente no Google Colab.")
        st.write("Estado de São Paulo e Brasil serão incluídos na Visão Geral apenas com agregados oficiais de 2025.")
        st.caption("Os downloads respeitam os filtros aplicados.")

st.markdown(
    '<div class="footer">Painel educacional • dados organizados a partir das bases SAEB/IDEB fornecidas para o projeto.</div>',
    unsafe_allow_html=True
)
