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
            x=df["Ano"], y=df["Língua Portuguesa"], mode="lines+markers+text",
            text=[fmt(v,1) if pd.notna(v) else "" for v in df["Língua Portuguesa"]],
            textposition="top center", name="Língua Portuguesa",
            line=dict(color=ROSA, width=3), marker=dict(size=8)
        ))
        fig.add_trace(go.Scatter(
            x=df["Ano"], y=df["Matemática"], mode="lines+markers+text",
            text=[fmt(v,1) if pd.notna(v) else "" for v in df["Matemática"]],
            textposition="bottom center", name="Matemática",
            line=dict(color=LILAS, width=3), marker=dict(size=8)
        ))
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
        textposition="bottom center", line=dict(color=LARANJA, width=3), marker=dict(size=8)
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
        (df1, "Língua Portuguesa", f"{nome1} — LP", ROSA, "top center"),
        (df1, "Matemática", f"{nome1} — Matemática", LILAS, "bottom center"),
        (df2, "Língua Portuguesa", f"{nome2} — LP", AZUL_COMP, "top center"),
        (df2, "Matemática", f"{nome2} — Matemática", VERDE_COMP, "bottom center"),
    ]
    for d, col, nome, cor, pos in traces:
        fig.add_trace(go.Scatter(
            x=d["Ano"], y=d[col], mode="lines+markers+text", name=nome,
            text=[fmt(v,1) if pd.notna(v) else "" for v in d[col]],
            textposition=pos, line=dict(color=cor, width=3), marker=dict(size=8)
        ))
    return estilo_fig(fig, titulo, "Proficiência SAEB", 540)

def grafico_comparacao_indicador(datasets, indicador, titulo):
    fig = go.Figure()
    cores = [AZUL, AZUL_COMP, VERDE_COMP, ROSA, LILAS, LARANJA]
    for i, (nome, df) in enumerate(datasets):
        fig.add_trace(go.Scatter(
            x=df["Ano"], y=df[indicador], mode="lines+markers+text", name=nome,
            text=[fmt(v,2 if indicador not in ["IDEB","Aprovação Geral"] else 1) if pd.notna(v) else "" for v in df[indicador]],
            textposition="top center", line=dict(color=cores[i % len(cores)], width=3), marker=dict(size=8)
        ))
    return estilo_fig(fig, titulo, indicador, 515)


def grafico_comparacao_etapas(df_fi, df_fii, indicador, titulo, casas=2, metas=False):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_fi["Ano"], y=df_fi[indicador],
        mode="lines+markers+text",
        name=f"Fundamental I — {indicador}",
        text=[fmt(v, casas) if pd.notna(v) else "" for v in df_fi[indicador]],
        textposition="top center",
        line=dict(color=AZUL, width=3),
        marker=dict(size=8)
    ))

    fig.add_trace(go.Scatter(
        x=df_fii["Ano"], y=df_fii[indicador],
        mode="lines+markers+text",
        name=f"Fundamental II — {indicador}",
        text=[fmt(v, casas) if pd.notna(v) else "" for v in df_fii[indicador]],
        textposition="bottom center",
        line=dict(color=LILAS, width=3),
        marker=dict(size=8)
    ))

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
    fig = estilo_fig(fig, titulo, ytitle, 500)

    if indicador == "Aprovação Geral":
        fig.update_yaxes(range=[0, 105])

    return fig


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
    ["Visão da rede","Escolas","Aprendizagem","Território"],
    horizontal=True,
    label_visibility="collapsed",
    key="nav_principal"
)

if pagina == "Visão da rede":
    st.markdown('<div class="eyebrow">Visão da rede • Barueri municipal</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">Rede Municipal de Ensino de Barueri — IDEB e Saeb, 2005–2025</div>', unsafe_allow_html=True)

    etapa = st.segmented_control(
        "Etapa", ETAPAS, default="Fundamental I",
        selection_mode="single", label_visibility="collapsed"
    ) or "Fundamental I"

    sub = st.tabs([
        "Panorama",
        "Trajetória e metas",
        "Fundamental I × Fundamental II",
        "IDEB, LP e Matemática",
        "Fluxo e permanência",
        "Movimento da rede",
        "Transparência"
    ])

    base = dados_municipio("Barueri", etapa)
    anos_disp = sorted(int(a) for a in base["Ano"].dropna().unique())

    with sub[0]:
        ano = st.selectbox("Ano de referência", anos_disp, index=len(anos_disp)-1, key="pan_ano")
        row = ultima_linha(base, ano)
        if row is not None:
            st.markdown('<div class="section-title">Síntese da rede</div>', unsafe_allow_html=True)
            cards_rede(row)
            if pd.notna(row.get("Meta IDEB")):
                st.markdown(
                    f'<div class="info"><b>Meta IDEB:</b> {fmt(row.get("Meta IDEB"),1)} &nbsp; • &nbsp; '
                    f'<b>Situação:</b> {row.get("Situação Meta","—")} &nbsp; • &nbsp; '
                    f'<b>Diferença:</b> {fmt(row.get("Diferença para Meta"),1)}</div>',
                    unsafe_allow_html=True
                )
            st.plotly_chart(grafico_ideb_meta(base, f"Série histórica do IDEB — {etapa}"), use_container_width=True)

    with sub[1]:
        c1, c2 = st.columns(2)
        with c1:
            ano_ini = st.selectbox("Ano inicial", anos_disp, index=0, key="traj_ini")
        finais = [a for a in anos_disp if a >= ano_ini]
        with c2:
            ano_fim = st.selectbox("Ano final", finais, index=len(finais)-1, key="traj_fim")
        d = base[base["Ano"].between(ano_ini, ano_fim)]
        st.plotly_chart(grafico_ideb_meta(d, f"IDEB e metas — {ano_ini}–{ano_fim}"), use_container_width=True)

    with sub[2]:
        st.markdown('<div class="section-title">Fundamental I × Fundamental II</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="hero-sub">Comparação das duas etapas da Rede Municipal de Barueri no mesmo gráfico, '
            'com seleção livre do período entre 2005 e 2025.</div>',
            unsafe_allow_html=True
        )

        base_fi = dados_municipio("Barueri", "Fundamental I")
        base_fii = dados_municipio("Barueri", "Fundamental II")

        anos_comp = sorted(
            set(int(a) for a in base_fi["Ano"].dropna().unique())
            | set(int(a) for a in base_fii["Ano"].dropna().unique())
        )

        c1, c2 = st.columns(2)
        with c1:
            comp_ini = st.selectbox(
                "Ano inicial",
                anos_comp,
                index=0,
                key="fi_fii_ini"
            )

        finais_comp = [a for a in anos_comp if a >= comp_ini]

        with c2:
            comp_fim = st.selectbox(
                "Ano final",
                finais_comp,
                index=len(finais_comp)-1,
                key="fi_fii_fim"
            )

        fi = base_fi[base_fi["Ano"].between(comp_ini, comp_fim)].copy()
        fii = base_fii[base_fii["Ano"].between(comp_ini, comp_fim)].copy()

        tabs_comp = st.tabs([
            "IDEB",
            "Língua Portuguesa",
            "Matemática",
            "Aprovação",
            "N",
            "P"
        ])

        with tabs_comp[0]:
            st.plotly_chart(
                grafico_comparacao_etapas(
                    fi, fii, "IDEB",
                    f"IDEB — Fundamental I × Fundamental II — {comp_ini}–{comp_fim}",
                    casas=1,
                    metas=True
                ),
                use_container_width=True
            )

        with tabs_comp[1]:
            st.plotly_chart(
                grafico_comparacao_etapas(
                    fi, fii, "Língua Portuguesa",
                    f"Língua Portuguesa — Fundamental I × Fundamental II — {comp_ini}–{comp_fim}",
                    casas=1
                ),
                use_container_width=True
            )

        with tabs_comp[2]:
            st.plotly_chart(
                grafico_comparacao_etapas(
                    fi, fii, "Matemática",
                    f"Matemática — Fundamental I × Fundamental II — {comp_ini}–{comp_fim}",
                    casas=1
                ),
                use_container_width=True
            )

        with tabs_comp[3]:
            st.plotly_chart(
                grafico_comparacao_etapas(
                    fi, fii, "Aprovação Geral",
                    f"Aprovação Geral — Fundamental I × Fundamental II — {comp_ini}–{comp_fim}",
                    casas=1
                ),
                use_container_width=True
            )

        with tabs_comp[4]:
            st.plotly_chart(
                grafico_comparacao_etapas(
                    fi, fii, "N",
                    f"Nota Média Padronizada (N) — Fundamental I × Fundamental II — {comp_ini}–{comp_fim}",
                    casas=2
                ),
                use_container_width=True
            )

        with tabs_comp[5]:
            st.plotly_chart(
                grafico_comparacao_etapas(
                    fi, fii, "P",
                    f"Indicador de Rendimento (P) — Fundamental I × Fundamental II — {comp_ini}–{comp_fim}",
                    casas=3
                ),
                use_container_width=True
            )

    with sub[3]:
        c1, c2 = st.columns(2)
        with c1:
            ano_ini = st.selectbox("Ano inicial", anos_disp, index=0, key="lp_ini")
        finais = [a for a in anos_disp if a >= ano_ini]
        with c2:
            ano_fim = st.selectbox("Ano final", finais, index=len(finais)-1, key="lp_fim")
        d = base[base["Ano"].between(ano_ini, ano_fim)]
        row = ultima_linha(d)
        if row is not None:
            cards_rede(row)
        st.plotly_chart(grafico_lp_mat(d, f"Língua Portuguesa × Matemática — {etapa}"), use_container_width=True)
        st.plotly_chart(grafico_lp_mat_n(d, f"LP + Matemática × Nota Média Padronizada (N) — {etapa}"), use_container_width=True)
        st.plotly_chart(grafico_np_ideb(d, f"N × P × IDEB — {etapa}"), use_container_width=True)

    with sub[4]:
        c1, c2 = st.columns(2)
        with c1:
            ano_ini = st.selectbox("Ano inicial", anos_disp, index=0, key="fl_ini")
        finais = [a for a in anos_disp if a >= ano_ini]
        with c2:
            ano_fim = st.selectbox("Ano final", finais, index=len(finais)-1, key="fl_fim")
        d = base[base["Ano"].between(ano_ini, ano_fim)]
        st.plotly_chart(grafico_aprovacao_linhas(d, etapa, f"Taxa de aprovação por série — {etapa}"), use_container_width=True)
        st.plotly_chart(grafico_aprovacao_series_p(d, etapa, f"Aprovação por série × Indicador de Rendimento (P) — {etapa}"), use_container_width=True)

    with sub[5]:
        st.markdown('<div class="section-title">Movimento da rede</div>', unsafe_allow_html=True)
        st.markdown('<div class="note">A base atual não contém diretamente as variáveis de estabilidade/movimento das unidades mostradas no modelo visual. Esta área ficou reservada para essa incorporação.</div>', unsafe_allow_html=True)

    with sub[6]:
        st.dataframe(
            base[["Ano","Rede","Etapa","IDEB","Meta IDEB","Matemática","Língua Portuguesa","N","P","Aprovação Geral"]],
            hide_index=True, use_container_width=True
        )
        inv = investimento[investimento["Etapa"] == etapa].sort_values("Ano")
        if not inv.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=inv["Ano"], y=inv["Investimento por Estudante"],
                mode="lines+markers", name="Investimento por estudante",
                line=dict(color=AZUL, width=3)
            ))
            estilo_fig(fig, f"Investimento por estudante — referência INEP ({etapa})", "R$ por estudante", 430)
            st.plotly_chart(fig, use_container_width=True)

elif pagina == "Escolas":
    st.markdown('<div class="eyebrow">Escolas</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">Painel de consulta às unidades escolares</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">A busca ignora acentos. Ex.: digitar "jose" encontra "José".</div>', unsafe_allow_html=True)

    @st.fragment
    def painel_escolas():
        etapa = st.selectbox("Etapa", ETAPAS, key="esc_etapa")

        busca_escola = st.text_input(
            "Buscar escola",
            placeholder="Digite parte do nome — não é necessário usar acentos",
            key="busca_escola_principal"
        )

        lista = filtrar_nomes_busca(
            escolas,
            "Escola",
            "Escola_Busca",
            busca_escola,
            filtro=(escolas["Etapa"] == etapa)
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

        t1,t2,t3 = st.tabs(["Desempenho","Fluxo e rendimento","Comparar escolas"])

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

    painel_escolas()

elif pagina == "Aprendizagem":
    st.markdown('<div class="eyebrow">Aprendizagem</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">Diagnóstico de Língua Portuguesa e Matemática</div>', unsafe_allow_html=True)

    modo = st.radio("Visão", ["Por rede","Por escola"], horizontal=True, label_visibility="collapsed")
    etapa = st.segmented_control("Etapa", ETAPAS, default="Fundamental I", selection_mode="single") or "Fundamental I"

    if modo == "Por rede":
        base = dados_municipio("Barueri", etapa)
        anos_disp = sorted(int(a) for a in base["Ano"].dropna().unique())
        ano = st.selectbox("Ano de referência", anos_disp, index=len(anos_disp)-1)
        row = ultima_linha(base, ano)
        if row is not None:
            cards_rede(row)
            c1,c2 = st.columns(2)
            c1.markdown(
                f'<div class="metric-card"><div class="metric-label">Língua Portuguesa</div>'
                f'<div class="metric-value">Nível {fmt(row.get("Nível Língua Portuguesa"),0)}</div>'
                f'<div class="metric-foot">{row.get("Padrão Língua Portuguesa","—")}</div></div>',
                unsafe_allow_html=True
            )
            c2.markdown(
                f'<div class="metric-card"><div class="metric-label">Matemática</div>'
                f'<div class="metric-value">Nível {fmt(row.get("Nível Matemática"),0)}</div>'
                f'<div class="metric-foot">{row.get("Padrão Matemática","—")}</div></div>',
                unsafe_allow_html=True
            )
        st.plotly_chart(grafico_lp_mat(base, f"Série histórica das proficiências — {etapa}"), use_container_width=True)
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
                grafico_lp_mat(
                    d,
                    f"Série histórica das proficiências — {escola}"
                ),
                use_container_width=True
            )

elif pagina == "Território":
    st.markdown('<div class="eyebrow">Território</div>', unsafe_allow_html=True)
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
            "Barueri", etapa, ano_ini, ano_fim
        )

        st.markdown(
            '<div class="section-title">Modelo comparativo do Colab — LP e Matemática</div>',
            unsafe_allow_html=True
        )

        if outros:
            comp = outros[0]
            dc = dados_municipio(
                comp, etapa, ano_ini, ano_fim
            )
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
                    nome,
                    dados_municipio(
                        nome, etapa, ano_ini, ano_fim
                    )
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
            '<div class="section-title">N × P × IDEB</div>',
            unsafe_allow_html=True
        )

        ano_ref = st.selectbox(
            "Ano para comparação transversal",
            finais,
            index=len(finais)-1,
            key="np_ano"
        )

        nomes = ["Barueri"] + outros
        rows = []

        for nome in nomes:
            d = dados_municipio(
                nome, etapa, ano_ref, ano_ref
            )
            if not d.empty:
                r = d.iloc[-1]
                rows.append({
                    "Município": nome,
                    "N": r.get("N"),
                    "P": r.get("P"),
                    "IDEB": r.get("IDEB")
                })

        trans = pd.DataFrame(rows)

        if not trans.empty:
            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=trans["Município"],
                y=trans["N"],
                name="N",
                marker_color="#E5A8BA"
            ))

            fig.add_trace(go.Scatter(
                x=trans["Município"],
                y=trans["IDEB"],
                name="IDEB",
                mode="lines+markers+text",
                text=[fmt(v,1) for v in trans["IDEB"]],
                textposition="bottom center",
                line=dict(color=LARANJA,width=3)
            ))

            fig.add_trace(go.Scatter(
                x=trans["Município"],
                y=trans["P"],
                name="P",
                mode="lines+markers+text",
                yaxis="y2",
                text=[fmt(v,3) for v in trans["P"]],
                textposition="top center",
                line=dict(color=LILAS_P,width=3)
            ))

            estilo_fig(
                fig,
                f"N × P × IDEB — {ano_ref} — {etapa}",
                "N / IDEB",
                520
            )

            fig.update_layout(
                yaxis2=dict(
                    title="P",
                    overlaying="y",
                    side="right",
                    showgrid=False
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    painel_territorio()

st.markdown(
    '<div class="footer">Painel educacional • dados organizados a partir das bases SAEB/IDEB fornecidas para o projeto.</div>',
    unsafe_allow_html=True
)
