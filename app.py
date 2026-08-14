import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import unicodedata
from pathlib import Path

# ============================================================
# CONFIGURAÇÃO
# ============================================================
st.set_page_config(
    page_title="Painel de Indicadores Educacionais",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

st.markdown("""
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
[data-testid="stMetric"] {
    border: 1px solid rgba(49,51,63,.18);
    padding: 14px 16px;
    border-radius: 12px;
    background: rgba(250,250,250,.6);
}
h1, h2, h3 {letter-spacing: -0.02em;}
.small-note {font-size:.88rem; opacity:.72;}
</style>
""", unsafe_allow_html=True)


# ============================================================
# CARREGAMENTO + PRÉ-PROCESSAMENTO CACHEADO
# ============================================================
def _normalizar_serie(s: pd.Series) -> pd.Series:
    # Vetorização Pandas; a transformação ocorre somente dentro do cache.
    return (
        s.astype("string")
         .str.normalize("NFKD")
         .str.encode("ascii", errors="ignore")
         .str.decode("utf-8")
         .str.lower()
         .str.strip()
    )


@st.cache_data(show_spinner="Carregando bases...")
def carregar_bases():
    municipios = pd.read_csv(DATA_DIR / "base_municipios.csv", encoding="utf-8-sig")
    escolas = pd.read_csv(DATA_DIR / "base_escolas.csv", encoding="utf-8-sig")
    investimento = pd.read_csv(DATA_DIR / "investimento_inep.csv", encoding="utf-8-sig")

    municipios["Município_Busca"] = _normalizar_serie(municipios["Município"])
    escolas["Escola_Busca"] = _normalizar_serie(escolas["Escola"])

    for df in (municipios, escolas, investimento):
        if "Ano" in df.columns:
            df["Ano"] = pd.to_numeric(df["Ano"], errors="coerce").astype("Int64")

    return municipios, escolas, investimento


municipios, escolas, investimento = carregar_bases()

ANOS = sorted(int(x) for x in municipios["Ano"].dropna().unique())
ETAPAS = [x for x in ["Fundamental I", "Fundamental II"] if x in municipios["Etapa"].dropna().unique()]
INDICADORES = ["IDEB", "Matemática", "Língua Portuguesa", "N", "P", "Aprovação Geral"]
SERIES_FI = ["1º", "2º", "3º", "4º", "5º"]
SERIES_FII = ["6º", "7º", "8º", "9º"]


# ============================================================
# FUNÇÕES
# ============================================================
def barueri_referencia(df, etapa=None):
    x = df[df["Município"].eq("Barueri")]
    if "Rede" in x.columns and (x["Rede"] == "Municipal").any():
        x = x[x["Rede"].eq("Municipal")]
    if etapa:
        x = x[x["Etapa"].eq(etapa)]
    return x.copy()


def filtrar_municipio(nome, etapa, ano_ini, ano_fim, rede=None):
    x = municipios[
        municipios["Município"].eq(nome)
        & municipios["Etapa"].eq(etapa)
        & municipios["Ano"].between(ano_ini, ano_fim)
    ].copy()
    if rede and "Rede" in x.columns:
        x = x[x["Rede"].eq(rede)]
    elif nome == "Barueri" and "Rede" in x.columns and (x["Rede"] == "Municipal").any():
        x = x[x["Rede"].eq("Municipal")]
    return x.sort_values("Ano")


def redes_do_municipio(nome, etapa):
    vals = municipios.loc[
        municipios["Município"].eq(nome) & municipios["Etapa"].eq(etapa), "Rede"
    ].dropna().unique().tolist()
    return sorted(vals)


def fig_linhas_comparacao(datasets, indicador, titulo, y_title=None):
    fig = go.Figure()
    for rotulo, df in datasets:
        if df.empty or indicador not in df:
            continue
        fig.add_trace(go.Scatter(
            x=df["Ano"], y=df[indicador],
            mode="lines+markers",
            name=rotulo,
            connectgaps=False,
            hovertemplate="%{x}<br>%{y:.2f}<extra>%{fullData.name}</extra>",
        ))
    fig.update_layout(
        title=titulo, height=470, hovermode="x unified",
        xaxis=dict(tickmode="array", tickvals=ANOS),
        yaxis_title=y_title or indicador,
        legend_title_text="",
        margin=dict(l=30, r=30, t=70, b=30),
    )
    return fig


def fig_lp_mat(df, titulo):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Ano"], y=df["Língua Portuguesa"],
        mode="lines+markers", name="Língua Portuguesa"
    ))
    fig.add_trace(go.Scatter(
        x=df["Ano"], y=df["Matemática"],
        mode="lines+markers", name="Matemática"
    ))
    fig.update_layout(
        title=titulo, height=460, hovermode="x unified",
        xaxis=dict(tickmode="array", tickvals=ANOS),
        yaxis_title="Proficiência SAEB", legend_title_text=""
    )
    return fig


def fig_lp_mat_n(df, titulo):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["Ano"], y=df["Língua Portuguesa"], name="Língua Portuguesa"))
    fig.add_trace(go.Bar(x=df["Ano"], y=df["Matemática"], name="Matemática"))
    fig.add_trace(go.Scatter(
        x=df["Ano"], y=df["N"], name="Nota Média Padronizada (N)",
        mode="lines+markers", yaxis="y2"
    ))
    fig.update_layout(
        title=titulo, barmode="group", height=490,
        xaxis=dict(tickmode="array", tickvals=ANOS),
        yaxis=dict(title="Proficiência SAEB"),
        yaxis2=dict(title="N", overlaying="y", side="right", showgrid=False),
        hovermode="x unified", legend_title_text=""
    )
    return fig


def fig_aprovacao_p(df, etapa, titulo):
    series = SERIES_FI if etapa == "Fundamental I" else SERIES_FII
    fig = go.Figure()
    for serie in series:
        if serie in df and df[serie].notna().any():
            fig.add_trace(go.Bar(x=df["Ano"], y=df[serie], name=f"Aprovação {serie}"))
    fig.add_trace(go.Scatter(
        x=df["Ano"], y=df["P"], name="Indicador de Rendimento (P)",
        mode="lines+markers", yaxis="y2"
    ))
    fig.update_layout(
        title=titulo, barmode="group", height=500,
        xaxis=dict(tickmode="array", tickvals=ANOS),
        yaxis=dict(title="Taxa de aprovação (%)", range=[0, 105]),
        yaxis2=dict(title="P", overlaying="y", side="right", showgrid=False),
        hovermode="x unified", legend_title_text=""
    )
    return fig


def fig_n_p_ideb(df, titulo):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["Ano"], y=df["N"], name="N"))
    fig.add_trace(go.Scatter(x=df["Ano"], y=df["IDEB"], name="IDEB", mode="lines+markers"))
    fig.add_trace(go.Scatter(
        x=df["Ano"], y=df["P"], name="P", mode="lines+markers", yaxis="y2"
    ))
    fig.update_layout(
        title=titulo, height=490,
        xaxis=dict(tickmode="array", tickvals=ANOS),
        yaxis=dict(title="N / IDEB", rangemode="tozero"),
        yaxis2=dict(title="P", overlaying="y", side="right", showgrid=False),
        hovermode="x unified", legend_title_text=""
    )
    return fig


def ultimo_valor(df, col):
    x = df.dropna(subset=[col]).sort_values("Ano")
    if x.empty:
        return None, None
    r = x.iloc[-1]
    return r[col], int(r["Ano"])


# ============================================================
# CABEÇALHO
# ============================================================
st.title("📊 Painel de Indicadores Educacionais")
st.caption("SAEB • IDEB • Fluxo escolar • Comparações territoriais")

with st.sidebar:
    st.header("Painel")
    pagina = st.radio(
        "Navegação",
        ["Visão da Rede", "Comparações", "Escolas", "Aprendizagem", "Território"],
        label_visibility="collapsed"
    )
    st.divider()
    st.caption(f"Série histórica disponível: {min(ANOS)}–{max(ANOS)}")


# ============================================================
# VISÃO DA REDE
# ============================================================
if pagina == "Visão da Rede":
    st.subheader("Visão da Rede")

    c1, c2, c3 = st.columns([1.4, 1, 1])
    with c1:
        etapa = st.selectbox("Etapa", ETAPAS, key="vr_etapa")
    with c2:
        ano = st.selectbox("Ano de referência", ANOS, index=len(ANOS)-1, key="vr_ano")
    with c3:
        rede = st.selectbox("Rede", redes_do_municipio("Barueri", etapa), key="vr_rede")

    base = filtrar_municipio("Barueri", etapa, min(ANOS), max(ANOS), rede)
    atual = base[base["Ano"].eq(ano)]

    if atual.empty:
        st.info("Não há dados para a combinação selecionada.")
    else:
        row = atual.iloc[-1]
        cols = st.columns(5)
        cards = [
            ("IDEB", row.get("IDEB")),
            ("Língua Portuguesa", row.get("Língua Portuguesa")),
            ("Matemática", row.get("Matemática")),
            ("Aprovação", row.get("Aprovação Geral")),
            ("Rendimento (P)", row.get("P")),
        ]
        for col, (rot, val) in zip(cols, cards):
            col.metric(rot, "—" if pd.isna(val) else f"{val:.2f}")

        st.plotly_chart(
            fig_linhas_comparacao([("Barueri", base)], "IDEB",
                                  f"Série histórica do IDEB — {etapa}", "IDEB"),
            use_container_width=True
        )

        a, b = st.columns(2)
        with a:
            st.plotly_chart(fig_lp_mat(base, f"Aprendizagem — {etapa}"), use_container_width=True)
        with b:
            st.plotly_chart(fig_aprovacao_p(base, etapa, f"Fluxo e rendimento — {etapa}"),
                            use_container_width=True)


# ============================================================
# COMPARAÇÕES
# ============================================================
elif pagina == "Comparações":

    @st.fragment
    def area_comparacoes():
        st.subheader("Comparações entre municípios")
        st.caption("Barueri permanece como referência. Selecione outros municípios e aplique a comparação.")

        with st.form("formulario_comparacao"):
            f1, f2, f3 = st.columns(3)
            with f1:
                etapa = st.selectbox("Etapa", ETAPAS, key="cmp_etapa")
            with f2:
                ano_ini = st.selectbox("Ano inicial", ANOS, index=0, key="cmp_ini")
            with f3:
                finais = [a for a in ANOS if a > ano_ini]
                ano_fim = st.selectbox("Ano final", finais, index=len(finais)-1, key="cmp_fim")

            opcoes = sorted(x for x in municipios["Município"].dropna().unique() if x != "Barueri")
            outros = st.multiselect(
                "Municípios para comparar com Barueri",
                opcoes, max_selections=5,
                placeholder="Selecione até 5 municípios"
            )
            indicador = st.selectbox("Indicador", INDICADORES)
            aplicar = st.form_submit_button("Aplicar comparação", type="primary")

        datasets = []
        bar = filtrar_municipio("Barueri", etapa, ano_ini, ano_fim)
        datasets.append(("Barueri — Municipal", bar))

        for nome in outros:
            redes = redes_do_municipio(nome, etapa)
            # Preferência pela rede municipal quando existir; caso contrário, usa Pública ou a primeira.
            if "Municipal" in redes:
                r = "Municipal"
            elif "Pública" in redes:
                r = "Pública"
            elif redes:
                r = redes[0]
            else:
                r = None
            datasets.append((f"{nome}" + (f" — {r}" if r else ""), filtrar_municipio(nome, etapa, ano_ini, ano_fim, r)))

        st.plotly_chart(
            fig_linhas_comparacao(
                datasets, indicador,
                f"{indicador}: Barueri × municípios — {etapa}",
                indicador
            ),
            use_container_width=True
        )

        st.divider()
        st.markdown("### Modelos de comparação recuperados do Colab")

        somente_bar = bar
        if not somente_bar.empty:
            t1, t2 = st.tabs(["LP × Matemática × N", "Aprovação por série × P"])
            with t1:
                st.plotly_chart(
                    fig_lp_mat_n(somente_bar, f"Barueri — {etapa}: LP + Matemática × N"),
                    use_container_width=True
                )
            with t2:
                st.plotly_chart(
                    fig_aprovacao_p(somente_bar, etapa, f"Barueri — {etapa}: aprovação por série × P"),
                    use_container_width=True
                )

        st.markdown("### N × P × IDEB")
        st.plotly_chart(
            fig_n_p_ideb(somente_bar, f"Barueri — {etapa}: componentes do IDEB"),
            use_container_width=True
        )

    area_comparacoes()


# ============================================================
# ESCOLAS
# ============================================================
elif pagina == "Escolas":

    @st.fragment
    def area_escolas():
        st.subheader("Consulta por unidade escolar")

        with st.form("form_escolas"):
            etapa = st.selectbox("Etapa", ETAPAS, key="esc_etapa")
            nomes = sorted(escolas.loc[escolas["Etapa"].eq(etapa), "Escola"].dropna().unique())
            escola = st.selectbox("Escola", nomes)
            ano_ini, ano_fim = st.select_slider(
                "Período", options=ANOS, value=(min(ANOS), max(ANOS))
            )
            st.form_submit_button("Aplicar", type="primary")

        df = escolas[
            escolas["Escola"].eq(escola)
            & escolas["Etapa"].eq(etapa)
            & escolas["Ano"].between(ano_ini, ano_fim)
        ].sort_values("Ano")

        v_ideb, a_ideb = ultimo_valor(df, "IDEB")
        v_lp, a_lp = ultimo_valor(df, "Língua Portuguesa")
        v_mat, a_mat = ultimo_valor(df, "Matemática")
        m1, m2, m3 = st.columns(3)
        m1.metric(f"IDEB ({a_ideb or '—'})", "—" if v_ideb is None else f"{v_ideb:.2f}")
        m2.metric(f"LP ({a_lp or '—'})", "—" if v_lp is None else f"{v_lp:.2f}")
        m3.metric(f"Matemática ({a_mat or '—'})", "—" if v_mat is None else f"{v_mat:.2f}")

        st.plotly_chart(fig_lp_mat(df, f"{escola} — LP × Matemática"), use_container_width=True)

        a, b = st.columns(2)
        with a:
            st.plotly_chart(fig_aprovacao_p(df, etapa, "Aprovação por série × P"), use_container_width=True)
        with b:
            st.plotly_chart(fig_n_p_ideb(df, "N × P × IDEB"), use_container_width=True)

        st.markdown("### Comparar duas escolas")
        nomes2 = sorted(escolas.loc[escolas["Etapa"].eq(etapa), "Escola"].dropna().unique())
        c1, c2 = st.columns(2)
        e1 = c1.selectbox("Escola 1", nomes2, index=0, key="e1")
        e2 = c2.selectbox("Escola 2", nomes2, index=min(1, len(nomes2)-1), key="e2")
        d1 = escolas[(escolas["Escola"].eq(e1)) & (escolas["Etapa"].eq(etapa)) & escolas["Ano"].between(ano_ini, ano_fim)]
        d2 = escolas[(escolas["Escola"].eq(e2)) & (escolas["Etapa"].eq(etapa)) & escolas["Ano"].between(ano_ini, ano_fim)]
        ind = st.selectbox("Indicador da comparação", INDICADORES, key="ind_esc")
        st.plotly_chart(
            fig_linhas_comparacao([(e1, d1), (e2, d2)], ind, f"{ind}: comparação entre escolas"),
            use_container_width=True
        )

    area_escolas()


# ============================================================
# APRENDIZAGEM
# ============================================================
elif pagina == "Aprendizagem":
    st.subheader("Aprendizagem")
    etapa = st.selectbox("Etapa", ETAPAS, key="apr_etapa")
    ano = st.selectbox("Ano", ANOS, index=len(ANOS)-1, key="apr_ano")

    base = municipios[(municipios["Etapa"].eq(etapa)) & (municipios["Ano"].eq(ano))].copy()
    # Uma linha representativa por município, priorizando Municipal.
    base["_prio"] = base["Rede"].map({"Municipal": 0, "Pública": 1, "Estadual": 2, "Federal": 3}).fillna(9)
    base = base.sort_values(["Município", "_prio"]).drop_duplicates("Município")

    c1, c2 = st.columns(2)
    with c1:
        top = base.nlargest(20, "Língua Portuguesa")
        fig = px.bar(top.sort_values("Língua Portuguesa"), x="Língua Portuguesa", y="Município",
                     orientation="h", title=f"20 maiores resultados — Língua Portuguesa ({ano})")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        top = base.nlargest(20, "Matemática")
        fig = px.bar(top.sort_values("Matemática"), x="Matemática", y="Município",
                     orientation="h", title=f"20 maiores resultados — Matemática ({ano})")
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# TERRITÓRIO
# ============================================================
elif pagina == "Território":

    @st.fragment
    def area_territorio():
        st.subheader("Comparação territorial")
        st.caption("Compare Barueri com outros municípios usando os indicadores disponíveis na base completa.")

        with st.form("form_territorio"):
            etapa = st.selectbox("Etapa", ETAPAS, key="ter_etapa")
            ano = st.selectbox("Ano", ANOS, index=len(ANOS)-1, key="ter_ano")
            opcoes = sorted(x for x in municipios["Município"].dropna().unique() if x != "Barueri")
            comps = st.multiselect("Municípios", opcoes, max_selections=5)
            st.form_submit_button("Atualizar território", type="primary")

        nomes = ["Barueri"] + comps
        linhas = []
        for nome in nomes:
            redes = redes_do_municipio(nome, etapa)
            rede = "Municipal" if "Municipal" in redes else ("Pública" if "Pública" in redes else (redes[0] if redes else None))
            x = filtrar_municipio(nome, etapa, ano, ano, rede)
            if not x.empty:
                r = x.iloc[-1].copy()
                r["Município Exibido"] = nome
                linhas.append(r)

        if linhas:
            comp = pd.DataFrame(linhas)
            fig = go.Figure()
            fig.add_trace(go.Bar(x=comp["Município Exibido"], y=comp["IDEB"], name="IDEB"))
            fig.add_trace(go.Scatter(
                x=comp["Município Exibido"], y=comp["Aprovação Geral"],
                name="Aprovação Geral", mode="lines+markers", yaxis="y2"
            ))
            fig.update_layout(
                title=f"IDEB × Aprovação Geral — {etapa}, {ano}",
                height=480,
                yaxis=dict(title="IDEB"),
                yaxis2=dict(title="Aprovação (%)", overlaying="y", side="right", range=[0,105]),
                legend_title_text=""
            )
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                comp[["Município Exibido", "Rede", "IDEB", "Matemática", "Língua Portuguesa", "N", "P", "Aprovação Geral"]],
                use_container_width=True, hide_index=True
            )

        st.markdown("### Investimento por estudante — referência INEP")
        inv = investimento[investimento["Etapa"].eq(etapa)].sort_values("Ano")
        if not inv.empty:
            fig = px.line(inv, x="Ano", y="Investimento por Estudante", markers=True,
                          title=f"Investimento por estudante — {etapa}")
            st.plotly_chart(fig, use_container_width=True)

    area_territorio()
