"""
Agente de Crédito PJ com LLM — Aplicação Streamlit
==================================================

Fluxo do produto:
    1. O usuário (analista) insere os dados de uma empresa (PJ).
    2. O modelo de ML (Random Forest persistido) calcula a Probabilidade de
       Default (PD) -> score (0-1000) -> decisão (Aprovar / Análise manual / Negar).
    3. A API da Anthropic (Claude) gera um PARECER DE CRÉDITO em linguagem natural,
       coerente com o resultado quantitativo e explicável para o comitê de crédito.

Execução:
    pip install -r requirements.txt
    # defina a chave (uma das opções):
    #   - variável de ambiente:  setx ANTHROPIC_API_KEY "sk-ant-..."
    #   - app/.streamlit/secrets.toml  ->  ANTHROPIC_API_KEY = "sk-ant-..."
    #   - campo na barra lateral do app
    streamlit run app/app_credito.py

Sem chave/SDK o app continua funcional em "modo demonstração" (parecer baseado em regras).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Caminhos e constantes
# ---------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parents[1]
MODELO_PATH = RAIZ / "modelos" / "pipeline_credito.joblib"
SCHEMA_PATH = RAIZ / "modelos" / "schema_features.json"
METRICAS_PATH = RAIZ / "modelos" / "metricas.json"
MODELO_LLM = "claude-sonnet-4-6"

st.set_page_config(page_title="Agente de Crédito PJ • LLM",
                   page_icon="🏦", layout="wide")


# ---------------------------------------------------------------------------
# Carregamento de artefatos (cacheado)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def carregar_modelo():
    return joblib.load(MODELO_PATH)


@st.cache_data(show_spinner=False)
def carregar_json(caminho: Path) -> dict:
    return json.loads(Path(caminho).read_text(encoding="utf-8"))


def obter_api_key(entrada_manual: str) -> str:
    """Prioridade: campo na tela > variável de ambiente > st.secrets."""
    if entrada_manual:
        return entrada_manual
    if os.environ.get("ANTHROPIC_API_KEY"):
        return os.environ["ANTHROPIC_API_KEY"]
    try:  # st.secrets lança exceção se não houver secrets.toml
        return st.secrets.get("ANTHROPIC_API_KEY", "")
    except Exception:  # noqa: BLE001
        return ""


# ---------------------------------------------------------------------------
# Lógica de decisão
# ---------------------------------------------------------------------------
def decidir(pd_default: float, pd_aprovar: float, pd_negar: float) -> tuple[str, str]:
    """Converte a PD em decisão e cor de exibição."""
    if pd_default <= pd_aprovar:
        return "APROVAR", "green"
    if pd_default >= pd_negar:
        return "NEGAR", "red"
    return "ANÁLISE MANUAL", "orange"


def fatores_de_risco(dados: dict, schema: dict) -> list[str]:
    """Heurística simples para destacar drivers ao usuário e ao LLM."""
    faixas = schema["faixas_numericas"]
    flags = []
    if dados["score_serasa"] < 500:
        flags.append(f"Score Serasa baixo ({dados['score_serasa']})")
    if dados["restricao_serasa"] == 1:
        flags.append("Possui restrição ativa no Serasa (negativação)")
    if dados["num_atrasos_12m"] >= 2:
        flags.append(f"{dados['num_atrasos_12m']} atrasos nos últimos 12 meses")
    if dados["razao_divida_faturamento"] > 0.6:
        flags.append(f"Alavancagem elevada (dívida/faturamento = "
                     f"{dados['razao_divida_faturamento']:.2f})")
    if dados["margem_liquida"] < 0.03:
        flags.append(f"Margem líquida apertada ({dados['margem_liquida']:.1%})")
    if dados["indice_liquidez_corrente"] < 1.0:
        flags.append(f"Liquidez corrente abaixo de 1,0 "
                     f"({dados['indice_liquidez_corrente']:.2f})")
    if dados["tempo_atividade_anos"] < 2:
        flags.append(f"Empresa jovem ({dados['tempo_atividade_anos']:.1f} anos)")
    return flags or ["Sem alertas relevantes nos indicadores observados"]


def pontos_positivos(dados: dict) -> list[str]:
    bons = []
    if dados["score_serasa"] >= 700:
        bons.append(f"Score Serasa robusto ({dados['score_serasa']})")
    if dados["restricao_serasa"] == 0 and dados["num_atrasos_12m"] == 0:
        bons.append("Histórico limpo: sem restrições nem atrasos")
    if dados["margem_liquida"] >= 0.10:
        bons.append(f"Boa rentabilidade (margem líquida {dados['margem_liquida']:.1%})")
    if dados["indice_liquidez_corrente"] >= 1.5:
        bons.append(f"Liquidez confortável ({dados['indice_liquidez_corrente']:.2f})")
    if dados["tempo_atividade_anos"] >= 10:
        bons.append(f"Maturidade operacional ({dados['tempo_atividade_anos']:.0f} anos)")
    return bons or ["Sem destaques positivos relevantes"]


# ---------------------------------------------------------------------------
# Camada de IA Generativa (Anthropic)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "Você é um analista de crédito sênior de um banco médio brasileiro, especialista "
    "em risco de Pessoa Jurídica (PJ). Escreva pareceres técnicos, objetivos e em "
    "português do Brasil. Você RECEBE a decisão quantitativa de um modelo estatístico "
    "(PD, score e decisão) e NÃO deve contradizê-la — seu papel é explicá-la de forma "
    "clara para o comitê de crédito, fundamentando-a nos indicadores da empresa. "
    "Estruture o parecer em: (1) Síntese da recomendação; (2) Principais fatores de "
    "risco; (3) Pontos positivos; (4) Condicionantes e mitigadores sugeridos "
    "(garantias, covenants, limite/prazo); (5) Ressalva de que a decisão final cabe "
    "à alçada competente. Seja conciso (300–450 palavras) e profissional."
)


def montar_prompt(dados: dict, razao_social: str, pd_default: float, score: int,
                  decisao: str, riscos: list[str], positivos: list[str]) -> str:
    return (
        f"Gere o parecer de crédito para a empresa **{razao_social}**.\n\n"
        f"### Resultado do modelo de risco (Random Forest)\n"
        f"- Probabilidade de Default (PD): {pd_default:.1%}\n"
        f"- Score de crédito (0–1000): {score}\n"
        f"- Decisão recomendada pelo modelo: {decisao}\n\n"
        f"### Dados cadastrais e financeiros\n"
        f"- Setor: {dados['setor']} | Região: {dados['regiao']} | Porte: {dados['porte']}\n"
        f"- Tempo de atividade: {dados['tempo_atividade_anos']} anos | "
        f"Funcionários: {dados['num_funcionarios']}\n"
        f"- Faturamento anual: R$ {dados['faturamento_anual']:,.0f}\n"
        f"- Dívida total: R$ {dados['divida_total']:,.0f} "
        f"(dívida/faturamento = {dados['razao_divida_faturamento']:.2f})\n"
        f"- Margem líquida: {dados['margem_liquida']:.1%} | "
        f"Liquidez corrente: {dados['indice_liquidez_corrente']:.2f}\n"
        f"- Score Serasa: {dados['score_serasa']} | "
        f"Atrasos 12m: {dados['num_atrasos_12m']} | "
        f"Restrição ativa: {'Sim' if dados['restricao_serasa'] else 'Não'}\n\n"
        f"### Sinais já identificados\n"
        f"- Fatores de risco: {'; '.join(riscos)}\n"
        f"- Pontos positivos: {'; '.join(positivos)}\n"
    ).replace(",", ".")  # formato pt-BR para milhares


def gerar_parecer_llm(prompt: str, api_key: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=MODELO_LLM,
        max_tokens=1200,
        system=[{"type": "text", "text": SYSTEM_PROMPT,
                 "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


def parecer_demonstracao(razao_social: str, decisao: str, pd_default: float,
                         score: int, riscos: list[str], positivos: list[str]) -> str:
    """Parecer baseado em regras, usado quando não há chave/SDK da Anthropic."""
    return (
        f"**Síntese da recomendação** — Para **{razao_social}**, o modelo aponta "
        f"probabilidade de inadimplência de **{pd_default:.1%}** (score {score}), "
        f"resultando na decisão **{decisao}**.\n\n"
        f"**Principais fatores de risco**\n" +
        "".join(f"- {r}\n" for r in riscos) +
        f"\n**Pontos positivos**\n" +
        "".join(f"- {p}\n" for p in positivos) +
        f"\n**Condicionantes sugeridas** — Caso de análise/aprovação, avaliar garantias "
        f"reais ou aval dos sócios, covenants de alavancagem e limite compatível com o "
        f"faturamento.\n\n"
        f"*Modo demonstração (sem API). Configure a chave Anthropic para o parecer "
        f"gerado por IA. A decisão final cabe à alçada competente.*"
    )


# ===========================================================================
# INTERFACE
# ===========================================================================
def main() -> None:
    # ---- Verifica artefatos ----
    if not MODELO_PATH.exists():
        st.error("Modelo não encontrado. Rode antes: `python src/treinar_modelos.py`")
        st.stop()

    modelo = carregar_modelo()
    schema = carregar_json(SCHEMA_PATH)
    metricas = carregar_json(METRICAS_PATH) if METRICAS_PATH.exists() else {}
    pol = schema["politica"]

    # ---- Sidebar ----
    with st.sidebar:
        st.header("⚙️ Configuração")
        entrada_manual = st.text_input("Anthropic API Key", type="password",
                                       help="Ou defina ANTHROPIC_API_KEY / st.secrets")
        api_key = obter_api_key(entrada_manual)
        usar_llm = st.toggle("Usar IA Generativa (Anthropic)", value=bool(api_key))
        st.divider()
        st.caption("**Modelo de produção**")
        if metricas:
            m = metricas["metricas"][metricas["modelo_producao"]]
            st.metric("Modelo", metricas["modelo_producao"])
            c1, c2, c3 = st.columns(3)
            c1.metric("KS", m["ks"]); c2.metric("AUC", m["auc"]); c3.metric("Gini", m["gini"])
        st.caption(f"Política: PD ≤ {pol['pd_aprovar']:.0%} aprova • "
                   f"PD > {pol['pd_negar']:.0%} nega")

    # ---- Cabeçalho ----
    st.title("🏦 Agente de Crédito PJ com LLM")
    st.markdown("Análise de crédito **Pessoa Jurídica** combinando *machine learning* "
                "(score) e *IA generativa* (parecer em linguagem natural).")

    # ---- Formulário ----
    op = schema["opcoes_categoricas"]
    fx = schema["faixas_numericas"]
    with st.form("dados_cliente"):
        razao_social = st.text_input("Razão social", "Empresa Exemplo Ltda")

        st.subheader("Cadastro")
        c = st.columns(3)
        setor = c[0].selectbox("Setor", op["setor"])
        regiao = c[1].selectbox("Região", op["regiao"])
        porte = c[2].selectbox("Porte", op["porte"])
        c = st.columns(2)
        tempo = c[0].number_input("Tempo de atividade (anos)", 0.0, 60.0, 8.0, 0.5)
        funcionarios = c[1].number_input("Nº de funcionários", 1, 50000, 25)

        st.subheader("Indicadores financeiros")
        c = st.columns(2)
        faturamento = c[0].number_input("Faturamento anual (R$)", 0.0, 1e9,
                                        float(fx["faturamento_anual"]["mediana"]), 10000.0)
        divida = c[1].number_input("Dívida total (R$)", 0.0, 1e9,
                                   float(fx["divida_total"]["mediana"]), 10000.0)
        c = st.columns(2)
        margem = c[0].slider("Margem líquida", -0.15, 0.45, 0.09, 0.01)
        liquidez = c[1].slider("Índice de liquidez corrente", 0.2, 5.0, 1.2, 0.1)

        st.subheader("Comportamento de crédito")
        c = st.columns(3)
        score_serasa = c[0].slider("Score Serasa", 0, 1000, 600, 10)
        atrasos = c[1].number_input("Atrasos (12m)", 0, 24, 0)
        restricao = 1 if c[2].selectbox("Restrição ativa?", ["Não", "Sim"]) == "Sim" else 0

        enviar = st.form_submit_button("🔎 Analisar crédito", type="primary",
                                       use_container_width=True)

    if not enviar:
        st.info("Preencha os dados da empresa e clique em **Analisar crédito**.")
        return

    # ---- Predição ----
    razao_df = round(divida / faturamento, 3) if faturamento > 0 else 0.0
    dados = {
        "setor": setor, "regiao": regiao, "porte": porte,
        "tempo_atividade_anos": tempo, "num_funcionarios": funcionarios,
        "faturamento_anual": faturamento, "divida_total": divida,
        "razao_divida_faturamento": razao_df, "margem_liquida": margem,
        "indice_liquidez_corrente": liquidez, "score_serasa": score_serasa,
        "num_atrasos_12m": atrasos, "restricao_serasa": restricao,
    }
    X = pd.DataFrame([dados])[schema["numericas"] + schema["categoricas"]]
    pd_default = float(modelo.predict_proba(X)[0, 1])
    score = round((1 - pd_default) * 1000)
    decisao, cor = decidir(pd_default, pol["pd_aprovar"], pol["pd_negar"])
    riscos = fatores_de_risco(dados, schema)
    positivos = pontos_positivos(dados)

    # ---- Resultado quantitativo ----
    st.divider()
    k1, k2, k3 = st.columns(3)
    k1.metric("Score de crédito", f"{score}/1000")
    k2.metric("Prob. de Default (PD)", f"{pd_default:.1%}")
    k3.markdown(f"### Decisão\n<h2 style='color:{cor};margin-top:-8px'>{decisao}</h2>",
                unsafe_allow_html=True)
    st.progress(min(score / 1000, 1.0))

    # ---- Parecer (LLM ou demonstração) ----
    st.subheader("📝 Parecer de crédito")
    if usar_llm and api_key:
        prompt = montar_prompt(dados, razao_social, pd_default, score, decisao,
                               riscos, positivos)
        try:
            with st.spinner("Gerando parecer com IA generativa (Claude)..."):
                texto = gerar_parecer_llm(prompt, api_key)
            st.markdown(texto)
        except Exception as e:  # noqa: BLE001
            st.warning(f"Falha ao chamar a API Anthropic ({e}). Exibindo parecer "
                       f"em modo demonstração.")
            st.markdown(parecer_demonstracao(razao_social, decisao, pd_default,
                                             score, riscos, positivos))
    else:
        st.markdown(parecer_demonstracao(razao_social, decisao, pd_default,
                                         score, riscos, positivos))

    with st.expander("Ver sinais detectados pelo motor de regras"):
        st.write("**Fatores de risco:**", riscos)
        st.write("**Pontos positivos:**", positivos)


if __name__ == "__main__":
    main()
