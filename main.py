"""Auditoria e preparação reprodutível dos metadados clínicos de ECG.

Execução: python main.py
As decisões de preparação são conservadoras: não se removem registos; os
valores originais são preservados e as variáveis analíticas são acrescentadas.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "dados_ecg.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

COLUNAS_OBRIGATORIAS = [
    "registo_id", "doente_id", "idade", "sexo", "altura_cm", "peso_kg",
    "centro_codigo", "dispositivo", "data_registo", "segunda_opiniao",
    "validado_humano", "deriva_baseline", "ruido_estatico", "ruido_rajada",
    "problemas_eletrodos",
]
CAMPOS_QUALIDADE = [
    "deriva_baseline", "ruido_estatico", "ruido_rajada", "problemas_eletrodos"
]


def carregar_dados(caminho=DATA_FILE):
    """Carrega o CSV e produz erros explícitos em caso de problemas."""
    if not caminho.exists():
        raise FileNotFoundError(
            f"Não foi encontrado o ficheiro de entrada: {caminho}\n"
            "Confirme que dados_ecg.csv está na pasta data/."
        )
    try:
        return pd.read_csv(caminho)
    except (pd.errors.ParserError, UnicodeDecodeError, OSError) as exc:
        raise RuntimeError(f"Não foi possível ler o CSV: {exc}") from exc


def validar_estrutura(df):
    """Verifica colunas, identificadores e conversões numéricas essenciais."""
    em_falta = [c for c in COLUNAS_OBRIGATORIAS if c not in df.columns]
    if em_falta:
        raise ValueError(f"Colunas obrigatórias em falta: {em_falta}")
    if df.empty:
        raise ValueError("O ficheiro não contém registos.")
    for coluna in ["idade", "altura_cm", "peso_kg"]:
        convertido = pd.to_numeric(df[coluna], errors="coerce")
        conversoes_falhadas = df[coluna].notna() & convertido.isna()
        if conversoes_falhadas.any():
            raise ValueError(
                f"A coluna {coluna!r} contém {int(conversoes_falhadas.sum())} "
                "valores não nulos que não podem ser convertidos em números."
            )
        df[coluna] = convertido
    return df


def tabela_missing(df):
    """Resumo de valores em falta por variável."""
    resultado = pd.DataFrame({
        "variavel": df.columns,
        "n_missing": df.isna().sum().to_numpy(),
        "percentagem_missing": (df.isna().mean().to_numpy() * 100),
    })
    return resultado.sort_values(
        ["percentagem_missing", "variavel"], ascending=[False, True]
    ).reset_index(drop=True)


def estatisticas_numericas(serie):
    """Devolve estatísticas descritivas sem alterar a série original."""
    s = pd.to_numeric(serie, errors="coerce")
    return {
        "n_observado": int(s.count()),
        "media": s.mean(),
        "mediana": s.median(),
        "desvio_padrao": s.std(),
        "minimo": s.min(),
        "Q1": s.quantile(0.25),
        "Q2_mediana": s.quantile(0.50),
        "Q3": s.quantile(0.75),
        "maximo": s.max(),
    }


def criar_variaveis(df):
    """Acrescenta variáveis analíticas, preservando todos os campos originais."""
    preparado = df.copy()

    # Idade: o valor 300 é uma anomalia de codificação provável, não uma idade
    # interpretável. Não é possível inferir a idade verdadeira apenas deste CSV.
    idade_valida = preparado["idade"].between(0, 120, inclusive="both")
    preparado["idade_analitica"] = preparado["idade"].where(idade_valida)

    limites_idade = [0, 18, 35, 50, 65, 121]
    rotulos_idade = ["0–17", "18–34", "35–49", "50–64", "65+"]

    preparado["grupo_etario"] = pd.cut(
        preparado["idade_analitica"],
        bins=limites_idade,
        labels=rotulos_idade,
        right=False,
        include_lowest=True
    )

    # Identificação de pesos potencialmente suspeitos.
    # A idade tem de ser válida e igual ou superior a 10 anos para sinalizar.
    preparado["peso_suspeito"] = (
        preparado["peso_kg"].lt(30)
        & preparado["idade_analitica"].ge(10)
    ).fillna(False).astype(bool)

    # Critério antropométrico: altura entre 50 e 220 cm e peso entre 1 e 350 kg.
    # Estes limites evitam valores fora dos intervalos definidos para o cálculo.
    # Não constituem um diagnóstico ou juízo clínico.
    altura_utilizavel = preparado["altura_cm"].between(
        50, 220, inclusive="both"
    )

    peso_utilizavel = preparado["peso_kg"].between(
        1, 350, inclusive="both"
    )

    antropometria_utilizavel = altura_utilizavel & peso_utilizavel

    altura_m = preparado["altura_cm"] / 100.0
    imc = preparado["peso_kg"] / altura_m.pow(2)

    preparado["imc"] = imc.where(
        antropometria_utilizavel & np.isfinite(imc)
    )

    # Criação dos quartis de IMC.
    preparado["imc_quartil"] = pd.Series(
        pd.NA,
        index=preparado.index,
        dtype="object"
    )

    mascara_imc = (
        preparado["imc"].notna()
        & np.isfinite(preparado["imc"])
    )

    if int(mascara_imc.sum()) >= 4:
        try:
            quartis = pd.qcut(
                preparado.loc[mascara_imc, "imc"],
                q=4,
                labels=["Q1", "Q2", "Q3", "Q4"],
                duplicates="drop"
            )

            preparado.loc[
                mascara_imc, "imc_quartil"
            ] = quartis.astype("object")

        except ValueError as exc:
            raise ValueError(
                f"Não foi possível criar quartis de IMC: {exc}"
            ) from exc

    # Uma anotação existe quando pelo menos um dos quatro campos não é NA
    anotacoes = preparado[CAMPOS_QUALIDADE].apply(
        lambda coluna: (
            coluna.notna()
            & coluna.astype("string").str.strip().ne("")
        )
    )

    preparado["tem_anotacao_qualidade"] = (
        anotacoes.any(axis=1).astype(bool)
    )

    return preparado


def frequencias_categoricas(df):
    """Gera tabelas de frequências para as variáveis categóricas pedidas."""
    variaveis = ["sexo", "centro_codigo", "dispositivo", "segunda_opiniao", "validado_humano"]
    tabelas = []
    for coluna in variaveis:
        contagens = df[coluna].value_counts(dropna=False)
        for categoria, n in contagens.items():
            tabelas.append({
                "variavel": coluna,
                "categoria": "<EM FALTA>" if pd.isna(categoria) else str(categoria),
                "n": int(n),
                "percentagem": float(n / len(df) * 100),
            })
    return pd.DataFrame(tabelas)


def guardar_tabela(tabela, nome):
    """
     Guarda uma tabela num ficheiro CSV na pasta definida por OUTPUT_DIR.
     em codificação UTF-8 (utf-8-sig) para garantir uma melhor compatibilidade com o Excel.
     """
    tabela.to_csv(OUTPUT_DIR / nome, index=False, encoding="utf-8-sig")


def fmt_num(valor, casas=2):
    """
    Formata um valor numérico com um número definido de casas decimais.
    Caso o valor seja NA, devolve a expressão 'não disponível'.
    Parâmetros:
        valor: valor numérico a formatar.
        casas: número de casas decimais pretendido (por defeito,2).
    Devolve:
        Uma string com o valor formatado ou 'não disponível'.
    """
    if pd.isna(valor):
        return "não disponível"
    return f"{valor:.{casas}f}"




def analisar_e_exportar(original, preparado, missing):
    """Exporta apenas os três ficheiros finais do projeto."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Garante que outputs contém apenas os três resultados pretendidos
    ficheiros_permitidos = {
        "auditoria_missing.csv",
        "resumo_qualidade.txt",
        "dados_ecg_preparados.csv",
    }

    for ficheiro in OUTPUT_DIR.iterdir():
        if ficheiro.is_file() and ficheiro.name not in ficheiros_permitidos:
            ficheiro.unlink()

    # Auditoria dos valores em falta
    guardar_tabela(missing, "auditoria_missing.csv")

    n = len(original)
    n_pacientes = int(original["doente_id"].nunique())
    duplicados_completos = int(original.duplicated().sum())

    ids_duplicados_linhas = int(
        original["registo_id"].duplicated(keep=False).sum()
    )

    ids_duplicados_distintos = int(
        original.loc[
            original["registo_id"].duplicated(keep=False),
            "registo_id"
        ].nunique()
    )

    registos_por_doente = original.groupby(
        "doente_id", dropna=False
    ).size()

    idade_invalida = ~original["idade"].between(
        0, 120, inclusive="both"
    )

    altura_em_falta = int(original["altura_cm"].isna().sum())
    peso_em_falta = int(original["peso_kg"].isna().sum())

    # Contagem de pesos suspeitos
    pesos_suspeitos = int(preparado["peso_suspeito"].sum())

    # Estatísticas necessárias para o relatório
    stats_idade_original = estatisticas_numericas(original["idade"])
    stats_idade_analitica = estatisticas_numericas(
        preparado["idade_analitica"]
    )
    stats_imc = estatisticas_numericas(preparado["imc"])
    # Variaveis complementares para a análise
    anotados = int(preparado["tem_anotacao_qualidade"].sum())
    validados = int(preparado["validado_humano"].sum())

    ambas = int(
        (
            original["altura_cm"].notna()
            & original["peso_kg"].notna()
        ).sum()
    )

    alturas_fora = int(
        (
            original["altura_cm"].notna()
            & ~original["altura_cm"].between(
                50, 220, inclusive="both"
            )
        ).sum()
    )

    pesos_fora = int(
        (
            original["peso_kg"].notna()
            & ~original["peso_kg"].between(
                1, 350, inclusive="both"
            )
        ).sum()
    )

    idade_300_registos = int((original["idade"] == 300).sum())

    idade_300_doentes = int(
        original.loc[
            original["idade"] == 300, "doente_id"
        ].nunique()
    )

    n_singletons = int((registos_por_doente == 1).sum())
    n_multiplos = int((registos_por_doente > 1).sum())
    n_imc = int(preparado["imc"].notna().sum())

    # Contagem de registos por quartil de IMC
    contagem_quartis_imc = (
        preparado["imc_quartil"]
        .value_counts()
        .reindex(["Q1", "Q2", "Q3", "Q4"], fill_value=0)
    )

    registos_sem_quartil_imc = int(
        preparado["imc_quartil"].isna().sum()
    )

    # Diagnóstico NumPy das variáveis antropométricas (Operações Vetoriais)
    array_antropometria = preparado[
        ["altura_cm", "peso_kg"]
    ].to_numpy(dtype=float)

    diagnostico_numpy = {
        "shape": array_antropometria.shape,
        "ndim": array_antropometria.ndim,
        "dtype": str(array_antropometria.dtype),
        "valores_finitos": int(
            np.isfinite(array_antropometria).sum()
        ),
    }

    # Validação final antes de exportar os dados preparados
    assert len(preparado) == len(original), (
        "O número de registos mudou inesperadamente."
    )

    assert preparado["registo_id"].equals(original["registo_id"]), (
        "A ordem ou os IDs dos registos mudaram."
    )

    assert "idade" in preparado.columns
    assert "idade_analitica" in preparado.columns
    assert preparado["tem_anotacao_qualidade"].notna().all()
    assert preparado["imc"].dropna().map(np.isfinite).all()

    assert (
        preparado["registo_id"].is_unique
        == original["registo_id"].is_unique
    )

    # Dados preparados
    preparado.to_csv(
        OUTPUT_DIR / "dados_ecg_preparados.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Relatório da auditoria
    linhas = [
        "RELATÓRIO COMPUTACIONAL — AUDITORIA E PREPARAÇÃO DE DADOS ECG",
        "=" * 68,
        f"Ficheiro de entrada: {DATA_FILE.name}",
        f"Número inicial de registos: {len(original)}",
        f"Número final de registos: {len(preparado)}",
        f"Indivíduos únicos: {n_pacientes}",
        f"Registos por indivíduo → máximo: {int(registos_por_doente.max())}",
        f"Indivíduos com um único registo: {n_singletons}",
        f"Indivíduos com mais de um registo: {n_multiplos}",
        f"Duplicados completos identificados: {duplicados_completos}",
        f"IDs de registo duplicados (grupo registo_id): {ids_duplicados_linhas}",
        f"IDs de registo duplicados distintos: {ids_duplicados_distintos}",
        "",
        "IDADE",
        f"Registos com idade fora do intervalo 0–120: {int(idade_invalida.sum())}",
        f"Registos com idade == 300: {idade_300_registos}; indivíduos afetados: {idade_300_doentes}",
        "Decisão: preservar a idade original e atribuir NA a valores não interpretáveis em idade_analitica.",
        "Justificação: o CSV não permite recuperar a idade verdadeira a partir do código 300.",
        f"Idade original → n={stats_idade_original['n_observado']}, média={fmt_num(stats_idade_original['media'])}, mediana={fmt_num(stats_idade_original['mediana'])}, mínimo={fmt_num(stats_idade_original['minimo'])}, máximo={fmt_num(stats_idade_original['maximo'])}",
        f"Idade analítica → n={stats_idade_analitica['n_observado']}, média={fmt_num(stats_idade_analitica['media'])}, mediana={fmt_num(stats_idade_analitica['mediana'])}, mínimo={fmt_num(stats_idade_analitica['minimo'])}, máximo={fmt_num(stats_idade_analitica['maximo'])}",
        "Intervalos etários: [0,18), [18,35), [35,50), [50,65), [65,121)",
        "",
        "ANTROPOMETRIA E IMC",
        f"Registos com altura em falta: {altura_em_falta} ({altura_em_falta/n*100:.2f}%)",
        f"Registos com peso em falta: {peso_em_falta} ({peso_em_falta/n*100:.2f}%)",
        f"Registos com pesos suspeitos (peso inferior a 30 kg e idade >= 10 anos): {pesos_suspeitos} ({pesos_suspeitos / n * 100:.2f}%)",
        f"Alturas fora do critério 50–220 cm: {alturas_fora}; pesos fora do critério 1–350 kg: {pesos_fora}",
        "Critério de cálculo: altura entre 50 e 220 cm e peso entre 1 e 350 kg; nenhum registo é eliminado.",
        f"Registos com IMC calculável: {n_imc} ({n_imc/n*100:.2f}%)",
        f"IMC → média={fmt_num(stats_imc['media'])}, mediana={fmt_num(stats_imc['mediana'])}, desvio-padrão={fmt_num(stats_imc['desvio_padrao'])}, mínimo={fmt_num(stats_imc['minimo'])}, máximo={fmt_num(stats_imc['maximo'])}",
        "",
        "REGISTOS POR QUARTIL DE IMC",
        f"Q1: {int(contagem_quartis_imc['Q1'])} registos",
        f"Q2: {int(contagem_quartis_imc['Q2'])} registos",
        f"Q3: {int(contagem_quartis_imc['Q3'])} registos",
        f"Q4: {int(contagem_quartis_imc['Q4'])} registos",
        f"Registos sem quartil atribuído: {registos_sem_quartil_imc}",
        "",
        "QUALIDADE E VALIDAÇÃO",
        f"Registos com pelo menos uma anotação de qualidade: {anotados} ({anotados/n*100:.2f}%)",
        f"Registos com validação humana: {validados} ({validados/n*100:.2f}%)",
        "A ausência de anotação não significa que o ECG esteja livre de problemas.",
        "Uma diferença entre grupos não demonstra causalidade, esta análise é observacional.",
        "",
        "DUPLICADOS E PRESERVAÇÃO",
        "Não foram removidas linhas. Um doente com vários registos não representa, por si só, duplicação.",
        "Duplicados completos, IDs de registo repetidos e doentes repetidos são situações distintas.",
        "",
        "NUMPY E EFICIÊNCIA",
        f"Array antropométrico: shape={diagnostico_numpy['shape']}, ndim={diagnostico_numpy['ndim']}, dtype={diagnostico_numpy['dtype']}, valores finitos={diagnostico_numpy['valores_finitos']}",
        "shape indica linhas e colunas; ndim indica o número de dimensões; dtype indica o tipo dos elementos.",
        "O cálculo do IMC, as máscaras e as agregações utilizam operações vetorizadas.",
        "",
        "VALIDAÇÃO FINAL",
        f"Valores em falta em idade_analitica: {int(preparado['idade_analitica'].isna().sum())}",
        f"Valores em falta em imc: {int(preparado['imc'].isna().sum())}",
        f"Valores em falta em tem_anotacao_qualidade: {int(preparado['tem_anotacao_qualidade'].isna().sum())}",
        "",
        "FICHEIROS PRODUZIDOS",
        "auditoria_missing.csv → Valores em falta por variável",
        "dados_ecg_preparados.csv → Dados originais e variáveis analíticas",
        "resumo_qualidade.txt → Relatório desta auditoria",
    ]

    (OUTPUT_DIR / "resumo_qualidade.txt").write_text(
        "\n".join(linhas) + "\n",
        encoding="utf-8",
    )




def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    original = validar_estrutura(carregar_dados())
    missing = tabela_missing(original)
    preparado = criar_variaveis(original)
    analisar_e_exportar(original, preparado, missing)
    print("Pipeline concluído com sucesso.")
    print(f"Registos processados: {len(original)}")
    print(f"Doentes únicos: {original['doente_id'].nunique()}")
    print(f"IMC calculável: {preparado['imc'].notna().sum()}")
    print(f"Resultados guardados em: {OUTPUT_DIR}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError, AssertionError) as erro:
        print(f"ERRO: {erro}", file=sys.stderr)
        sys.exit(1)
