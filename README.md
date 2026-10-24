# Caso #1 — Auditoria e Preparação de Dados Clínicos de ECG

## Objetivo
Implementar um pipeline reprodutível de auditoria e preparação dos metadados de registos de ECG. O projeto não faz diagnóstico a partir do sinal de ECG nem treina modelos de aprendizagem automática.

## Requisitos
- Python 3.10 ou superior
- pandas
- NumPy

Instalar dependências, se necessário:

```bash
python -m pip install -r requirements.txt
```

## Como executar
A partir desta pasta, executar:

```bash
python main.py
```

O programa lê `dados_ecg.csv`, valida as colunas necessárias e cria automaticamente os ficheiros em `outputs/`. Não necessita de Internet nem de caminhos absolutos fixos. Se faltar o CSV ou alguma coluna obrigatória, apresenta um erro explícito e termina com código de saída diferente de zero.

## Estrutura
```text
Grupo_04
├── main.py
├── README.md
├── requirements.txt
├── data/
│   └── dados_ecg.csv
└── outputs/
    ├── dados_ecg_preparados.csv
    ├── auditoria_missing.csv
    ├── resumo_qualidade.txt
```
## Decisões metodológicas principais
1. **Unidade de análise:** cada linha é um registo de ECG. Um mesmo `doente_id` pode ter vários registos.
2. **Duplicados:** são contabilizadas separadamente linhas totalmente duplicadas, IDs de registo repetidos e registos do mesmo doente. Não são eliminadas linhas automaticamente.
3. **Idade:** os valores entre 0 e 120 são considerados quantitativamente interpretáveis para esta análise descritiva. O valor 300 observado no ficheiro é preservado em `idade`, mas convertido em valor em falta em `idade_analitica`; o ficheiro não permite descobrir a idade verdadeira.
4. **Grupos etários:** `[0,18)`, `[18,35)`, `[35,50)`, `[50,65)` e `[65,121)`, com rótulos descritivos. Não são categorias diagnósticas.
5. **Antropometria:** para calcular IMC, o código exige altura entre 50 e 220 cm e peso entre 1 e 350 kg. São critérios operacionais conservadores e explícitos, não limites clínicos universais. Valores originais são preservados e nenhuma linha é removida.
6. **IMC:** calculado por operação vetorizada `peso_kg / (altura_cm / 100)**2`; `imc_quartil` é construído por `pandas.qcut` nos IMC calculáveis.
7. **Anotações de qualidade:** `tem_anotacao_qualidade=True` quando pelo menos um dos quatro campos de anotação não está vazio nem em falta. `False` significa apenas que não foi encontrada anotação nesses campos.
8. **Causalidade:** comparações entre anotação e validação são descritivas; não demonstram que uma variável cause a outra.
9. **NumPy:** o programa cria um array NumPy com altura e peso e relata `shape`, `ndim`, `dtype` e o número de valores finitos.
10. **Eficiência:** utiliza máscaras booleanas, operações vetorizadas e `groupby`, evitando ciclos Python por cada linha para os cálculos principais.
## Ficheiros de saída
- `auditoria_missing.csv`: número e percentagem de valores em falta por variável.
- `dados_ecg_preparados.csv`: dados originais e variáveis derivadas, sem eliminar registos.
- `resumo_qualidade.txt`: indicadores calculados automaticamente e decisões de preparação.
