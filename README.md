# Triagem de baixo risco de óbito hospitalar sob deriva temporal

Um modelo de risco treinado com dados do SIH-SUS só entra em uso meses depois das internações que o treinaram, e nesse
meio-tempo a mortalidade e o perfil de quem se interna mudam, como mudaram na pandemia de 2020 e 2021. Uma regra que
usa esse modelo para dispensar de atenção adicional as internações de baixo risco só serve se a garantia de segurança
dela continuar valendo depois dessas mudanças. Neste TCC eu testo se isso acontece com uma regra conformal. Os dados
são as AIH (autorizações de internação hospitalar) de 2019 a 2026 dos hospitais dos 89 municípios da mesorregião Norte
de Minas, e o modelo só usa variáveis de admissão.

A regra parte da probabilidade de óbito calibrada de um modelo de boosting e libera as internações de baixo risco. A
garantia G1 põe um teto de α na chance de um óbito sair liberado, e a FDL é a fração dos óbitos que a regra de fato
libera. Já a G2 trata da taxa de óbito entre as liberadas, que não deveria passar de r̂, o limite superior dessa taxa na
parte B, a metade da calibração que ajusta a regra; como essa taxa depende da prevalência, tende a subir junto com a
mortalidade. Fixei as hipóteses, os critérios que as refutariam e a regra antes de baixar os dados de 2026, no
[pré-registro](docs/pre-registro.md); os critérios estão em `triagem_baixo_risco/avaliacao.py` e
`triagem_baixo_risco/experimentos/confirmatorio.py`, e as mudanças feitas depois do pré-registro, em
[`reports/confirmatorio/desvios.md`](reports/confirmatorio/desvios.md).

Alguns termos mudam de nome entre este README, os notebooks, os arquivos de resultado e o artigo:

| README e notebooks | Artigo | Chave nos JSON |
|---|---|---|
| FDL | fração dos óbitos liberados | `fdl` |
| TOL | taxa de óbito entre as liberadas | `tol` |
| razão de prevalência | razão de mortalidade (teste sobre parte B) | `razao_prevalencia` |
| Mondrian cruzado | regra por faixa etária | `cruzado` |
| logística de 4 variáveis | logística com 4 variáveis | `logistica_simples` |

## Resultados de 2026

A rodada principal, X7a, treinou o modelo com as internações de 2019 a 2024, calibrou a regra com as de 2025 e a aplicou
às 36.621 internações do primeiro trimestre de 2026. Com α = 0,05, a regra liberou 23.769 delas, e 59 desses pacientes
morreram na internação. Isso dá uma FDL de 59/1.345 = 4,39%, com p = 0,81 no teste de violação de G1. Mesmo com 1.345
óbitos no teste, o intervalo de 95% da FDL vai de 3,4% a 5,6%, e uma FDL real um pouco acima de 5% não fica descartada.
Entre as liberadas, a taxa de óbito foi de 0,25%, abaixo do r̂ de 0,35%. Era o que H2 previa: G2 deve valer quando a
razão de prevalência de óbito, a do teste sobre a da parte B, fica em até 1,0, e no X7a ela foi de 0,92.
Com razão de 1,2 ou mais, H2 prevê que G2 falhe; entre os dois valores, não prevê nada.

No total, a regra liberou 64,9% das internações do teste e 54,0% das de adultos não obstétricos, acima dos mínimos de
50% e 40% da hipótese de utilidade, H4. O boosting liberou 10,1 pontos percentuais a mais que uma logística de 4
variáveis (idade, sexo, urgência e capítulo da CID), cada um com a sua regra conformal. Contra a regra sem modelo, que
libera por idade, caráter eletivo e obstetrícia, a comparação é feita na mesma FDL, com o limiar do boosting que
reproduz, na parte B, a dela; nessa conta, a vantagem foi de 17,8 pontos. H4 também pede que o IC de 95% das duas
diferenças fique acima de zero, e ficou. Nenhuma hipótese foi refutada no X7a.

O X7b recua treino e calibração em três meses, com a calibração de outubro de 2024 a setembro de 2025. Com α = 0,05, ele
liberou 65,6% das internações do teste, com FDL de 63/1.345 = 4,68%, e, como o X7a, não refutou nenhuma hipótese, nem
com esse α nem com 0,10. As seis sensibilidades do X7a também não refutaram nenhuma com α = 0,05. Com α = 0,10, a S2 (o
modelo sem o CID de 3 caracteres) refutou H4: a vantagem sobre a logística de 4 variáveis caiu para +3,0 pontos, com IC
de -0,3 a 7,3 pontos. Já a S4 (coorte pela competência, o mês de processamento da AIH) deixou a razão de prevalência em
1,0004, logo acima de 1,0, e H2 sem previsão. As tabelas de todas as rodadas estão no notebook
[6.0](notebooks/6.0-pa-rodada-confirmatoria.ipynb). O X7c, com as internações de abril a junho de 2026, depende das
competências até outubro de 2026 e ainda não rodou.

As decisões sobre as hipóteses vêm das execuções de 26 e 27/09/2026. Os arquivos de `reports/confirmatorio/` foram
gerados por essas execuções; o campo `executado_em` deles marca as datas delas (26 e 27/09/2026). O registro está em
[`desvios.md`](reports/confirmatorio/desvios.md).

## Instalação

É preciso ter o [uv](https://github.com/astral-sh/uv) e o `make`. Na raiz do repositório:

```bash
make ambiente
```

Isso cria o `.venv` com Python 3.12 e instala as versões exatas de `requirements.txt`, as mesmas que geraram os
resultados; as ferramentas de desenvolvimento, do extra `dev` do `pyproject.toml`, entram depois sem trocar nenhuma
delas. Não é preciso ativar o ambiente, porque os outros alvos chamam `.venv/bin/python` direto.

## Como reproduzir

Os dados não estão no repositório; `make dados` baixa as AIH do FTP do DATASUS.

```bash
make dados            # competências de jan/2019 a jul/2026
make exploratorios    # cerca de 25 minutos
make confirmatorio    # cerca de 15 minutos
make figuras
make notebooks
```

Os resultados publicados já estão em `reports/`. Como a rodada confirmatória não sobrescreve resultado, ela
para no X7a; para refazê-la, apague `reports/confirmatorio/` antes de `make confirmatorio`, depois de guardar uma cópia
de segurança. O passo a passo, com o que
cada alvo grava, está em [docs/reproduzir.md](docs/reproduzir.md).

## Estrutura das pastas

```text
triagem-baixo-risco-sus/
├── LICENSE                   MIT, para o código
├── LICENSE-textos            CC BY 4.0, para textos, figuras e resultados
├── Makefile                  um alvo para cada passo, de make ambiente a make docs
├── README.md
├── mkdocs.yml                configuração do site da documentação
├── pyproject.toml            pacote, dependências, pytest e ruff
├── requirements.txt          versões exatas das dependências de execução
├── data/                     fora do repositório, menos a lista de municípios
│   ├── external/             municipios_norte_minas.csv, os 89 municípios (IBGE)
│   ├── raw/                  .dbc do DATASUS, apagados depois de lidos
│   ├── interim/              um parquet por competência (MG_AAAAMM.parquet) e etl_log.jsonl
│   └── processed/            sih_norte_minas.parquet (até 2025) e sih_norte_minas_selado.parquet (2026)
├── docs/                     index.md, dados.md, reproduzir.md e pre-registro.md
├── models/                   só um LEIAME: os modelos não são salvos
├── notebooks/                do 1.0 (coorte e deriva) ao 7.0 (simulação dos testes)
├── references/
│   ├── referencias.bib
│   └── dicionario-rd.md      colunas do arquivo RD do SIH usadas no projeto
├── reports/
│   ├── exploratorios/        resultados de X1 a X6 e X8, por desfecho e sensibilidade
│   ├── confirmatorio/        x7a.json, x7b.json, x7a_s1.json a x7a_s6.json e desvios.md
│   └── figures/              figuras dos exploratórios, uma pasta por desfecho
├── tests/                    testes com dados sintéticos
└── triagem_baixo_risco/
    ├── config.py             caminhos e constantes
    ├── dataset.py            ETL, municípios, coorte e separação das internações de 2026
    ├── features.py           variáveis de admissão, estratos e ligação de AIH em episódios
    ├── modeling/
    │   ├── train.py          modelos, calibradores e divisão da calibração em duas partes
    │   ├── predict.py        regra de liberação, garantias e testes estatísticos
    │   └── atualizacao.py    estratégias de atualização (a) a (g)
    ├── avaliacao.py          métricas, avaliação por período e critérios de H1 a H5
    ├── monitoramento.py      gráfico de controle e martingale conformal
    ├── experimentos/
    │   ├── exploratorios.py  X1 a X6 e X8, com as sensibilidades S1 a S6
    │   └── confirmatorio.py  X7a, X7b e X7c
    └── plots.py              figuras a partir dos resultados salvos
```

## Licenças

O código está sob a licença MIT ([LICENSE](LICENSE)), e isso vale também para as células de código dos notebooks. Os
textos do repositório, inclusive o dos notebooks, estão sob CC BY 4.0 ([LICENSE-textos](LICENSE-textos)), junto com as
figuras e os arquivos de resultado de `reports/`.

## Como citar

```text
ALMEIDA, Paulo Victor Soares. Triagem de baixo risco de óbito hospitalar sob deriva temporal. 2026. Disponível em:
https://github.com/yihongz/triagem-baixo-risco-sus.
```

```bibtex
@misc{almeida2026triagem,
  author = {Almeida, Paulo Victor Soares},
  title  = {Triagem de baixo risco de óbito hospitalar sob deriva temporal},
  year   = {2026},
  url    = {https://github.com/yihongz/triagem-baixo-risco-sus}
}
```
