# Pré-registro da rodada confirmatória


Fixado em 26/09/2026, antes do download dos dados de 2026.


## 1. Pergunta


Uma regra que libera, na admissão, internações de baixo risco de óbito hospitalar com garantias conformais mantém
essas garantias sob deriva temporal? Esta rodada testa a regra, congelada com dados até 2025, nas internações de
2026, que ainda não foram vistas.


## 2. O que já foi visto antes deste registro


- Uma primeira rodada exploratória, por competência, com os dados de 2019 a 2025.
- Uma revisão crítica do desenho, em 24/09/2026, com reproduções e análises exploratórias de 2019 a 2025, incluindo
  uma demonstração da deriva por ano.
- Testes de fumaça do X2 e do X6 em 2019-2025, em 26/09/2026, durante a implementação: G1 violada no 1º trimestre de
  2024 (7,5% dos óbitos liberados, com 23 dos 30 óbitos por dengue liberados). Com o teste binomial usado até a
  revisão final do código, H1 e H1b saíam refutadas também pelo teste agregado com calibração estática (FDL de 5,5%
  e, sem surtos, 5,4%). A revisão final mostrou que esse teste ignora a variação da calibração e rejeita demais sem
  deriva; com o teste beta-binomial adotado (seção 6), o 1º trimestre de 2024 continua significativo após Holm
  (p = 0,0009; 0,011 após o ajuste de Holm), o agregado deixa de ser (p = 0,14; sem surtos, 0,20) e H1b não é
  refutada no X2. H1b foi formulada depois do primeiro resultado, e a troca de teste foi feita depois de ver os dois.
- Na revisão final, com as internações de 2019 a 2024: 0,08% dos óbitos foram cobrados mais de 4 meses depois do mês
  da internação, e nenhum depois de 6 meses; 2.454 internações (0,3%) têm COBRANCA de 61 a 67 fora do capítulo XV,
  nenhuma com óbito.
- Nenhuma internação com data a partir de 01/01/2026 foi baixada, lida ou analisada.


## 3. Dados


- SIH-SUS, arquivos RD de MG, FTP do DATASUS; hospitais dos 89 municípios da mesorregião Norte de Minas (IBGE 3102),
  filtrados pelo município do estabelecimento (MUNIC_MOV).
- Coorte por data de internação (DT_INTER), a partir de 01/01/2019 (`config.COHORT_START`).
- Desfecho: óbito durante a AIH (MORTE = 1).
- Exclusões: internações com data anterior a 01/01/2019 (removidas por `dataset.load_open`) e linhas sem data de
  internação válida (descartadas por `experimentos.exploratorios.prepare`, que registra quantas). Subpopulações:
  adultos não obstétricos (idade ≥ 18, fora do capítulo XV e da COBRANCA de 61 a 67), para H4; faixas etárias, para
  H5. O marcador obstétrico usa a COBRANCA, que é informação da saída; a regra sem modelo e o subgrupo da H4 o usam
  como definido (tamanho na seção 2).
- Seguimento: X7a e X7b usam as competências até jul/2026; X7c usa as mesmas linhas de X7a no treino e na calibração
  e as competências até out/2026 no teste. Linhas com competência posterior ao corte ficam fora, e a rodada recusa
  dados com algum mês faltando entre jan/2019 e o corte.
- O ETL de 2026 só roda depois deste registro:
  `python -m triagem_baixo_risco.dataset --uf MG --years 2019 2026 --by mov --last-month 2026-07`. A consolidação
  grava as internações com data a partir de 01/01/2026 (`config.SEAL_DATE`) em
  `data/processed/sih_norte_minas_selado.parquet`. Só `experimentos.confirmatorio` lê as variáveis desse arquivo;
  antes do X7, as checagens do ETL leem só a coluna DT_INTER, e o ETL não mostra óbitos das competências de 2026. Os
  exploratórios usam `load_open`, que recusa linhas lacradas.
- A rodada confere N_AIH único, IDENT = 1 e todas as linhas lacradas com data a partir de 01/01/2026.
- Seguimento do teste: antes de rodar, a confirmatória confere o volume de internações de cada mês do teste contra o
  mesmo mês do ano anterior; com alguma razão abaixo de 0,90, a rodada recusa por seguimento incompleto.


## 4. Modelo, calibração e regra (congelados)


- Variáveis: `modeling.train.NUMERIC` (idade, urgência, mesmo município) e `modeling.train.CATEGORICAL` (sexo,
  caráter, especialidade, capítulo CID, CID de 3 caracteres, grupo e subgrupo do procedimento solicitado, município de
  residência, município de atendimento, hospital, mês e dia da semana da internação). Raça/cor e COMPLEX ficam fora.
- Modelo: `modeling.train.make_hgb` (HistGradientBoosting; taxa 0,06; até 400 iterações; 31 folhas; L2 = 1,0;
  parada antecipada com 10%; semente 42; categorias nativas com até 250 níveis).
- Calibração: Platt na parte A e regra na parte B, numa divisão aleatória 50/50 do período de calibração com semente
  42 (`modeling.train.fit_window`).
- Regra de liberação (`modeling.predict`): libera se 1 − p > q₁(α), com q₁ o quantil conformal por classe dos
  escores 1 − p dos óbitos da parte B. α = 0,05 (principal) e 0,10 (secundário). r̂ = limite superior unilateral de
  95% (Clopper-Pearson) da taxa de óbito entre as liberadas na parte B.
- Mondrian cruzado por faixa etária (0-17, 18-39, 40-59, 60-79, 80+) para H5.
- Linhas de base: `make_logistic_full`, `make_logistic_simple` e `rule_based_release` (libera se obstétrica, ou
  eletiva com menos de 60 anos, ou menos de 40 anos).
- Versões das bibliotecas: `requirements.txt`; cada rodada registra as versões usadas na própria saída.


## 5. Rodadas confirmatórias


| Variante | Treino (internações) | Calibração | Teste | Hipóteses | Quando |
|---|---|---|---|---|---|
| X7a (principal) | 01/01/2019 a 31/12/2024 | 01/01/2025 a 31/12/2025 | 01/01/2026 a 31/03/2026 | H1, H1b, H2, H4; H5 descritiva | depois do ETL com competências até jul/2026 |
| X7b | 01/01/2019 a 30/09/2024 | 01/10/2024 a 30/09/2025 | 01/01/2026 a 31/03/2026 | H1, H1b, H2, H4 | idem |
| X7c | como X7a | como X7a | 01/04/2026 a 30/06/2026 | H1, H1b, H2, H4, H5 | depois do ETL com competências até out/2026 |


Comando: `python -m triagem_baixo_risco.experimentos.confirmatorio --variante a` (e `b`, `c`). Cada variante roda uma
única vez, a partir do código deste registro. X7c refaz o ajuste de X7a sobre as mesmas linhas e só prossegue se a
regra for idêntica à registrada em X7a (q₁, r̂ e n₁ da regra global; q₁ e n₁ de cada faixa; nos dois níveis α).
Reexecução só com `--desvio "motivo"`, que grava um arquivo novo (`x7a_desvio_<data>.json`), sem apagar o primeiro,
e registra o motivo em `reports/confirmatorio/desvios.md`; correções de defeito feitas depois deste registro também
são declaradas lá.


Sensibilidades sobre X7a, cada uma uma única vez, relatadas como análises de sensibilidade, sem valor confirmatório
próprio: S1, isotônica no lugar de Platt; S2, só o capítulo da CID-10; S3, sem as AIH com desfecho censurado; S4,
coorte por competência; S5, desfecho por episódio, ligando AIH de continuação (COBRANCA de 21 a 29 seguida de nova
AIH no mesmo hospital em até 1 dia) e transferências (COBRANCA 31 seguida de AIH em outro hospital da região em até
1 dia) pela chave data de nascimento, sexo, CEP e município de residência, com a taxa de ligação relatada; S6, sem
hospital e sem município de atendimento. Comando: `--variante a --sensibilidade s1` (e `s2` a `s6`).


## 6. Hipóteses e critérios


A definição operativa é o código de `triagem_baixo_risco/avaliacao.py` e
`triagem_baixo_risco/experimentos/confirmatorio.py`.


- **H1.** G1 vale: refutada se o número de óbitos liberados no teste for significativamente maior que o esperado sob
  permutabilidade entre os óbitos da parte B e os do teste (teste beta-binomial unilateral a 5%: com n₁ óbitos na
  parte B e k = ⌈(n₁ + 1)(1 − α)⌉, a chance de um óbito do teste ser liberado segue Beta(n₁ + 1 − k, k);
  `modeling.predict.g1_pvalue`).
- **H1b.** G1 vale sem os óbitos de diagnósticos em surto (CID de 3 caracteres com pelo menos 10 óbitos no teste e
  pelo menos 3 vezes o esperado pela participação entre os óbitos dos 12 meses que terminam 3 meses antes do início
  do teste): mesmo critério de H1.
- **H2.** Previsão condicional à razão entre a prevalência do teste e a da parte B da calibração: com razão ≤ 1,0,
  G2 vale (refutada se a taxa de óbito entre as liberadas exceder r̂ com significância); com razão ≥ 1,2, G2 falha
  (refutada se não houver violação significativa); entre 1,0 e 1,2, sem previsão.
- **H4.** Com α = 0,05: fração liberada de pelo menos 50% das internações e 40% das de adultos não obstétricos; IC
  95% (bootstrap de hospitais, 1.000 reamostragens, semente 42; o bootstrap de internações é relatado ao lado) da
  diferença de fração liberada inteiramente acima de zero contra a regra sem modelo (com o limiar do modelo que
  reproduz, na parte B, a FDL da regra) e contra a logística de 4 variáveis.
- **H5.** Com o Mondrian cruzado, G1 vale em cada faixa etária com pelo menos 100 óbitos no teste: refutada se o
  teste beta-binomial de alguma faixa (com os n₁ óbitos da faixa na parte B) rejeitar após Holm entre as faixas;
  faixas menores são descritivas. H5 é confirmatória só em X7c; em X7a, o campo `h5` da saída é descritivo; em X7b,
  H5 não é avaliada e esse campo fica fora das conclusões.


Os resultados com α = 0,10 são secundários. Métricas por período, intervalos e correções como implementadas em
`triagem_baixo_risco/avaliacao.py`.


## 7. Análises exploratórias (2019-2025)


X1 a X6, X8, sensibilidades S1 a S6 e desfecho secundário, declarados como exploratórios. Parâmetros: atraso de 3
meses; janela de 12 meses; mínimo de 50 óbitos por janela (senão reaproveita a última calibração); γ = 0,02
(sensibilidade 0,01 e 0,05); meia-vida de 4 trimestres (sensibilidade 2 e 8); na estratégia (g), limiar da
estimativa de prevalência igual à prevalência da parte B. Critérios de H1, H1b, H2, H3 e H5 nessas análises:
`triagem_baixo_risco/avaliacao.py`, aplicados por `experimentos.exploratorios.exploratory_hypotheses`. No teste
agregado de H2 (períodos com razão ≤ 1,0), a taxa de óbito entre as liberadas, somada nesses períodos, é comparada
com a média de r̂ ponderada pelo número de liberadas em cada período, porque r̂ muda entre períodos nas estratégias
com atualização. Nos testes agregados de H1 e H1b, com uma calibração só (estratégia a) a soma dos períodos segue a
beta-binomial exata; com calibrações diferentes, o p-valor supõe dependência máxima entre elas
(`modeling.predict.g1_aggregate_pvalue`). O X1 cobre 2019-2025; de 2026, só o que as saídas do X7 registram
(internações, óbitos, prevalência do teste e contagens por competência). No X8, o limiar de Ville (log10 M ≥ 2) não
garante 1% de alarmes falsos, porque os p-valores conformais compartilham o conjunto de calibração; os alarmes são
descritivos.


## 8. Desvios


Qualquer mudança depois deste registro, inclusive correção de defeito, é registrada em
`reports/confirmatorio/desvios.md`, com data, motivo e efeito esperado, e relatada no TCC.
