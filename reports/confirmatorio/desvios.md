# Desvios do pré-registro

O pré-registro fixou o código, as hipóteses e os critérios de refutação em 26/09/2026, antes do download dos dados de
2026, e pede que qualquer mudança feita depois disso fique registrada aqui, com data, motivo e efeito esperado.


## Correção da logística completa na S2 e na S6


**26/09/2026.** A S2 da X7a, que tira do modelo a categoria de três caracteres da CID-10, parou antes de calcular
qualquer resultado. A logística completa, uma das linhas de base, montava a matriz sempre com a mesma lista de
variáveis, que inclui essa categoria, e o código não achou a coluna. Nenhum arquivo foi gravado e nenhuma métrica de
2026 foi calculada. A S6, que tira o hospital e o município de atendimento, tinha o mesmo defeito e não foi rodada. A
S3, a S4 e a S5 não mexem nas colunas e rodaram em seguida.


**27/09/2026.** Corrigi o defeito: a logística completa passou a usar as variáveis da própria sensibilidade, como o
boosting já fazia. Quando decidi, eu já conhecia os resultados da X7a e da X7b; a S1 tinha rodado, mas eu não tinha
visto o resultado, e a S3, a S4 e a S5 já tinham rodado (cada uma levou cerca de dois minutos), sem que eu tivesse lido os resultados delas. A S2 e a S6 rodaram uma única vez cada, com o código
corrigido, ainda em 27/09/2026.


**Efeito esperado.** Nenhum sobre as decisões de hipótese. H1 a H5 se decidem pelo boosting, pela regra sem modelo e
pela logística de 4 variáveis, que já seguiam as variáveis de cada sensibilidade; a logística completa aparece só num
campo descritivo. Nas outras rodadas, a lista de variáveis é a de sempre, e a correção não muda nada.


**Resultado.** Com α = 0,05, nem a S2 nem a S6 refutaram hipótese. Com α = 0,10, a S2 refutou a H4, pela diferença
contra a logística de 4 variáveis, em que a correção não mexe.


Arquivos: `reports/confirmatorio/x7a_s2.json` e `reports/confirmatorio/x7a_s6.json`.
- 2026-09-27T20:10:46, X7a s2: correção de defeito: a logística completa passa a usar as variáveis da sensibilidade; execução da S2 com o código corrigido (decisão de 27/09) (arquivo x7a_s2.json)
- 2026-09-27T20:11:42, X7a s6: correção de defeito: a logística completa passa a usar as variáveis da sensibilidade; execução da S6 com o código corrigido (decisão de 27/09) (arquivo x7a_s6.json)


## Mudanças depois do pré-registro que não mexem em decisão


- as figuras dos experimentos, geradas a partir dos resultados salvos (30/09/2026);
- a documentação e os notebooks (30/09/2026);
- o notebook 7.0, que calcula os tamanhos exatos dos testes estatísticos e os confere por simulação (30/09/2026);
- testes automatizados: um que fixa as variáveis da logística completa fora da S2 e da S6 (27/09/2026), os das
  figuras e um que confere que as variáveis do modelo não mudam quando se embaralham todas as colunas que não são de
  admissão (30/09/2026);
- os exploratórios, refeitos depois do download de 2026, que completou o seguimento das internações do fim de 2025
  (29/09/2026): no agregado da estratégia estática, o p da G1 passou de 0,135 para 0,157, e sem os óbitos em surto, de
  0,204 para 0,231, sem mudar nenhuma decisão sobre as hipóteses. O pré-registro cita os valores anteriores com duas
  casas, 0,14 e 0,20; o valor exato de antes do pré-registro (0,135) é do registro de execuções do autor, não versionado.

- a reexecução de validação em pasta limpa (29/09) cobriu as bases e os exploratórios: a base lacrada manteve o SHA-256 dos JSONs; a aberta canônica (`fd5fec25…`, o `sha256_dados.aberto` dos JSONs da rodada) foi rebaixada e saiu como `76f6bcc5…`, com as mesmas 825.434 linhas e os mesmos resultados exploratórios; a diferença é a formatação de campos de valores pagos de fev/2019 e jan/2024, que não entram nas análises. As rodadas confirmatórias não foram reexecutadas: cada arquivo saiu uma vez, entre a noite de 26/09 e a de 27/09. Limitação: os relatórios das explorações não gravam o SHA-256 da base, só os da rodada; os números exploratórios são conferíveis pelas linhas e campos, não pelo hash. Nada muda nas decisões; entra aqui porque o pré-registro pede que reexecuções fiquem no diário.
- as sensibilidades de parâmetro declaradas no pré-registro, γ = 0,01 e 0,05 na estratégia (e) e meia-vida de 2 e 8 trimestres na (f), constam dos relatórios versionados (`x3.csv`, `x4.csv`, figuras de garantias); o que faltava era o relato no texto, e ele entra aqui: em 2024T1, e(γ=0,01) e e(γ=0,05) acompanham a referência (0,020 e 0,026 ajustados por Holm), e a (f) com meia-vida 8 violaria a G1 (6,97% dos óbitos; ajustado 0,040) onde a (f) pré-registrada resiste (0,058), resistência que as brutas de 0,005 a 0,007 não teriam sem Holm. Nenhuma linha de parâmetro altera decisão confirmatória: H3 se avalia com γ = 0,02 e meia-vida 4.


- os arquivos publicados da rodada não registram o commit do código que os gerou; o que ancora a versão é a tag `pre-registro-v1` (26/09, 22:19) para as execuções pré-registradas, e o commit da correção de 27/09 para S2 e S6.


- que a H2 já estava refutada na exploração e foi mantida porque a decisão vale pela rodada de 2026, o pré-registro não declara; declara-se aqui, e é a primeira das duas omissões que a seção 3.5 reconhece.
- a suíte de testes entrou no repositório quinze minutos depois do congelamento, como verificação do código recém-congelado; para a X7c, a rodada passará a gravar o commit da execução no JSON de saída.
## Troca da regra de decisão da H5 (declarada tarde)


**30/09/2026.** O pré-registro e o código versionados decidem a H5 só por Holm. A regra anterior, que refutava também
se qualquer faixa tivesse p binomial abaixo de 0,05 (a mesma que o caderno 7.0, escrito hoje, ainda descreve), foi
trocada durante o desenvolvimento, depois de vistos os resultados exploratórios com as duas regras, sem registro na
época. Efeito: sob a regra antiga, a H5 teria sido refutada no X5: na faixa de 60 a 79 anos em 2023, 135 de 2.200
óbitos liberados (6,1%), p binomial 0,0098; a linha cruzada de `resultados_x5.json` dá p da G1 = 0,078, que Holm não
refutaria de todo modo; os 88 de 2.200 da regra única não são essa linha. O agregado de 2023 a 2025 (3,1% a 5,0%) não é o recorte
que a H5 avaliou.


## Errata do pré-registro


- Na seção 2, as 2.454 internações com COBRANCA de 61 a 67 fora do capítulo XV, nenhuma com óbito, são as de 2019 a
  2025 contadas com as competências até dez/2025, e não as de 2019 a 2024, que são 1.939. Na base atual, com as
  competências até jul/2026, são 2.483 (0,3%), também sem óbito. Nada muda nas análises.


## O que o pré-registro já declarava


- A checagem do seguimento (razão entre as internações de cada mês do teste e as do mesmo mês do ano anterior, com
  recusa abaixo de 0,90) já estava no código do pré-registro; as razões de janeiro, fevereiro e março de 2026, 1,06,
  1,01 e 1,14, foram calculadas fora das rodadas, sem olhar nenhum desfecho.
- A troca do teste binomial pelo beta-binomial na G1 veio antes do pré-registro, depois de eu ver os resultados
  exploratórios com os dois testes.
- Datação da troca acima: o commit de modelagem de 08/09 é a "revisão final do código" do pré-registro; a família binomial de G1 existiu só no ambiente de desenvolvimento e nunca foi versionada (o binomial que sobrevive no módulo serve à G2). A comparação dos dois testes foi feita nesse desenvolvimento, no início de setembro, sobre a coorte do ETL inicial; os testes de fumaça de 26/09 (X2 e X6) que o pré-registro relata reconfirmaram o padrão, e é a essa sequência que o caderno 7.0 chama "depois de ver o X2 com os dois". A "revisão final mostrou" do pré-registro é o efeito visto nas simulações rápidas do desenvolvimento; a quantificação (26,1%) saiu depois, no caderno 7.0.
- A H1b foi formulada depois de ver a violação da G1 no 1º trimestre de 2024.

---


## Acréscimos de 16/10/2026


- Mudança depois do pré-registro que não mexe em decisão e que faltava na lista acima: a conferência da COMPLEX
  (`triagem_baixo_risco/experimentos/validacao_etl.py` e `reports/validacao_etl.json`, commit `2c32cb4`, 29/09/2026),
  que calcula, nas internações de 2019 a 2025 da base aberta, a concordância da COMPLEX com a complexidade mais
  frequente entre as AIH com o mesmo procedimento realizado (99,6%) e com o mesmo procedimento solicitado (97,6%). A
  COMPLEX já estava fora do modelo no pré-registro; a conta só documenta o motivo.
- Limitação de prova: duas datas deste registro não são verificáveis de fora. O horário do download de 2026 vem do
  `etl_log.jsonl`, que fica na máquina do autor, e a tag `pre-registro-v1` não é assinada nem tem testemunha pública.
  A cronologia (commit do pré-registro às 21:40 de 26/09, tag às 22:19, download das 22:54 às 22:56, primeira rodada
  às 23:02) é declaração do autor, não prova. `docs/reproduzir.md` já trazia essa ressalva, e o artigo passa a
  trazê-la na seção 3.5 e nas limitações.
