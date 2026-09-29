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
visto o resultado, e a S3, a S4 e a S5 já tinham rodado — cada uma levou cerca de dois minutos —, sem que eu tivesse lido os resultados delas. A S2 e a S6 rodaram uma única vez cada, com o código
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


- testes automatizados: um que fixa as variáveis da logística completa fora da S2 e da S6 (27/09/2026);
- os exploratórios, refeitos depois do download de 2026, que completou o seguimento das internações do fim de 2025
  (29/09/2026): no agregado da estratégia estática, o p da G1 passou de 0,135 para 0,157, e sem os óbitos em surto, de
  0,204 para 0,231, sem mudar nenhuma decisão sobre as hipóteses. O pré-registro cita os valores anteriores com duas
  casas, 0,14 e 0,20; o valor exato de antes do pré-registro (0,135) é do registro de execuções do autor, não versionado.

- a reexecução de validação em pasta limpa (29/09) cobriu as bases e os exploratórios: a base lacrada manteve o SHA-256 dos JSONs; a aberta canônica (`fd5fec25…`, o `sha256_dados.aberto` dos JSONs da rodada) foi rebaixada e saiu como `76f6bcc5…` — mesmas 825.434 linhas, mesmos resultados exploratórios; a diferença é a formatação de campos de valores pagos de fev/2019 e jan/2024, que não entram nas análises. As rodadas confirmatórias não foram reexecutadas: cada arquivo saiu uma vez, entre a noite de 26/09 e a de 27/09. Limitação: os relatórios das explorações não gravam o SHA-256 da base — só os da rodada; os números exploratórios são conferíveis pelas linhas e campos, não pelo hash. Nada muda nas decisões; entra aqui porque o pré-registro pede que reexecuções fiquem no diário.


- os arquivos publicados da rodada não registram o commit do código que os gerou; o que ancora a versão é a tag `pre-registro-v1` (26/09, 22:19) para as execuções pré-registradas, e o commit da correção de 27/09 para S2 e S6.


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
- A H1b foi formulada depois de ver a violação da G1 no 1º trimestre de 2024.

---
