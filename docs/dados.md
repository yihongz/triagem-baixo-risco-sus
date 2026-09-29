# Dados

O modelo só usa o que se sabe na hora da admissão. As variáveis saem das colunas do arquivo RD do SIH-SUS; o
significado de cada coluna e o lugar dela no código estão em `references/dicionario-rd.md`.

## Variáveis do modelo

| Variável | O que é | Coluna do RD |
|---|---|---|
| `idade_anos` | idade em anos; o SIH registra a de bebês em dias ou meses, e a de quem passou dos 100 anos com um código à parte | IDADE e COD_IDADE |
| `urgencia` | 1 quando a internação não é eletiva | CAR_INT |
| `mesmo_municipio` | 1 quando o paciente mora no município do hospital | MUNIC_RES e MUNIC_MOV |
| `sexo` | sexo do paciente | SEXO |
| `carater` | caráter da internação: eletiva, urgência ou um dos tipos de acidente e lesão | CAR_INT |
| `especialidade` | especialidade do leito | ESPEC |
| `cid_capitulo` | capítulo da CID-10 do diagnóstico principal | DIAG_PRINC |
| `cid3` | as três primeiras posições do diagnóstico principal | DIAG_PRINC |
| `proc_grupo` | grupo do procedimento solicitado, os 2 primeiros dígitos do código | PROC_SOLIC |
| `proc_subgrupo` | subgrupo do procedimento solicitado, os 4 primeiros dígitos | PROC_SOLIC |
| `munic_res` | município de residência | MUNIC_RES |
| `munic_mov` | município do hospital | MUNIC_MOV |
| `cnes` | o hospital | CNES |
| `mes_internacao` | mês da internação | DT_INTER |
| `dia_semana_internacao` | dia da semana da internação | DT_INTER |

As três primeiras entram como números, e as outras doze, como categorias, inclusive o mês e o dia da semana. No
boosting, cada categórica pode ter até 250 níveis, e os mais raros ficam juntos num nível só. Isso pesa no CID de 3
caracteres e no município de residência, que passam de mil valores cada na base de 2019 a 2025. Nas duas logísticas,
as categóricas entram em one-hot, e os níveis com menos de 20 internações no treino também ficam juntos; a idade entra
por spline cúbica, com 5 nós. A logística completa usa todas as variáveis, e a de 4 variáveis fica com idade, sexo,
urgência e capítulo da CID.

## O que ficou de fora

Tirei raça/cor porque a codificação mudou no meio do período. A categoria "sem informação" (código 99) ficou entre 28%
e 33% das internações em cada ano de 2019 a 2021, caiu ao longo de 2022 e sumiu a partir do 2º trimestre de 2023. O X1
ainda acompanha essa fração por trimestre, como uma das medidas de deriva.

COMPLEX, a complexidade da AIH, saiu porque é determinada pelo procedimento realizado, que só se conhece depois da
admissão: em 99,6% das internações de 2019 a 2025, a COMPLEX da AIH é a mais frequente entre as AIH com o mesmo
procedimento realizado (com o procedimento solicitado, seriam 97,6%; conta reproduzível em `reports/validacao_etl.json`). Por isso ela está na lista de colunas que vazam o desfecho, junto com as que só existem
depois da admissão ou são o próprio desfecho: permanência e diárias, UTI e UCI, valores pagos, data e motivo de saída
(DT_SAIDA e COBRANCA), óbito e CID do óbito, procedimento realizado, diagnósticos secundários, infecção hospitalar e
exame VDRL. A lista, coluna por coluna, está no dicionário. Um teste impede que alguma dessas colunas, ou RACA_COR,
vire coluna de origem das variáveis, e que COMPLEX saia da lista.

Algumas dessas colunas voltam, fora do modelo. MORTE define o desfecho principal, e DIAS_PERM, o secundário
(permanência de 8 dias ou mais). A COBRANCA entra nos marcadores e, com a DT_SAIDA, na ligação de episódios da S5.

## Marcadores

Os marcadores servem para relatar resultados por subgrupo e para definir as subpopulações das hipóteses. Nenhum entra
no modelo.

| Marcador | Definição | Onde entra |
|---|---|---|
| censurada | COBRANCA de 21 a 29 (permanência), 31 ou 32 (transferência) ou 51 (encerramento administrativo): a AIH termina sem alta e sem óbito | a S3 tira essas AIH; o X1 acompanha a fração delas por trimestre |
| obstétrica | diagnóstico principal no capítulo XV da CID-10, ou COBRANCA de 61 a 67 | regra sem modelo; subgrupo de adultos não obstétricos de H4 |
| adulto | 18 anos ou mais | subgrupo de H4 |
| urgência | CAR_INT diferente de 01, qualquer caráter que não o eletivo | regra sem modelo, que libera a eletiva de quem tem menos de 60 anos |
| faixa etária | 0-17, 18-39, 40-59, 60-79 e 80 anos ou mais | Mondrian cruzado de H5; X5 |
| polo | hospital em Montes Claros (MUNIC_MOV 314330) | X5 |

Junto com os marcadores, o código guarda a idade em anos, que a regra sem modelo usa, e o hospital (CNES), que define
os grupos do bootstrap de hospitais. O sexo (1 masculino, 3 feminino) fica ali para o X5.

A COBRANCA só se conhece na saída, mas entra no marcador obstétrico. Das internações de 2019 a 2025, 2.483 (0,3%) entram
no marcador só pela COBRANCA de 61 a 67, e nenhuma delas terminou em óbito.

## Ligação de AIH na S5

Na S5, o desfecho é o óbito no episódio, que pode juntar mais de uma AIH. Uma AIH de permanência (COBRANCA de 21 a 29) é
ligada à seguinte do mesmo paciente no mesmo hospital, e uma de transferência (COBRANCA 31), à seguinte em outro
hospital da região, desde que a seguinte comece até 1 dia depois da saída. A COBRANCA 32, transferência para internação
domiciliar, fica fora da ligação. O SIH não tem identificador de paciente, então a chave é aproximada: data de
nascimento, sexo, CEP e município de residência. Portela et al. (1997) já juntavam as AIH de uma mesma internação
usando, entre outros campos, o hospital, a data de nascimento e o sexo. Fica na análise só a primeira AIH de cada
episódio, com o óbito de qualquer AIH dele, e a taxa de ligação vai junto com os resultados.
