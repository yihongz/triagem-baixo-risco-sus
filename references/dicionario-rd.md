# Colunas do RD usadas no código

O SIH-SUS publica um arquivo RD, a AIH reduzida, por UF e competência: `RDMG2401.dbc` é o de Minas Gerais de janeiro de
2024. O ETL guarda todas as colunas das linhas dos 89 municípios, mas o código só usa as desta tabela. As marcadas como
bloqueadas estão na lista de colunas que vazam o desfecho e nunca entram no modelo, mesmo quando servem a outra coisa.

| Coluna | Descrição | Uso no código |
|---|---|---|
| N_AIH | número da AIH | ETL; a rodada confirmatória confere que nenhum número se repete nas duas bases juntas |
| IDENT | tipo da AIH | a rodada confirmatória exige 1 em todas as linhas |
| ANO_CMPT | ano da competência; a competência é o mês de processamento da AIH | ETL; corte de competência do X7; datas da S4 |
| MES_CMPT | mês da competência | ETL; corte de competência do X7; datas da S4 |
| DT_INTER | data de internação (AAAAMMDD) | ETL, que separa as internações de 2026; coorte, trimestres e janelas; modelo (mês e dia da semana); ligação da S5 |
| DT_SAIDA | data de saída | ligação da S5; bloqueada |
| IDADE | idade, na unidade de COD_IDADE | modelo; marcadores de adulto e faixa etária; regra sem modelo |
| COD_IDADE | unidade da idade: 2 dias, 3 meses, 4 anos, 5 acima de 100 anos | junto com IDADE |
| SEXO | sexo: 1 masculino, 3 feminino | modelo; sexo nos resultados do X5; chave da S5 |
| NASC | data de nascimento | chave da S5 |
| CEP | CEP de residência | chave da S5 |
| MUNIC_RES | município de residência, código IBGE de 6 dígitos | modelo; chave da S5 |
| MUNIC_MOV | município do hospital, código IBGE de 6 dígitos | ETL, que filtra os 89 municípios; modelo; marcador de polo |
| CNES | código do hospital no Cadastro Nacional de Estabelecimentos de Saúde | modelo; bootstrap de hospitais; ligação da S5 |
| CAR_INT | caráter da internação: 01 eletivo, 02 urgência, 03 a 06 acidentes e lesões | modelo (caráter e urgência); marcador de urgência |
| ESPEC | especialidade do leito | modelo |
| DIAG_PRINC | diagnóstico principal, CID-10 | modelo (capítulo e CID de 3 caracteres); marcador obstétrico; surtos de H1b |
| PROC_SOLIC | procedimento solicitado, código SIGTAP de 10 dígitos | modelo (grupo e subgrupo) |
| COBRANCA | motivo de saída ou de permanência | marcadores de censurada e obstétrica; ligação da S5; bloqueada |
| MORTE | óbito durante a AIH: 1 sim, 0 não | desfecho principal; ETL, que conta os óbitos de cada competência e esconde os de 2026; bloqueada |
| DIAS_PERM | dias de permanência | desfecho secundário, 8 dias ou mais; bloqueada |
| RACA_COR | raça/cor; 99 é "sem informação" | só no X1, que acompanha a fração de "sem informação" |
| COMPLEX | complexidade da AIH: 02 média, 03 alta | bloqueada |
| QT_DIARIAS, DIAR_ACOM | diárias da AIH e diárias de acompanhante | bloqueadas |
| UTI_MES_TO, UTI_INT_TO | dias de UTI no mês e diárias em unidade intermediária | bloqueadas |
| MARCA_UTI, MARCA_UCI | tipo de UTI e tipo de UCI usados | bloqueadas |
| VAL_SH, VAL_SP, VAL_TOT, VAL_UTI, VAL_UCI, US_TOT | valores de serviços hospitalares, de serviços profissionais, total, de UTI, de UCI e total em dólar | bloqueadas |
| VAL_SH_FED, VAL_SP_FED, VAL_SH_GES, VAL_SP_GES | complementos federal e do gestor aos valores de serviços hospitalares e profissionais | bloqueadas |
| TOT_PT_SP | pontos de serviços profissionais, zerado de 2019 a 2025 | bloqueada |
| PROC_REA | procedimento realizado | bloqueada |
| CID_MORTE | CID da causa do óbito, zerado de 2019 a 2025 | bloqueada |
| DIAG_SECUN | diagnóstico secundário do formato antigo, zerado de 2019 a 2025 | bloqueada |
| DIAGSEC1 a DIAGSEC9 | diagnósticos secundários | bloqueadas |
| TPDISEC1 a TPDISEC9 | tipo de cada diagnóstico secundário | bloqueadas |
| CID_ASSO | CID associado, zerado de 2019 a 2025 | bloqueada |
| INFEHOSP | infecção hospitalar, vazia de 2019 a 2025 | bloqueada |
| IND_VDRL | indica se houve exame VDRL | bloqueada |

O ETL recusa o arquivo de um mês se faltar alguma destas 17 colunas: N_AIH, ANO_CMPT, MES_CMPT, DT_INTER, IDADE,
COD_IDADE, SEXO, CAR_INT, ESPEC, DIAG_PRINC, PROC_SOLIC, MUNIC_RES, MUNIC_MOV, CNES, COBRANCA, MORTE e DIAS_PERM.
