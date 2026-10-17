# Como reproduzir

Cada passo tem um alvo no `Makefile`, e todos rodam da raiz do repositório:

```bash
make ambiente
make dados
make exploratorios
make confirmatorio
make figuras
make notebooks
```

Os resultados publicados estão em `reports/`. Rodar de novo os exploratórios e as figuras sobrescreve esses
arquivos; para comparar o que mudou, guarde antes uma cópia da pasta.

## Ambiente

`make ambiente` usa o [uv](https://github.com/astral-sh/uv) para criar o `.venv` com Python 3.12. As dependências de
execução vêm de `requirements.txt`, com as versões exatas que geraram os resultados. As de desenvolvimento (pytest,
ruff, mkdocs, jupyter, nbconvert e ipykernel) são instaladas depois, com essas versões como restrição. Os outros alvos
chamam `.venv/bin/python`, então não é preciso ativar o ambiente.

## Dados

Os dados vêm do FTP do DATASUS (`ftp.datasus.gov.br`). `make dados` baixa os arquivos RD de Minas Gerais, um por
competência, de janeiro de 2019 até `ULTIMO_MES`, que por padrão é 2026-07. Se o FTP falha num mês, o ETL tenta pelo
PySUS, e cada mês tem até três tentativas. De cada arquivo ficam só as internações em hospitais dos 89 municípios
listados em `data/external/municipios_norte_minas.csv`. O ETL grava:

- `data/interim/MG_AAAAMM.parquet`, um por competência;
- `data/interim/etl_log.jsonl`, uma linha por competência, com a origem do arquivo (FTP ou PySUS), o SHA-256 do `.dbc`
  e as contagens de linhas (as linhas de 2019 a 2025 são do formato antigo do log, sem origem, SHA-256 e data de
  download; só as de 2026 trazem esses campos);
- `data/processed/sih_norte_minas.parquet`, a base aberta, com as internações até 31/12/2025;
- `data/processed/sih_norte_minas_selado.parquet`, a base lacrada, com as internações a partir de 01/01/2026,
  reservada para a rodada confirmatória.

Com as competências até julho de 2026, a base aberta fica com 825.434 linhas e a lacrada com 79.237; o ETL mostra
esses números no fim. Nas competências de 2026, nem a tela nem o log mostram quantos óbitos houve.

Os `.dbc` passam por `data/raw/` e são apagados depois de lidos. O ETL apaga também os arquivos do mesmo mês em
`~/pysus`, onde a versão 2.11 do PySUS grava mesmo quando recebe outra pasta. As competências que já estão em
`data/interim/` são puladas, então um download interrompido continua de onde parou. Um mês que falha nas três
tentativas fica registrado no log com o erro, e a consolidação acontece sem ele; nesse caso, rode `make dados` de novo.

Para refazer exatamente os resultados deste repositório, deixe `ULTIMO_MES` em 2026-07. Competências mais novas mudam as
duas bases: a lacrada ganha internações de 2026, e a aberta pode ganhar AIH de 2025 cobradas com atraso, que os
exploratórios usam. Mesmo com 2026-07, o DATASUS pode ter republicado algum arquivo depois que baixei os dados. Cada
`x7*.json` guarda, no campo `sha256_dados`, o SHA-256 das duas bases de `data/processed/` que geraram os resultados:

| Base | SHA-256 |
|---|---|
| `sih_norte_minas.parquet` | `fd5fec252d613728cbfc99407819b7a9f5345f7a8d4ea44f24e9e1dfdbc23ea0` |
| `sih_norte_minas_selado.parquet` | `5267b86fa61992f0091a8d13db09f22d2d3ba70edda10acca4854838bf849c6c` |

Se as suas bases derem outro SHA-256, os números podem não bater com os publicados. Há uma diferença conhecida que não
afeta os resultados: baixando tudo de novo numa pasta limpa, a base lacrada saiu com o mesmo SHA-256, e a aberta, com as
mesmas 825.434 linhas, saiu com `76f6bcc5fca7dc8d7db23080b273aac93b8213a086cafba995a08e37da2fe364`. Os dois primeiros
meses que eu tinha baixado, fev/2019 e jan/2024, gravaram os 19 campos de valores pagos (os VAL_ e o US_TOT) com duas
casas, como `258.30`, e o FTP traz `258.3`. Nenhum desses campos entra nas análises; os exploratórios, as figuras e os notebooks refeitos com essa base saíram
iguais aos publicados. As rodadas confirmatórias não foram refeitas: cada arquivo saiu uma vez, como registra o diário
de desvios.

## Experimentos exploratórios

Os exploratórios usam só as internações de 2019 a 2025: a leitura da base para com erro se encontrar alguma de 2026. O
X1 descreve a deriva por trimestre. Do X2 ao X4, a pergunta é o que acontece com G1 e G2: no X2 a calibração de 2022 só
envelhece, e o X3 põe nesses mesmos trimestres as estratégias de atualização (a) a (g), comparação que o X4 repete na
pandemia, sem a (d), que exigiria dados anteriores a 2019. Equidade e utilidade ficam com o X5 e o X6, e o X8 verifica quando dois métodos de monitoramento sinalizam a
deriva. `make exploratorios` roda tudo em cerca de 25 minutos e grava em oito pastas:

| Pasta | Conteúdo |
|---|---|
| `reports/exploratorios/morte/nenhuma/` | X1 a X6 e X8, com o óbito como desfecho |
| `reports/exploratorios/morte/s1/` a `s6/` | X3 com cada sensibilidade |
| `reports/exploratorios/perm_prolongada/nenhuma/` | X3 e X4 com a permanência prolongada como desfecho |

Cada pasta tem um `resultados_<experimento>.json` por experimento, tabelas por período em CSV e um `resumo.json` com as
hipóteses exploratórias.

## Rodada confirmatória

`make confirmatorio` leva cerca de 15 minutos e grava em `reports/confirmatorio/` os arquivos `x7a.json`, `x7b.json` e
`x7a_s1.json` a `x7a_s6.json`. Antes de ajustar os modelos, a rodada confere a base lacrada (datas de internação a
partir de 01/01/2026 e nenhum N_AIH repetido entre as duas bases), o seguimento dos meses do teste contra o mesmo
mês do ano anterior (razão abaixo de 0,90 recusa) e para se faltar alguma competência entre janeiro de
2019 e julho de 2026. Cada arquivo registra as versões das bibliotecas e o SHA-256 das duas bases. Os arquivos de 26 e
27/09 não registram o commit do código; os gravados depois deles, como o da X7c, trazem também no bloco `ambiente` o
commit (`commit`) e se a árvore de trabalho tinha mudanças não commitadas (`arvore_suja`), com `null` nos dois quando
não há git. Com mudanças não commitadas, inclusive um arquivo novo ainda não rastreado, a rodada para antes de ler as
bases, mesmo numa reexecução com `--desvio`: commite ou descarte as mudanças antes de rodar. A checagem do seguimento
já estava no código do pré-registro.

Os arquivos publicados trazem os resultados das execuções de 26 e 27/09/2026 que decidiram as hipóteses.

A rodada não sobrescreve um resultado que já existe. Os resultados publicados já estão na
pasta, e `make confirmatorio` para logo no X7a. Para refazer a rodada inteira, apague `reports/confirmatorio/` antes,
depois de guardar uma cópia de segurança dos arquivos publicados.

O X7c aplica a regra do X7a às internações de abril a junho de 2026 e precisa das competências até outubro:

```bash
make dados ULTIMO_MES=2026-10
make confirmatorio-c
```

Ele refaz o ajuste do X7a com as mesmas linhas e só segue se a regra sair igual à registrada em `x7a.json`. O resultado
vai para `reports/confirmatorio/x7c.json`.

## Figuras e notebooks

`make figuras` usa só os exploratórios sem sensibilidade: lê `reports/exploratorios/morte/nenhuma/` e
`reports/exploratorios/perm_prolongada/nenhuma/` e grava em `reports/figures/morte/` e
`reports/figures/perm_prolongada/`. Os dois desfechos ganham as figuras de garantias por trimestre do X3 e do X4, com as
variantes de parâmetro à parte. Só o óbito tem as de equidade (X5), calibração e utilidade (X6).

Os notebooks leem os resultados de `reports/`, então `make notebooks` vem por último; ele executa cada notebook e grava
as saídas no próprio arquivo. O 1.0 também lê as bases de `data/processed/` (da lacrada, só as datas de internação e as
competências, para as razões de seguimento), o 2.0 lê a base aberta, e o 7.0 calcula os tamanhos exatos dos testes
estatísticos e os confere por simulação, com semente fixa.

## Testes, lint e documentação

A suíte de testes não depende de `make dados`: os testes geram dados sintéticos (`tests/dados_sinteticos.py`).
`make testes` roda o pytest com `RuntimeWarning` tratado como erro, e `make lint`, o `ruff check` seguido de
`ruff format --check`. O site da documentação é montado em `site/` por `make docs`, com o mkdocs no modo estrito, em que
qualquer aviso vira erro.


## Limitação de prova

Duas datas deste repositório não são verificáveis de fora: o horário do download de 2026 vem do log do ETL, que mora
na máquina do autor, e a tag do pré-registro não é assinada nem tem testemunha pública. Quem recebe este repositório
recebe a cronologia como declaração, não como prova. A X7c nasce diferente: antes de sair a competência de outubro,
código e plano serão depositados em arquivo com carimbo externo (Zenodo ou OSF), e a rodada gravará o commit da
execução no JSON de saída. A limitação está registrada no diário.
