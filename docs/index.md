# Triagem de baixo risco de óbito hospitalar sob deriva temporal

O trabalho usa as AIH do SIH-SUS dos hospitais dos 89 municípios do Norte de Minas, de 2019 a 2026. Com as internações
até 2025, treinei um modelo que estima, na admissão, o risco de óbito hospitalar e montei sobre ele uma regra conformal
que libera as de baixo risco; as de 2026 ficaram guardadas para o teste. O TCC testa se as garantias dessa regra
continuam valendo quando ela é aplicada a internações de meses ou anos depois das que serviram para ajustá-la.

- [Dados](dados.md): as variáveis de admissão, as que ficaram de fora e os marcadores de subgrupo.
- [Como reproduzir](reproduzir.md): os alvos do `Makefile` e o que cada um grava.
- [Pré-registro](pre-registro.md): a pergunta, os dados, o modelo, as rodadas e os critérios de refutação, fixados
  antes do download dos dados de 2026.

Os resultados de 2026 estão resumidos no `README.md` do repositório, e as tabelas completas ficam nos notebooks, em
`notebooks/`. As mudanças feitas depois do pré-registro estão em `reports/confirmatorio/desvios.md`.
