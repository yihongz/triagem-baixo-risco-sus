const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, AlignmentType, HeadingLevel, Header, Footer, PageNumber,
  FootnoteReferenceRun, Table, TableRow, TableCell, WidthType, BorderStyle, NumberFormat, ImageRun,
} = require("docx");

const cm = (x) => Math.round(x * 567);
const path = require("path");
// Pasta com o texto das seções: segundo argumento da linha de comando ou ../texto ao lado do gerador
const TEXTO = process.argv[3] || path.join(__dirname, "..", "texto");
// *itálico* e **negrito** dentro do parágrafo
const runsDe = (texto) =>
  texto
    .split(/(\*\*[^*]+\*\*|\*[^*]+\*)/)
    .filter(Boolean)
    .map((t) =>
      t.startsWith("**")
        ? new TextRun({ text: t.slice(2, -2), bold: true })
        : t.startsWith("*")
          ? new TextRun({ text: t.slice(1, -1), italics: true })
          : new TextRun(t),
    );
const blocos = (arquivo) =>
  fs
    .readFileSync(path.join(TEXTO, arquivo), "utf8")
    .split(/\n\s*\n/)
    .map((b) => b.replace(/\s*\n\s*/g, " ").trim())
    .filter(Boolean);
// Um bloco [[nome]] no texto insere os elementos de insercoes[nome] (quadros, tabelas)
const insercoes = {};
const paragrafos = (arquivo) => blocos(arquivo).flatMap((b) => {
  const m = b.match(/^\[\[(.+)\]\]$/);
  if (m && !insercoes[m[1]]) throw new Error(`inserção desconhecida: ${m[1]}`);
  return m ? insercoes[m[1]] : [new Paragraph({ children: runsDe(b) })];
});
const referencias = (arquivo) => blocos(arquivo).map((b) => new Paragraph({ style: "Referencia", children: runsDe(b) }));
const FONTE = "Times New Roman";
const AUTOR = "Paulo Victor Soares Almeida";
const ORIENTADOR = "Allysson Steve Mota Lacerda";
const TITULO = "TRIAGEM DE BAIXO RISCO DE ÓBITO HOSPITALAR SOB DERIVA TEMPORAL";
const SUBTITULO = "garantias conformais com dados do SIH-SUS no Norte de Minas (2019-2026)";
// Nas folhas pré-textuais o título vai inteiro em caixa alta e quebra sozinho; no artigo, as quebras são fixas,
// para o intervalo "2019-2026" não se partir no fim da linha
const tituloRuns = ({ size = 24, caixaAlta = false }) =>
  caixaAlta
    ? [new TextRun({ text: `${TITULO}: ${SUBTITULO.toUpperCase()}`, size, bold: true })]
    : [
        new TextRun({ text: "TRIAGEM DE BAIXO RISCO DE ÓBITO HOSPITALAR", size, bold: true }),
        new TextRun({ text: "SOB DERIVA TEMPORAL: garantias conformais com dados", size, bold: true, break: 1 }),
        new TextRun({ text: "do SIH-SUS no Norte de Minas (2019-2026)", size, bold: true, break: 1 }),
      ];
const tituloPar = (opcoes, paragrafo = {}) =>
  new Paragraph({ style: "Centro", ...paragrafo, children: tituloRuns(opcoes) });
const TITULO_EN =
  "Triage of hospitalizations at low risk of in-hospital death under temporal drift: conformal guarantees " +
  "using SIH-SUS data from northern Minas Gerais (2019-2026)";
const NATUREZA =
  "Artigo apresentado ao curso de Pós-Graduação Lato Sensu em Ciência de Dados da Universidade Estadual de " +
  "Montes Claros, como requisito parcial para obtenção do título de Especialista em Ciência de Dados.";
const ANO = "2026";

const margens = { top: cm(3), left: cm(3), bottom: cm(2), right: cm(2), header: cm(2), footer: cm(1.25) };

// Parágrafos utilitários
const centro = (texto, { size = 24, bold = false, before = 0, after = 0, pageBreakBefore = false, allCaps = false } = {}) =>
  new Paragraph({
    style: "Centro",
    pageBreakBefore,
    spacing: { before, after, line: 240 },
    children: [new TextRun({ text: texto, size, bold, allCaps })],
  });

// Local e ano ficam no rodapé da seção pré-textual: sem negrito na capa, em negrito nas outras folhas
const rodapeLocalAno = (bold) =>
  new Footer({
    children: [
      new Paragraph({ style: "Centro", children: [new TextRun({ text: "Montes Claros - MG", size: 28, bold })] }),
      new Paragraph({ style: "Centro", children: [new TextRun({ text: ANO, size: 28, bold })] }),
    ],
  });

const orientacao = (texto) =>
  new Paragraph({ style: "Orientacao", children: [new TextRun({ text: `[${texto}]` })] });

const h1 = (texto, opcoes = {}) => new Paragraph({ heading: HeadingLevel.HEADING_1, ...opcoes, children: [new TextRun(texto)] });
const h2 = (texto) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(texto)] });

// Capa
const capa = [
  centro("UNIVERSIDADE ESTADUAL DE MONTES CLAROS", { size: 28, bold: true }),
  centro("Centro de Ciências Exatas e Tecnológicas", { size: 28 }),
  centro("Pós-Graduação Lato Sensu em Ciência de Dados", { size: 28 }),
  centro(AUTOR, { size: 28, before: 560 }),
  tituloPar({ size: 32, caixaAlta: true }, { spacing: { before: cm(4.2), line: 240 } }),
];

// Folha de rosto
const rosto = [
  centro(AUTOR, { size: 28, bold: true, pageBreakBefore: true }),
  tituloPar({ size: 28, caixaAlta: true }, { spacing: { before: cm(5.5), line: 240 } }),
  new Paragraph({ style: "Natureza", spacing: { before: 280, line: 240 }, children: [new TextRun(NATUREZA)] }),
  new Paragraph({
    style: "Natureza",
    spacing: { before: 280, line: 240 },
    children: [new TextRun("Orientador: Prof. Dr. "), new TextRun({ text: ORIENTADOR, bold: true })],
  }),
];

// Folha de aprovação
const linhaAssinatura = (nome, instituicao, before) => [
  centro("_______________________________________________", { before }),
  centro(nome),
  centro(instituicao),
];
const aprovacao = [
  centro(AUTOR, { size: 28, bold: true, pageBreakBefore: true }),
  tituloPar({ size: 28, caixaAlta: true }, { spacing: { before: cm(1.5), line: 240 } }),
  new Paragraph({ style: "Natureza", spacing: { before: 280, line: 240 }, children: [new TextRun(NATUREZA)] }),
  new Paragraph({
    style: "Centro",
    alignment: AlignmentType.LEFT,
    spacing: { before: cm(1.2), line: 240 },
    children: [new TextRun(`Aprovado em: ____ de ____________________ de ${ANO}.`)],
  }),
  centro("BANCA EXAMINADORA", { bold: true, before: cm(1) }),
  ...linhaAssinatura(`Prof. Dr. ${ORIENTADOR} (Orientador)`, "Universidade Estadual de Montes Claros", cm(1.4)),
  ...linhaAssinatura("Prof.(a) [nome do avaliador]", "[instituição]", cm(1.4)),
  ...linhaAssinatura("Prof.(a) [nome do avaliador]", "[instituição]", cm(1.4)),
];

// Tabela no padrão do IBGE: traços em cima, abaixo do cabeçalho e embaixo; sem linhas verticais nem internas
const SEM = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const TRACO = { style: BorderStyle.SINGLE, size: 8, color: "000000" };
const soma = (v) => v.reduce((a, b) => a + b, 0);
const linhasDe = (arquivo) =>
  fs.readFileSync(path.join(TEXTO, arquivo), "utf8").split("\n").map((l) => l.trim()).filter(Boolean)
    .map((l) => l.split(" | "));
const legenda = (texto) => new Paragraph({ style: "Legenda", keepNext: true, children: [new TextRun(texto)] });
const nota = (texto, ultima) => new Paragraph({ style: "Fonte", keepNext: !ultima, indent: { left: 0, firstLine: 0 }, spacing: { before: 60, after: ultima ? 240 : 0, line: 240 }, children: runsDe(texto) });
const rodape = (linhas) => linhas.map((t, i) => nota(t, i === linhas.length - 1));
const celula = (texto, largura, { negrito, alinhamento, bordas }) => new TableCell({
  width: { size: largura, type: WidthType.DXA },
  borders: bordas,
  margins: { left: 60, right: 60 },
  children: [new Paragraph({ style: "Tabela", keepNext: true, alignment: alinhamento, children: negrito ? [new TextRun({ text: texto, bold: true })] : runsDe(texto) })],
});
const tabelaDe = (arquivo, larg) => {
  const rows = linhasDe(arquivo);
  return new Table({
    width: { size: soma(larg), type: WidthType.DXA },
    columnWidths: larg,
    borders: { top: SEM, bottom: SEM, left: SEM, right: SEM, insideHorizontal: SEM, insideVertical: SEM },
    rows: rows.map((linha, r) => new TableRow({
      tableHeader: r === 0,
      cantSplit: true,
      children: linha.map((t, i) => celula(t, larg[i], {
        negrito: r === 0,
        alinhamento: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
        bordas: { top: r === 0 ? TRACO : SEM, bottom: r === 0 || r === rows.length - 1 ? TRACO : SEM, left: SEM, right: SEM },
      })),
    })),
  });
};
// Quadro no padrão da ABNT: texto em células fechadas por traços em todos os lados
const quadroDe = (arquivo, larg) => {
  const rows = linhasDe(arquivo);
  const bordas = { top: TRACO, bottom: TRACO, left: TRACO, right: TRACO };
  return new Table({
    width: { size: soma(larg), type: WidthType.DXA },
    columnWidths: larg,
    rows: rows.map((linha, r) => new TableRow({
      tableHeader: r === 0,
      cantSplit: true,
      children: linha.map((t, i) => celula(t, larg[i], { negrito: r === 0, alinhamento: r === 0 ? AlignmentType.CENTER : AlignmentType.LEFT, bordas })),
    })),
  });
};
// Figura: título em cima, imagem centralizada na largura do texto e fonte embaixo
const FIGURAS = path.join(__dirname, "..", "figuras");
const figuraDe = (arquivo, larguraCm = 16) => {
  const dados = fs.readFileSync(path.join(FIGURAS, arquivo));
  const w = dados.readUInt32BE(16), h = dados.readUInt32BE(20);
  const largura = Math.round((larguraCm / 2.54) * 96);
  return new Paragraph({
    alignment: AlignmentType.CENTER, keepNext: true, indent: { firstLine: 0 }, spacing: { line: 240, before: 0, after: 0 },
    children: [new ImageRun({ type: "png", data: dados, transformation: { width: largura, height: Math.round((largura * h) / w) } })],
  });
};
const FONTE_DADOS = "Fonte: elaborada pelo autor (2026), com dados do SIH-SUS.";
const tabela = (titulo, arquivo, larg, notas = []) => [legenda(titulo), tabelaDe(arquivo, larg), ...rodape([FONTE_DADOS, ...notas])];
const figura = (titulo, arquivo, notas = []) => [legenda(titulo), figuraDe(arquivo), ...rodape([FONTE_DADOS, ...notas])];
const colunas = (primeira, n) => [primeira, ...Array(n).fill(Math.floor((cm(16) - primeira) / n))];

insercoes["quadro-hipoteses"] = [
  legenda("Quadro 1 – Hipóteses da rodada confirmatória e critérios de refutação"),
  quadroDe("quadro-hipoteses.md", [1000, 3700, 4372]),
  ...rodape(["Fonte: elaborado pelo autor (2026), com base no pré-registro."]),
];
insercoes["tabela-coorte"] = tabela("Tabela 1 – Características das internações por ano, Norte de Minas, 2019 a 2025",
  "tabela-coorte.md", colunas(2832, 7),
  ["Nota: sinal convencional utilizado: .. não se aplica dado numérico. B34 e U07 são as categorias da CID-10 que incluem a covid-19. A última linha é a AUROC de um classificador treinado para distinguir as internações de cada ano das de 2019."]);
insercoes["figura-coorte"] = figura("Figura 1 – Mortalidade por trimestre e distinção de cada ano em relação a 2019", "figura-coorte.png");
insercoes["figura-garantias"] = figura("Figura 2 – Garantias G1 e G2 por trimestre, com as estratégias estática e trimestral com atraso", "figura-garantias.png",
  ["Nota: na linha de baixo, valores acima de 1 indicam taxa de óbito entre as liberadas acima do limite r̂."]);
insercoes["tabela-estrategias"] = tabela("Tabela 2 – Trimestres com violação das garantias e distância média do O/E até 1, por estratégia de atualização",
  "tabela-estrategias.md", colunas(2600, 6),
  ["Nota: violação significativa após a correção de Holm entre os trimestres de cada experimento. Sinal convencional utilizado: .. não se aplica dado numérico; a estratégia (d) não foi avaliada na pandemia."]);
insercoes["figura-equidade"] = figura("Figura 3 – Óbitos liberados por faixa etária com a regra única e com a regra por faixa etária, 2023 a 2025", "figura-equidade.png",
  ["Nota: as barras de erro são intervalos de confiança exatos de 95%."]);
insercoes["tabela-utilidade"] = tabela("Tabela 3 – Internações e óbitos liberados por modelo ou regra em 2025, com α = 0,05",
  "tabela-utilidade.md", colunas(3672, 3),
  ["Nota: sinal convencional utilizado: ... dado numérico não disponível. A última linha usa o limiar que reproduz, na parte B, a fração de óbitos liberados pela regra sem modelo. As diferenças citadas no texto foram calculadas antes do arredondamento."]);
insercoes["tabela-confirmatoria"] = tabela("Tabela 4 – Resultados da rodada confirmatória nas internações de janeiro a março de 2026, com α = 0,05",
  "tabela-confirmatoria.md", colunas(4672, 2),
  ["Nota: p.p. = pontos percentuais. Os intervalos das diferenças vêm de 1.000 reamostragens de hospitais. Contra a regra sem modelo, o *boosting* usa o limiar que reproduz, na parte B, a fração de óbitos liberados por ela; contra a logística, ambos usam α = 0,05."]);
insercoes["tabela-decisoes"] = tabela("Tabela 5 – Decisões sobre as hipóteses na rodada confirmatória, com α = 0,05",
  "tabela-decisoes.md", colunas(3072, 2),
  ["Nota: com a regra de α = 0,10, as decisões não mudam."]);

// Artigo
const artigo = [
  tituloPar({ size: 24 }, { spacing: { line: 360 } }),
  new Paragraph({
    style: "Centro",
    spacing: { before: 240, line: 360 },
    children: [new TextRun({ text: TITULO_EN, italics: true })],
  }),
  new Paragraph({
    style: "Centro",
    alignment: AlignmentType.RIGHT,
    spacing: { before: 480, line: 240 },
    children: [new TextRun(AUTOR), new FootnoteReferenceRun(1)],
  }),
  new Paragraph({
    style: "Centro",
    alignment: AlignmentType.RIGHT,
    spacing: { after: 480, line: 240 },
    children: [new TextRun(ORIENTADOR), new FootnoteReferenceRun(2)],
  }),
  new Paragraph({
    style: "Resumo",
    children: [
      new TextRun({ text: "Resumo: ", bold: true }),
      ...runsDe(blocos("resumo.md")[0]),
    ],
  }),
  new Paragraph({
    style: "Resumo",
    children: [
      new TextRun({ text: "Palavras-chave: ", bold: true }),
      new TextRun("predição conformal; deriva temporal; mortalidade hospitalar; SIH-SUS; aprendizado de máquina."),
    ],
  }),
  new Paragraph({
    style: "Resumo",
    children: [
      new TextRun({ text: "Abstract: ", bold: true }),
      ...runsDe(blocos("abstract.md")[0]),
    ],
  }),
  new Paragraph({
    style: "Resumo",
    spacing: { line: 240, after: 480 },
    children: [
      new TextRun({ text: "Keywords: ", bold: true }),
      new TextRun("conformal prediction; temporal drift; in-hospital mortality; SIH-SUS; machine learning."),
    ],
  }),

  h1("1 INTRODUÇÃO"),
  ...paragrafos("introducao.md"),

  h1("2 FUNDAMENTAÇÃO TEÓRICA"),
  h2("2.1 DADOS HOSPITALARES DO SUS"),
  ...paragrafos("fundamentacao-dados-sus.md"),
  h2("2.2 PREDIÇÃO DE RISCO E CALIBRAÇÃO"),
  ...paragrafos("fundamentacao-risco-calibracao.md"),
  h2("2.3 PREDIÇÃO CONFORMAL"),
  ...paragrafos("fundamentacao-conformal.md"),
  h2("2.4 DERIVA TEMPORAL"),
  ...paragrafos("fundamentacao-deriva.md"),

  h1("3 MATERIAIS E MÉTODOS"),
  h2("3.1 DADOS E COORTE"),
  ...paragrafos("dados-e-coorte.md"),
  h2("3.2 MODELO, CALIBRAÇÃO E LINHAS DE BASE"),
  ...paragrafos("modelo-calibracao-linhas-de-base.md"),
  h2("3.3 REGRA DE LIBERAÇÃO E GARANTIAS"),
  ...paragrafos("regra-e-garantias.md"),
  h2("3.4 EXPERIMENTOS EXPLORATÓRIOS (2019-2025)"),
  ...paragrafos("experimentos-exploratorios.md"),
  h2("3.5 RODADA CONFIRMATÓRIA E PRÉ-REGISTRO"),
  ...paragrafos("rodada-confirmatoria.md"),
  h2("3.6 ANÁLISE ESTATÍSTICA"),
  ...paragrafos("analise-estatistica.md"),
  h2("3.7 DISPONIBILIDADE DOS DADOS E DO CÓDIGO"),
  ...paragrafos("disponibilidade.md"),

  h1("4 RESULTADOS"),
  h2("4.1 COORTE E DERIVA TEMPORAL"),
  ...paragrafos("resultados-coorte.md"),
  h2("4.2 GARANTIAS SOB DERIVA (2019-2025)"),
  ...paragrafos("resultados-garantias.md"),
  h2("4.3 EQUIDADE E UTILIDADE"),
  ...paragrafos("resultados-equidade-utilidade.md"),
  h2("4.4 RODADA CONFIRMATÓRIA DE 2026"),
  ...paragrafos("resultados-confirmatoria.md"),

  h1("5 DISCUSSÃO"),
  ...paragrafos("discussao.md"),

  h1("6 CONSIDERAÇÕES FINAIS"),
  ...paragrafos("consideracoes-finais.md"),

  h1("REFERÊNCIAS", { alignment: AlignmentType.CENTER }),
  ...referencias("referencias.md"),
];

const doc = new Document({
  creator: AUTOR,
  title: `${TITULO}: ${SUBTITULO}`,
  styles: {
    default: { document: { run: { font: FONTE, size: 24 } } },
    paragraphStyles: [
      {
        id: "Normal", name: "Normal", quickFormat: true,
        run: { font: FONTE, size: 24 },
        paragraph: { alignment: AlignmentType.JUSTIFIED, spacing: { line: 360, before: 0, after: 0 }, indent: { firstLine: cm(2) } },
      },
      {
        id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONTE, size: 24, bold: true, allCaps: true, color: "000000" },
        paragraph: { alignment: AlignmentType.LEFT, indent: { firstLine: 0 }, spacing: { before: 360, after: 360, line: 360 }, keepNext: true, outlineLevel: 0 },
      },
      {
        id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONTE, size: 24, bold: false, allCaps: true, color: "000000" },
        paragraph: { alignment: AlignmentType.LEFT, indent: { firstLine: 0 }, spacing: { before: 360, after: 360, line: 360 }, keepNext: true, outlineLevel: 1 },
      },
      {
        id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONTE, size: 24, bold: true, color: "000000" },
        paragraph: { alignment: AlignmentType.LEFT, indent: { firstLine: 0 }, spacing: { before: 360, after: 360, line: 360 }, keepNext: true, outlineLevel: 2 },
      },
      {
        id: "FootnoteText", name: "footnote text", basedOn: "Normal",
        run: { font: FONTE, size: 20 },
        paragraph: { alignment: AlignmentType.JUSTIFIED, indent: { firstLine: 0 }, spacing: { line: 240, before: 0, after: 0 } },
      },
      { id: "Centro", name: "Centralizado", basedOn: "Normal", paragraph: { alignment: AlignmentType.CENTER, indent: { firstLine: 0 }, spacing: { line: 240 } } },
      { id: "Natureza", name: "Natureza do trabalho", basedOn: "Normal", paragraph: { alignment: AlignmentType.JUSTIFIED, indent: { left: cm(8), firstLine: 0 }, spacing: { line: 240 } } },
      { id: "Resumo", name: "Resumo", basedOn: "Normal", paragraph: { alignment: AlignmentType.JUSTIFIED, indent: { firstLine: 0 }, spacing: { line: 240, after: 240 } } },
      { id: "Legenda", name: "Legenda", basedOn: "Normal", run: { size: 20 }, paragraph: { alignment: AlignmentType.CENTER, indent: { firstLine: 0 }, spacing: { line: 240, before: 240, after: 60 } } },
      { id: "Fonte", name: "Fonte da ilustração", basedOn: "Normal", run: { size: 20 }, paragraph: { alignment: AlignmentType.LEFT, indent: { firstLine: 0 }, spacing: { line: 240 } } },
      { id: "Tabela", name: "Texto de tabela", basedOn: "Normal", run: { size: 20 }, paragraph: { indent: { firstLine: 0 }, spacing: { line: 240, before: 40, after: 40 } } },
      { id: "Referencia", name: "Referência", basedOn: "Normal", paragraph: { alignment: AlignmentType.LEFT, indent: { firstLine: 0 }, spacing: { line: 240, after: 240 } } },
      { id: "CitacaoLonga", name: "Citação longa", basedOn: "Normal", run: { size: 20 }, paragraph: { alignment: AlignmentType.JUSTIFIED, indent: { left: cm(4), firstLine: 0 }, spacing: { line: 240, before: 240, after: 240 } } },
      { id: "Orientacao", name: "Orientação (apagar)", basedOn: "Normal", run: { italics: true, color: "7F7F7F" }, paragraph: { alignment: AlignmentType.LEFT } },
    ],
  },
  footnotes: {
    1: { children: [new Paragraph({ style: "FootnoteText", children: [new TextRun("Especializando em Ciência de Dados pela Universidade Estadual de Montes Claros (Unimontes). " +
      "Graduado em Engenharia Biomédica pelas Faculdades Unidas do Norte de Minas (FUNORTE) e em Engenharia de " +
      "Software pelo Centro de Ensino Superior de Maringá (CESUMAR). E-mail: engpauloxyz@gmail.com.")] })] },
    2: {
      children: [
        new Paragraph({
          style: "FootnoteText",
          children: [new TextRun("Orientador. Doutor em Engenharia Elétrica pela Universidade Federal de Minas Gerais (UFMG). Professor do Departamento de Ciências da Computação da Unimontes. E-mail: steve.lacerda@unimontes.br.")],
        }),
      ],
    },
  },
  sections: [
    {
      properties: { titlePage: true, page: { margin: { ...margens, footer: cm(2) } } },
      footers: { first: rodapeLocalAno(false), default: rodapeLocalAno(true) },
      children: [...capa, ...rosto, ...aprovacao],
    },
    {
      properties: { page: { margin: margens, pageNumbers: { start: 3, formatType: NumberFormat.DECIMAL } } },
      footers: { default: new Footer({ children: [new Paragraph({ style: "Centro", children: [] })] }) },
      headers: {
        default: new Header({
          children: [
            new Paragraph({
              alignment: AlignmentType.RIGHT,
              indent: { firstLine: 0 },
              children: [new TextRun({ children: [PageNumber.CURRENT], size: 20 })],
            }),
          ],
        }),
      },
      children: artigo,
    },
  ],
});

const saida = process.argv[2];
Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(saida, buf);
  console.log("gravado:", saida, buf.length, "bytes");
});
