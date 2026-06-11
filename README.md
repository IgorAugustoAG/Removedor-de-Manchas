# Removedor de Manchas

Script em Python que detecta, separa, reconstrói e remove manchas de imagens (documentos, papéis envelhecidos, fotos), gerando relatórios e uma comparação visual antes/depois.

## Requisitos

- Python 3.8 ou superior
- Bibliotecas:
  - `opencv-python`
  - `numpy`
  - `scikit-image`

### Instalação

```bash
pip install opencv-python numpy scikit-image
```

## Como usar

```bash
python3 removedor_manchas.py <imagem> [pasta_saida] [sensibilidade] [tipo]
```

### Parâmetros

| Parâmetro | Obrigatório | Padrão | Descrição |
|---|---|---|---|
| `imagem` | Sim | — | Caminho da imagem de entrada (jpg, png, etc.) |
| `pasta_saida` | Não | `saida` | Pasta onde os resultados serão salvos |
| `sensibilidade` | Não | `1.0` | Quanto maior, mais sensível à detecção de manchas (use valores entre 0.5 e 2.0) |
| `tipo` | Não | `auto` | Tipo da imagem: `branca` (documentos/papel), `colorida` (fotos) ou `auto` (detecção automática) |

### Exemplos

```bash
# Detecção automática
python3 removedor_manchas.py foto.jpg resultados

# Foto colorida, menos sensível
python3 removedor_manchas.py foto.jpg resultados 0.5 colorida

# Documento em papel branco
python3 removedor_manchas.py documento.png resultados 1.0 branca
```

## Saídas geradas

Dentro da pasta de saída são criados:

- `01_deteccao.png` — manchas destacadas na imagem original
- `02_frente.png` — imagem sem as áreas de mancha (com transparência)
- `02_fundo_manchas.png` — apenas as regiões de mancha isoladas
- `03_reconstruida.png` — imagem com as manchas reconstruídas
- `04_imagem_final.png` — resultado final com as manchas removidas
- `04_diferenca.png` — mapa de diferença entre original e final
- `05_antes_depois.png` — comparação lado a lado (antes/depois)
- `relatorio_completo.json` — relatório detalhado de todas as etapas do processamento

## Como funciona (pipeline em 4 etapas)

1. **Detecção** — identifica as áreas com manchas usando análise de cor, textura e desvio local
2. **Separação** — isola as regiões de mancha do restante da imagem
3. **Reconstrução** — repara as áreas marcadas (preenchimento de cor de fundo para documentos brancos, ou correção de cor preservando detalhes para fotos)
4. **Remoção** — aplica o resultado reconstruído sobre a imagem original e gera os relatórios finais
