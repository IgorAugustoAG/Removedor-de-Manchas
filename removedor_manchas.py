"""
=====================================================
  REMOVEDOR DE MANCHAS — Sistema de 4 Etapas v2.4
=====================================================
  Correções v2.1:
  ✅ Detecção por saturação HSV para fundo branco
     (captura centro claro das manchas de café)
  ✅ Preenchimento de contornos (elimina "anel vazio")
  ✅ Reconstrução por cor de fundo para imagens brancas
  ✅ Inpainting progressivo multi-escala para coloridas
  ✅ Auto-detecção do tipo de imagem
  ✅ Sem blur gaussiano na máscara (não borra texto)

  Correções v2.2:
  ✅ Correção de cor por referência GLOBAL do papel
  ✅ Kernel de blur adaptativo ao tamanho de CADA mancha

  Correções v2.3:
  ✅ CORRIGIDO BUG CRÍTICO: quando o fundo da imagem é muito
     uniforme (ex.: papel branco liso), o cálculo da "cor do
     papel" usando percentil podia resultar em NaN, e
     `NaN.astype(uint8)` virava 0 — o que pintava as manchas
     de PRETO SÓLIDO em vez de removê-las (imagem 4). Agora há
     fallback seguro + `np.nan_to_num` como rede de segurança.
  ✅ Kernel de correção agora é PROPORCIONAL e MENOR que cada
     mancha (antes era maior, o que fazia o blur "enxergar"
     demais o papel ao redor e resultar em fator≈1, ou seja,
     nenhuma correção visível em manchas grandes).
  ✅ Fator de correção limitado a 5x (era 6x) para reduzir
     manchas amarelas artificiais por overcorrection.
  ✅ Removido o detector "mascara_saturada" (saturação global)
     da composição final de `_detectar_colorida` — mesmo com
     desvio local, ele continuava marcando áreas naturalmente
     quentes/texturizadas de fotos reais (grama, pele, pedras
     ao sol) como mancha, criando borrões artificiais.

  Correções v2.4 (esta versão):
  ✅ Novo detector "mascara_clara": captura manchas CLARAS/
     translúcidas (ex.: respingos de água que clareiam e
     "lavam" a cor da foto por baixo). mascara_cafe e mascara_b
     só pegam manchas amareladas/acastanhadas; manchas que
     deixam a área mais clara e menos saturada que a vizinhança
     passavam batido. Continua exigindo desvio local
     (mascara_local) para não marcar áreas naturalmente claras
     da foto.
=====================================================
Dependências: pip install opencv-python numpy scikit-image
"""

import cv2
import numpy as np
from skimage import measure
import os
import json
from datetime import datetime


def _carregar_imagem(caminho: str) -> np.ndarray:
    img = cv2.imread(caminho)
    if img is None:
        raise FileNotFoundError(f"Imagem não encontrada: {caminho}")
    return img


def _salvar_imagem(img: np.ndarray, caminho: str):
    os.makedirs(os.path.dirname(caminho) if os.path.dirname(caminho) else ".", exist_ok=True)
    cv2.imwrite(caminho, img)


def _timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _imprimir_relatorio(titulo: str, dados: dict):
    largura = 60
    print("\n" + "=" * largura)
    print(f"  {titulo}")
    print(f"  {_timestamp()}")
    print("-" * largura)
    for chave, valor in dados.items():
        print(f"  {chave:<30} {valor}")
    print("=" * largura + "\n")


def _detectar_tipo_imagem(img_bgr: np.ndarray) -> str:
    img_hsv   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    brilho    = img_hsv[:, :, 2].astype(np.float32)
    saturacao = img_hsv[:, :, 1].astype(np.float32)

    pct_branco = float(np.mean((brilho > 200) & (saturacao < 40)))
    var_cor    = float(np.mean([np.std(img_bgr[:, :, c].astype(float)) for c in range(3)]))

    tipo = "branca" if (pct_branco > 0.25 or var_cor < 35) else "colorida"
    print(f"  Tipo detectado: {tipo.upper()}  "
          f"(fundo branco: {pct_branco:.1%} | variancia de cor: {var_cor:.1f})")
    return tipo


def _limpar_mascara(mascara: np.ndarray, tamanho_minimo: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    m = cv2.morphologyEx(mascara, cv2.MORPH_OPEN,  kernel)
    m = cv2.morphologyEx(m,       cv2.MORPH_CLOSE, kernel)

    contornos, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    m_preenchida = np.zeros_like(m)
    for contorno in contornos:
        if cv2.contourArea(contorno) >= tamanho_minimo:
            cv2.drawContours(m_preenchida, [contorno], -1, 1, cv2.FILLED)

    return m_preenchida


def _detectar_fundo_branco(
    img_bgr: np.ndarray,
    limiar_cor: float,
    limiar_relevo: float,
    limiar_textura: float,
) -> tuple:
    img_cinza = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_hsv   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    saturacao   = img_hsv[:, :, 1].astype(np.float32)
    brilho      = img_hsv[:, :, 2].astype(np.float32)
    limiar_s    = max(6.0, limiar_cor * 0.28)
    mascara_cor = ((saturacao > limiar_s) | (brilho < 242)).astype(np.uint8)

    gx  = cv2.Sobel(img_cinza, cv2.CV_64F, 1, 0, ksize=3)
    gy  = cv2.Sobel(img_cinza, cv2.CV_64F, 0, 1, ksize=3)
    mag = cv2.GaussianBlur(np.sqrt(gx ** 2 + gy ** 2), (15, 15), 0)
    m, s = np.mean(mag), np.std(mag)
    mascara_rel = (mag > (m + limiar_relevo * s / 100)).astype(np.uint8)

    k  = np.ones((15, 15), np.float32) / 225
    ml = cv2.filter2D(img_cinza.astype(np.float32), -1, k)
    ql = cv2.filter2D((img_cinza.astype(np.float32)) ** 2, -1, k)
    var = np.clip(ql - ml ** 2, 0, None)
    m, s = np.mean(var), np.std(var)
    mascara_tex = (var > (m + limiar_textura * s / 100)).astype(np.uint8)

    soma = mascara_cor + mascara_rel + mascara_tex
    return (soma >= 1).astype(np.uint8), mascara_cor, mascara_rel, mascara_tex


def _detectar_colorida(
    img_bgr: np.ndarray,
    sensibilidade: float,
) -> tuple:
    h, w = img_bgr.shape[:2]
    raio = max(min(h, w) // 20, 21)
    raio = raio | 1

    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

    sigma    = raio / 3.0
    base_lab = cv2.GaussianBlur(img_lab, (raio, raio), sigma)
    base_hsv = cv2.GaussianBlur(img_hsv, (raio, raio), sigma)

    desvio_local = np.sqrt(np.sum((img_lab - base_lab) ** 2, axis=2))
    p5  = np.percentile(desvio_local, 5)
    p95 = np.percentile(desvio_local, 95)
    dev_norm = np.clip((desvio_local - p5) / (p95 - p5 + 1e-6), 0, 1)
    mascara_local = (dev_norm > (0.35 / sensibilidade)).astype(np.uint8)

    faixa_cafe = (
        (img_hsv[:, :, 0] >= 8)  & (img_hsv[:, :, 0] <= 42) &
        (img_hsv[:, :, 1] >= 35) &
        (img_hsv[:, :, 2] >= 40) & (img_hsv[:, :, 2] <= 235)
    )
    h_diff = np.abs(img_hsv[:, :, 0] - base_hsv[:, :, 0])
    h_diff = np.minimum(h_diff, 180 - h_diff)
    s_diff = np.abs(img_hsv[:, :, 1].astype(float) - base_hsv[:, :, 1].astype(float))
    mascara_cafe = (faixa_cafe & ((h_diff > 18) | (s_diff > 35))).astype(np.uint8)

    b_diff = img_lab[:, :, 2] - base_lab[:, :, 2]
    p88    = np.percentile(b_diff, 88)
    std_b  = np.std(b_diff)
    mascara_b = (b_diff > (p88 + std_b * 1.5 / sensibilidade)).astype(np.uint8)

    l_diff   = img_lab[:, :, 0] - base_lab[:, :, 0]
    sat_diff = img_hsv[:, :, 1].astype(np.float32) - base_hsv[:, :, 1]
    p85_l    = np.percentile(l_diff, 85)
    std_l    = np.std(l_diff)
    mascara_clara = (
        (l_diff > (p85_l + std_l * 1.5 / sensibilidade)) &
        (sat_diff < -10)
    ).astype(np.uint8)

    mascara = (mascara_local & (mascara_cafe | mascara_b | mascara_clara)).astype(np.uint8)

    return mascara, mascara_local, mascara_cafe, mascara_b, mascara_clara


def detectar_manchas(
    caminho_imagem: str,
    caminho_saida: str    = "saida/01_deteccao.png",
    limiar_cor: float     = 35.0,
    limiar_relevo: float  = 15.0,
    limiar_textura: float = 8.0,
    tamanho_minimo: int   = 100,
    sensibilidade: float  = 1.0,
    tipo_imagem: str      = "auto",
) -> dict:
    print("\n[ETAPA 1] Iniciando DETECCAO DE MANCHAS...")
    img_bgr = _carregar_imagem(caminho_imagem)
    h, w    = img_bgr.shape[:2]

    if tipo_imagem == "auto":
        tipo_imagem = _detectar_tipo_imagem(img_bgr)

    if tipo_imagem == "branca":
        mascara_bruta, mc, mr, mt = _detectar_fundo_branco(
            img_bgr, limiar_cor, limiar_relevo, limiar_textura
        )
        nomes_metodos = ["Saturacao HSV", "Relevo", "Textura"]
        mascaras_sub  = [mc, mr, mt]
    else:
        mascara_bruta, ml, mcafe, mb, mclara = _detectar_colorida(img_bgr, sensibilidade)
        nomes_metodos = ["Desvio local (LAB)", "Cor cafe (HSV)", "Canal b* (LAB)", "Mancha clara/translucida"]
        mascaras_sub  = [ml, mcafe, mb, mclara]

    mascara_limpa = _limpar_mascara(mascara_bruta, tamanho_minimo)

    vis = img_bgr.copy()
    contornos, _ = cv2.findContours(mascara_limpa, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contornos, -1, (0, 0, 255), 2)
    overlay = vis.copy()
    overlay[mascara_limpa == 1] = (0, 80, 255)
    vis = cv2.addWeighted(overlay, 0.35, vis, 0.65, 0)
    _salvar_imagem(vis, caminho_saida)

    regioes  = measure.regionprops(measure.label(mascara_limpa))
    total_px = int(mascara_limpa.sum())
    pct      = round(total_px / (h * w) * 100, 2)

    relatorio = {
        "Tipo de imagem":              tipo_imagem,
        "Total de manchas detectadas": len(regioes),
        "Pixels marcados":             f"{total_px:,}",
        "Cobertura da imagem":         f"{pct}%",
        "Maior mancha (px)":           max((r.area for r in regioes), default=0),
        **{f"Metodo {n}":              int(m.sum()) for n, m in zip(nomes_metodos, mascaras_sub)},
        "Imagem salva em":             caminho_saida,
    }
    _imprimir_relatorio("RELATORIO - DETECCAO DE MANCHAS", relatorio)

    return {
        "imagem_original":   img_bgr,
        "mascara":           mascara_limpa,
        "regioes":           regioes,
        "tipo_imagem":       tipo_imagem,
        "relatorio":         relatorio,
        "caminho_resultado": caminho_saida,
    }


def separar_manchas(
    resultado_deteccao: dict,
    caminho_saida_frente: str = "saida/02_frente.png",
    caminho_saida_fundo: str  = "saida/02_fundo_manchas.png",
    margem_pixels: int        = 4,
) -> dict:
    print("\n[ETAPA 2] Iniciando SEPARACAO DAS MANCHAS...")

    img     = resultado_deteccao["imagem_original"]
    mascara = resultado_deteccao["mascara"].copy()

    kernel           = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (margem_pixels * 2 + 1,) * 2)
    mascara_dilatada = cv2.dilate(mascara, kernel, iterations=1)

    img_rgba          = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    img_rgba[:, :, 3] = np.where(mascara_dilatada == 1, 0, 255).astype(np.uint8)
    _salvar_imagem(img_rgba, caminho_saida_frente)

    fundo = np.zeros_like(img)
    fundo[mascara_dilatada == 1] = img[mascara_dilatada == 1]
    _salvar_imagem(fundo, caminho_saida_fundo)

    bboxes   = [r.bbox for r in resultado_deteccao["regioes"]]
    total_px = int(mascara_dilatada.sum())

    relatorio = {
        "Manchas separadas":        len(bboxes),
        "Pixels no 2o plano":       f"{total_px:,}",
        "Margem aplicada (px)":     margem_pixels,
        "Frente (PNG com alpha)":   caminho_saida_frente,
        "Fundo (manchas isoladas)": caminho_saida_fundo,
    }
    _imprimir_relatorio("RELATORIO - SEPARACAO DAS MANCHAS", relatorio)

    return {
        "imagem_original":  img,
        "mascara":          mascara,
        "mascara_dilatada": mascara_dilatada,
        "imagem_frente":    img_rgba,
        "imagem_fundo":     fundo,
        "bboxes":           bboxes,
        "tipo_imagem":      resultado_deteccao.get("tipo_imagem", "branca"),
        "relatorio":        relatorio,
        "caminho_frente":   caminho_saida_frente,
        "caminho_fundo":    caminho_saida_fundo,
    }


def _corrigir_mancha_colorida(img: np.ndarray, mascara: np.ndarray) -> np.ndarray:
    img_f      = img.astype(np.float32)
    mascara_u8 = mascara.astype(np.uint8)
    h, w       = img.shape[:2]

    px_fora = img_f.reshape(-1, 3)[mascara.flatten() == 0]
    if px_fora.size == 0:
        return img.copy()

    lum        = px_fora.mean(axis=1)
    limiar_lum = np.percentile(lum, 70)
    px_papel   = px_fora[lum >= limiar_lum]
    if px_papel.size == 0:
        px_papel = px_fora

    papel_cor = np.median(px_papel, axis=0)
    papel_cor = np.nan_to_num(papel_cor, nan=255.0)
    papel_cor = np.maximum(papel_cor, 1.0)
    print(f"    Cor do papel (BGR): {papel_cor.astype(int).tolist()}")

    contornos, _ = cv2.findContours(mascara_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    resultado = img.copy()

    for c in contornos:
        x, y, cw, ch = cv2.boundingRect(c)
        menor_dim = max(min(cw, ch), 1)

        k = int(menor_dim * 0.5) | 1
        k = max(k, 9)
        k = min(k, 81)
        k = min(k, (min(h, w) - 1) | 1)
        if k < 3:
            continue

        sigma = k / 4.0

        pad = k
        x0, y0 = max(x - pad, 0), max(y - pad, 0)
        x1, y1 = min(x + cw + pad, w), min(y + ch + pad, h)

        regiao = img_f[y0:y1, x0:x1]
        bg_regiao = cv2.GaussianBlur(regiao, (k, k), sigma)
        bg_regiao = np.maximum(bg_regiao, 1.0)

        fator = np.clip(papel_cor / bg_regiao, 0.5, 5.0)
        corr  = np.clip(regiao * fator, 0, papel_cor * 1.05)
        corr  = np.nan_to_num(corr, nan=0.0).astype(np.uint8)

        mascara_c = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mascara_c, [c], -1, 1, cv2.FILLED)
        mascara_regiao = mascara_c[y0:y1, x0:x1].astype(bool)

        recorte_resultado = resultado[y0:y1, x0:x1]
        recorte_resultado[mascara_regiao] = corr[mascara_regiao]

    kernel_b = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    borda    = cv2.dilate(mascara_u8, kernel_b) - mascara_u8
    if borda.sum() > 0:
        resultado = cv2.inpaint(resultado, borda, 2, cv2.INPAINT_TELEA)

    return resultado


def reconstruir_imagem(
    resultado_separacao: dict,
    caminho_saida: str      = "saida/03_reconstruida.png",
    metodo_inpainting: str  = "telea",
    raio_inpainting: int    = 5,
    ajuste_brilho: float    = 1.0,
    ajuste_contraste: float = 1.0,
) -> dict:
    print("\n[ETAPA 3] Iniciando RECONSTRUCAO DA IMAGEM...")

    img     = resultado_separacao["imagem_original"]
    mascara = resultado_separacao["mascara_dilatada"]
    tipo    = resultado_separacao.get("tipo_imagem", "branca")

    flag = cv2.INPAINT_TELEA if metodo_inpainting.lower() == "telea" else cv2.INPAINT_NS

    area_mancha  = int(mascara.sum())
    raio_efetivo = max(raio_inpainting, min(int(np.sqrt(area_mancha) // 6), 5))

    if tipo == "branca":
        pixels_fundo = img[mascara == 0].reshape(-1, 3).astype(np.float32)
        cor_fundo    = np.median(pixels_fundo, axis=0).astype(np.uint8)
        print(f"  Cor de fundo estimada: BGR{tuple(cor_fundo.tolist())}")

        img_reconstruida = img.copy()
        img_reconstruida[mascara == 1] = cor_fundo

        kernel_borda  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mascara_borda = (
            cv2.dilate(mascara.astype(np.uint8), kernel_borda) - mascara.astype(np.uint8)
        )
        if mascara_borda.sum() > 0:
            img_reconstruida = cv2.inpaint(img_reconstruida, mascara_borda, 3, flag)

    else:
        print(f"  Modo colorido — correcao de cor (preserva texto), "
              f"area da mancha: {area_mancha:,}px")
        img_reconstruida = _corrigir_mancha_colorida(img, mascara)

    if ajuste_brilho != 1.0 or ajuste_contraste != 1.0:
        reg = img_reconstruida[mascara == 1].astype(np.float32)
        reg = np.clip(reg * ajuste_contraste * ajuste_brilho, 0, 255)
        img_reconstruida[mascara == 1] = reg.astype(np.uint8)

    img_final = img.copy()
    img_final[mascara.astype(bool)] = img_reconstruida[mascara.astype(bool)]

    _salvar_imagem(img_final, caminho_saida)

    diff       = cv2.absdiff(img, img_final)
    erro_medio = float(np.mean(diff[mascara == 0]))

    relatorio = {
        "Tipo de imagem":            tipo,
        "Metodo de inpainting":      metodo_inpainting.upper(),
        "Raio de inpainting (px)":   raio_efetivo,
        "Ajuste de brilho":          f"x{ajuste_brilho}",
        "Ajuste de contraste":       f"x{ajuste_contraste}",
        "Erro medio (fora manchas)": f"{erro_medio:.4f}",
        "Resultado salvo em":        caminho_saida,
    }
    _imprimir_relatorio("RELATORIO - RECONSTRUCAO DA IMAGEM", relatorio)

    return {
        "imagem_reconstruida": img_final,
        "relatorio":           relatorio,
        "caminho_resultado":   caminho_saida,
    }


def remover_manchas(
    resultado_separacao: dict,
    resultado_reconstrucao: dict,
    caminho_saida_final: str = "saida/04_imagem_final.png",
    caminho_saida_diff: str  = "saida/04_diferenca.png",
) -> dict:
    print("\n[ETAPA 4] Iniciando REMOCAO DAS MANCHAS...")

    img_original     = resultado_separacao["imagem_original"]
    img_reconstruida = resultado_reconstrucao["imagem_reconstruida"]
    mascara          = resultado_separacao["mascara_dilatada"]

    img_final = img_original.copy()
    img_final[mascara == 1] = img_reconstruida[mascara == 1]

    diff          = cv2.absdiff(img_original, img_final).astype(np.float32)
    diff_amp      = np.clip(diff * 10, 0, 255).astype(np.uint8)
    diff_colorida = cv2.applyColorMap(diff_amp, cv2.COLORMAP_JET)

    _salvar_imagem(img_final,     caminho_saida_final)
    _salvar_imagem(diff_colorida, caminho_saida_diff)

    h, w         = img_original.shape[:2]
    pixels_mod   = int(mascara.sum())
    pct_mod      = round(pixels_mod / (h * w) * 100, 2)
    diff_manchas = float(np.mean(cv2.absdiff(img_original, img_final)[mascara == 1]))
    diff_fora    = float(np.mean(cv2.absdiff(img_original, img_final)[mascara == 0]))

    relatorio = {
        "Pixels modificados":         f"{pixels_mod:,} ({pct_mod}%)",
        "Diferenca media (manchas)":  f"{diff_manchas:.2f}",
        "Diferenca media (resto)":    f"{diff_fora:.4f}",
        "Imagem final salva em":      caminho_saida_final,
        "Mapa de diferenca salvo em": caminho_saida_diff,
        "Status":                     "Processamento concluido com sucesso",
    }
    _imprimir_relatorio("RELATORIO - REMOCAO FINAL DAS MANCHAS", relatorio)

    return {
        "imagem_final":      img_final,
        "diff_colorida":     diff_colorida,
        "relatorio":         relatorio,
        "caminho_resultado": caminho_saida_final,
        "caminho_diff":      caminho_saida_diff,
    }


def criar_comparacao(
    img_original: np.ndarray,
    img_final: np.ndarray,
    caminho_saida: str = "saida/05_antes_depois.png",
    largura_maxima: int = 1600,
) -> str:
    h, w = img_original.shape[:2]

    escala = min(1.0, largura_maxima / (w * 2))
    if escala < 1.0:
        nh = max(int(h * escala), 1)
        nw = max(int(w * escala), 1)
        img_a = cv2.resize(img_original, (nw, nh), interpolation=cv2.INTER_AREA)
        img_d = cv2.resize(img_final,    (nw, nh), interpolation=cv2.INTER_AREA)
    else:
        img_a = img_original.copy()
        img_d = img_final.copy()
        nh, nw = h, w

    divisor    = np.full((nh, 4, 3), 255, dtype=np.uint8)
    comparacao = np.concatenate([img_a, divisor, img_d], axis=1)

    fonte      = cv2.FONT_HERSHEY_SIMPLEX
    escala_txt = max(nw / 800, 0.7)
    espessura  = max(int(escala_txt * 2), 1)
    padding    = int(10 * escala_txt)

    for texto, x_base in [("ANTES", 0), ("DEPOIS", nw + 4)]:
        (tw, th), _ = cv2.getTextSize(texto, fonte, escala_txt, espessura)
        x1 = x_base + padding
        y1 = padding
        x2 = x1 + tw + padding
        y2 = y1 + th + padding

        roi    = comparacao[y1:y2, x1:x2]
        fundo  = np.zeros_like(roi)
        comparacao[y1:y2, x1:x2] = cv2.addWeighted(roi, 0.4, fundo, 0.6, 0)

        cv2.putText(comparacao, texto,
                    (x1 + padding // 2, y2 - padding // 2),
                    fonte, escala_txt, (255, 255, 255), espessura, cv2.LINE_AA)

    _salvar_imagem(comparacao, caminho_saida)
    print(f"  Comparacao antes/depois salva em: {caminho_saida}")
    return caminho_saida


def pipeline_removedor(
    caminho_imagem: str,
    pasta_saida: str            = "saida",
    limiar_cor: float           = 35.0,
    limiar_relevo: float        = 15.0,
    limiar_textura: float       = 8.0,
    tamanho_minimo: int         = 100,
    margem_separacao: int       = 4,
    metodo_inpainting: str      = "telea",
    raio_inpainting: int        = 5,
    ajuste_brilho: float        = 1.0,
    ajuste_contraste: float     = 1.0,
    sensibilidade: float        = 1.0,
    tipo_imagem: str            = "auto",
    salvar_relatorio_json: bool = True,
) -> dict:
    os.makedirs(pasta_saida, exist_ok=True)
    print("\n" + "=" * 60)
    print("   REMOVEDOR DE MANCHAS v2.4 - Pipeline Completo")
    print("=" * 60)
    print(f"   Imagem de entrada : {caminho_imagem}")
    print(f"   Pasta de saida    : {pasta_saida}")
    print(f"   Modo              : {tipo_imagem}  |  Sensibilidade: {sensibilidade}")
    print("=" * 60)

    r1 = detectar_manchas(
        caminho_imagem,
        caminho_saida  = f"{pasta_saida}/01_deteccao.png",
        limiar_cor     = limiar_cor,
        limiar_relevo  = limiar_relevo,
        limiar_textura = limiar_textura,
        tamanho_minimo = tamanho_minimo,
        sensibilidade  = sensibilidade,
        tipo_imagem    = tipo_imagem,
    )

    r2 = separar_manchas(
        r1,
        caminho_saida_frente = f"{pasta_saida}/02_frente.png",
        caminho_saida_fundo  = f"{pasta_saida}/02_fundo_manchas.png",
        margem_pixels        = margem_separacao,
    )

    r3 = reconstruir_imagem(
        r2,
        caminho_saida     = f"{pasta_saida}/03_reconstruida.png",
        metodo_inpainting = metodo_inpainting,
        raio_inpainting   = raio_inpainting,
        ajuste_brilho     = ajuste_brilho,
        ajuste_contraste  = ajuste_contraste,
    )

    r4 = remover_manchas(
        r2, r3,
        caminho_saida_final = f"{pasta_saida}/04_imagem_final.png",
        caminho_saida_diff  = f"{pasta_saida}/04_diferenca.png",
    )

    print("\n[ETAPA 5] Gerando comparacao ANTES / DEPOIS...")
    caminho_comp = criar_comparacao(
        img_original  = r1["imagem_original"],
        img_final     = r4["imagem_final"],
        caminho_saida = f"{pasta_saida}/05_antes_depois.png",
    )

    print("\n" + "=" * 60)
    print("   PIPELINE CONCLUIDO - Arquivos gerados:")
    print("=" * 60)
    for linha in [
        f"  {pasta_saida}/01_deteccao.png      <- manchas destacadas",
        f"  {pasta_saida}/02_frente.png         <- imagem sem manchas (alpha)",
        f"  {pasta_saida}/02_fundo_manchas.png  <- manchas isoladas",
        f"  {pasta_saida}/03_reconstruida.png   <- imagem reparada",
        f"  {pasta_saida}/04_imagem_final.png   <- resultado final limpo",
        f"  {pasta_saida}/04_diferenca.png      <- mapa de diferenca",
        f"  {pasta_saida}/05_antes_depois.png   <- comparacao ANTES/DEPOIS  <--",
    ]:
        print(linha)
    print("=" * 60 + "\n")

    resultado = {
        "etapa_1_deteccao":      r1["relatorio"],
        "etapa_2_separacao":     r2["relatorio"],
        "etapa_3_reconstrucao":  r3["relatorio"],
        "etapa_4_remocao":       r4["relatorio"],
    }

    if salvar_relatorio_json:
        caminho_json = f"{pasta_saida}/relatorio_completo.json"

        def serializar(obj):
            if isinstance(obj, np.integer):  return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            return str(obj)

        with open(caminho_json, "w", encoding="utf-8") as f:
            json.dump(resultado, f, ensure_ascii=False, indent=2, default=serializar)
        print(f"  Relatorio JSON salvo em: {caminho_json}\n")

    return resultado


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("\n  Uso: python removedor_manchas.py <imagem> [pasta_saida] [sensibilidade] [tipo]")
        print("  Exemplos:")
        print("    python removedor_manchas.py foto.jpg resultados")
        print("    python removedor_manchas.py foto.jpg resultados 0.5 colorida")
        print("    python removedor_manchas.py doc.png  resultados 1.0 branca\n")
        sys.exit(1)

    caminho = sys.argv[1]
    saida   = sys.argv[2] if len(sys.argv) > 2 else "saida"
    sensib  = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    tipo    = sys.argv[4] if len(sys.argv) > 4 else "auto"

    pipeline_removedor(
        caminho_imagem        = caminho,
        pasta_saida           = saida,
        limiar_cor            = 35.0,
        limiar_relevo         = 15.0,
        limiar_textura        = 8.0,
        tamanho_minimo        = 100,
        margem_separacao      = 4,
        metodo_inpainting     = "telea",
        raio_inpainting       = 5,
        ajuste_brilho         = 1.0,
        ajuste_contraste      = 1.0,
        sensibilidade         = sensib,
        tipo_imagem           = tipo,
        salvar_relatorio_json = True,
    )
