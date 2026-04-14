"""
ExtractSub.py
Funções para extrair legendas embutidas de arquivos de vídeo (MKV, MP4, etc.)
usando ffprobe e ffmpeg (dependência já exigida pelo projeto).

Instalação:
    Nenhuma dependência extra — usa apenas o ffmpeg já requerido pelo GenLeg.py.
    Windows : https://ffmpeg.org/download.html  (adicione ao PATH)
    Ubuntu  : sudo apt install ffmpeg
    macOS   : brew install ffmpeg
"""

import json
import os
import shutil
import subprocess
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# Tipos auxiliares
# ──────────────────────────────────────────────────────────────────────────────

class LegendaInfo:
    """Representa uma faixa de legenda encontrada no arquivo de vídeo."""

    def __init__(self, index: int, codec: str, idioma: str, titulo: str, forcada: bool):
        self.index   = index    # índice global da stream (usado no ffmpeg)
        self.codec   = codec    # ass, subrip, mov_text, dvd_subtitle…
        self.idioma  = idioma   # código ISO 639-2 (ex: 'eng', 'por')
        self.titulo  = titulo   # título opcional da faixa
        self.forcada = forcada  # flag "forced" da faixa

    def __repr__(self) -> str:
        partes = [f"[{self.index}]", self.codec, self.idioma or "?"]
        if self.titulo:
            partes.append(f'"{self.titulo}"')
        if self.forcada:
            partes.append("(forced)")
        return " ".join(partes)


# ──────────────────────────────────────────────────────────────────────────────
# Validação de dependências
# ──────────────────────────────────────────────────────────────────────────────

def _verificar_ffmpeg() -> None:
    """Garante que ffmpeg e ffprobe estão disponíveis no PATH."""
    for ferramenta in ("ffmpeg", "ffprobe"):
        if not shutil.which(ferramenta):
            raise EnvironmentError(
                f"'{ferramenta}' não encontrado no PATH.\n"
                "  Windows : https://ffmpeg.org/download.html\n"
                "  Ubuntu  : sudo apt install ffmpeg\n"
                "  macOS   : brew install ffmpeg"
            )


def _verificar_arquivo(caminho: str) -> None:
    """Verifica se o arquivo de vídeo existe."""
    if not os.path.isfile(caminho):
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")


# ──────────────────────────────────────────────────────────────────────────────
# Listagem de faixas de legenda
# ──────────────────────────────────────────────────────────────────────────────

def listar_legendas(arquivo_video: str) -> list[LegendaInfo]:
    """
    Lista todas as faixas de legenda embutidas no arquivo de vídeo.

    Usa ffprobe para inspecionar o container e retorna uma lista de
    LegendaInfo com os metadados de cada faixa de legenda encontrada.

    Args:
        arquivo_video: Caminho do arquivo de vídeo (MKV, MP4, AVI…).

    Returns:
        Lista de LegendaInfo. Lista vazia se o vídeo não tiver legendas.

    Raises:
        FileNotFoundError: Se o arquivo de vídeo não existir.
        EnvironmentError:  Se ffprobe não estiver no PATH.
    """
    _verificar_ffmpeg()
    _verificar_arquivo(arquivo_video)

    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-select_streams", "s",   # apenas streams de legenda
        arquivo_video,
    ]

    resultado = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")

    if resultado.returncode != 0:
        raise RuntimeError(
            f"ffprobe falhou ao inspecionar '{arquivo_video}':\n{resultado.stderr}"
        )

    dados = json.loads(resultado.stdout or "{}")
    streams = dados.get("streams", [])

    legendas = []
    for stream in streams:
        tags    = stream.get("tags", {})
        forcada = stream.get("disposition", {}).get("forced", 0) == 1

        legendas.append(LegendaInfo(
            index   = stream.get("index", -1),
            codec   = stream.get("codec_name", "desconhecido"),
            idioma  = tags.get("language", ""),
            titulo  = tags.get("title", ""),
            forcada = forcada,
        ))

    return legendas


def exibir_legendas(arquivo_video: str) -> None:
    """
    Imprime na tela as faixas de legenda disponíveis no arquivo.

    Args:
        arquivo_video: Caminho do arquivo de vídeo.
    """
    legendas = listar_legendas(arquivo_video)

    if not legendas:
        print(f"Nenhuma legenda embutida encontrada em: {arquivo_video}")
        return

    print(f"Legendas embutidas em '{os.path.basename(arquivo_video)}':")
    print(f"  {'#':<4} {'Codec':<14} {'Idioma':<8} {'Forçada':<9} Título")
    print("  " + "-" * 60)
    for leg in legendas:
        forcada = "sim" if leg.forcada else "não"
        print(f"  {leg.index:<4} {leg.codec:<14} {leg.idioma:<8} {forcada:<9} {leg.titulo}")


# ──────────────────────────────────────────────────────────────────────────────
# Extração de legenda
# ──────────────────────────────────────────────────────────────────────────────

def extrair_legenda(
    arquivo_video: str,
    arquivo_saida: str | None = None,
    indice: int | None = None,
    idioma: str | None = None,
) -> str:
    """
    Extrai uma faixa de legenda embutida e salva como arquivo .srt.

    A faixa é selecionada por `indice` (índice global da stream no container)
    ou por `idioma` (código ISO 639-2, ex: 'eng', 'por'). Se nenhum for
    fornecido, extrai a primeira faixa de legenda encontrada.

    Codecs baseados em imagem (dvd_subtitle, hdmv_pgs_subtitle) não podem ser
    convertidos para SRT e levantam ValueError.

    Args:
        arquivo_video: Caminho do arquivo de vídeo (MKV, MP4, AVI…).
        arquivo_saida: Caminho do .srt de saída. Se None, usa o mesmo nome
                       do vídeo com sufixo do idioma (ex: filme.eng.srt).
        indice:        Índice global da stream (coluna '#' de listar_legendas).
        idioma:        Código ISO 639-2 da faixa desejada (ex: 'eng', 'por').

    Returns:
        Caminho absoluto do arquivo .srt gerado.

    Raises:
        FileNotFoundError: Se o arquivo de vídeo não existir.
        EnvironmentError:  Se ffmpeg/ffprobe não estiverem no PATH.
        ValueError:        Se nenhuma faixa válida for encontrada ou se o codec
                           for baseado em imagem (não suportado).
        RuntimeError:      Se o ffmpeg falhar durante a extração.
    """
    _verificar_ffmpeg()
    _verificar_arquivo(arquivo_video)

    legendas = listar_legendas(arquivo_video)

    if not legendas:
        raise ValueError(f"Nenhuma legenda embutida encontrada em: {arquivo_video}")

    # ── Seleciona a faixa ────────────────────────────────────────────────────
    faixa: LegendaInfo | None = None

    if indice is not None:
        correspondentes = [leg for leg in legendas if leg.index == indice]
        if not correspondentes:
            disponiveis = ", ".join(str(leg.index) for leg in legendas)
            raise ValueError(
                f"Índice {indice} não encontrado. Disponíveis: {disponiveis}"
            )
        faixa = correspondentes[0]

    elif idioma is not None:
        idioma_norm = idioma.lower().strip()
        correspondentes = [leg for leg in legendas if leg.idioma.lower() == idioma_norm]
        if not correspondentes:
            disponiveis = ", ".join(leg.idioma for leg in legendas if leg.idioma)
            raise ValueError(
                f"Idioma '{idioma}' não encontrado. Disponíveis: {disponiveis}"
            )
        faixa = correspondentes[0]

    else:
        faixa = legendas[0]
        print(f"Nenhuma faixa especificada. Usando a primeira: {faixa}")

    # ── Valida codec ─────────────────────────────────────────────────────────
    _CODECS_IMAGEM = {"dvd_subtitle", "dvdsub", "hdmv_pgs_subtitle", "pgssub", "xsub"}
    if faixa.codec.lower() in _CODECS_IMAGEM:
        raise ValueError(
            f"A faixa [{faixa.index}] usa o codec '{faixa.codec}' (baseado em imagem) "
            "e não pode ser convertida para SRT.\n"
            "Use OCR (ex: Subtitle Edit) para converter legendas baseadas em imagem."
        )

    # ── Monta caminho de saída ────────────────────────────────────────────────
    if arquivo_saida is None:
        sufixo_idioma = f".{faixa.idioma}" if faixa.idioma else ""
        arquivo_saida = str(Path(arquivo_video).with_suffix(f"{sufixo_idioma}.srt"))

    # ── Executa ffmpeg ────────────────────────────────────────────────────────
    # map 0:N seleciona a stream pelo índice global dentro do arquivo de entrada
    cmd = [
        "ffmpeg",
        "-y",                        # sobrescreve sem perguntar
        "-i", arquivo_video,
        "-map", f"0:{faixa.index}",  # seleciona a stream exata
        "-c:s", "srt",               # converte para SubRip
        arquivo_saida,
    ]

    print(f"Extraindo faixa [{faixa.index}] ({faixa.codec} / {faixa.idioma or '?'})...")

    resultado = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    if resultado.returncode != 0:
        raise RuntimeError(
            f"ffmpeg falhou ao extrair a legenda:\n{resultado.stderr}"
        )

    caminho_abs = os.path.abspath(arquivo_saida)
    print(f"Legenda salva em: {caminho_abs}")
    return caminho_abs
