"""Render a CoRP 2.5D architecture diagram with PlotNeuralNet's TikZ macros."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plotneuralnet",
        type=Path,
        default=ROOT / ".cache/PlotNeuralNet-src",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "audit/corp_model_visualization_25d",
    )
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--png-dpi", type=int, default=180)
    args = parser.parse_args()
    if args.png_dpi <= 0:
        parser.error("--png-dpi must be positive")
    upstream = args.plotneuralnet.expanduser().resolve()
    tikzeng = upstream / "pycore" / "tikzeng.py"
    if not tikzeng.is_file():
        raise SystemExit(
            "PlotNeuralNet not found; clone HarisIqbal88/PlotNeuralNet first"
        )
    sys.path.insert(0, str(upstream))
    from pycore.tikzeng import (  # type: ignore[import-not-found]
        to_begin,
        to_Conv,
        to_connection,
        to_cor,
        to_end,
        to_generate,
        to_head,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    tikz_begin = to_begin().replace(
        r"\begin{document}",
        "\\begin{document}\n\\pagecolor{white}\n\\color{black}",
    )
    arch = [
        to_head(str(upstream)),
        to_cor(),
        tikz_begin,
        to_Conv(
            "rgb",
            s_filer="",
            n_filer="",
            offset="(0,0,0)",
            to="(0,0,0)",
            width=3.5,
            height=18,
            depth=18,
            caption=r"{3 views\\$[B,3,224,224]$}",
        ),
        to_Conv(
            "dino",
            s_filer="",
            n_filer="",
            offset="(2.4,0,0)",
            to="(rgb-east)",
            width=4.0,
            height=16,
            depth=16,
            caption=r"{DINOv2 ViT-S/14\\$[B,256,384]$ / view}",
        ),
        to_Conv(
            "resampler",
            s_filer="",
            n_filer="",
            offset="(2.4,0,0)",
            to="(dino-east)",
            width=4.0,
            height=14,
            depth=14,
            caption=r"{Resampler\\$[B,48,768]$ / view}",
        ),
        to_Conv(
            "visual",
            s_filer="",
            n_filer="",
            offset="(2.4,0,0)",
            to="(resampler-east)",
            width=4.0,
            height=14,
            depth=14,
            caption=r"{Visual prefix\\$[B,144,768]$}",
        ),
        to_Conv(
            "concat",
            s_filer="",
            n_filer="",
            offset="(2.8,0,0)",
            to="(visual-east)",
            width=4.5,
            height=16,
            depth=16,
            caption=r"{Fusion prefix\\$[B,177,768]$\\144 + 32 + 1}",
        ),
        to_Conv(
            "text",
            s_filer="",
            n_filer="",
            offset="(1.8,5,0)",
            to="(concat-north)",
            width=4.0,
            height=6,
            depth=6,
            caption=r"{T5\\language\\$[B,32,768]$}",
        ),
        to_Conv(
            "state",
            s_filer="",
            n_filer="",
            offset="(1.8,-5,0)",
            to="(concat-south)",
            width=4.0,
            height=5,
            depth=5,
            caption=r"{State token\\$[B,1,768]$}",
        ),
        to_Conv(
            "dit",
            s_filer="",
            n_filer="",
            offset="(2.8,0,0)",
            to="(concat-east)",
            width=4.5,
            height=11,
            depth=11,
            caption=r"{1-layer DiT\\noisy $[B,10,12]$ + $\tau$}",
        ),
        to_Conv(
            "velocity",
            s_filer="",
            n_filer="",
            offset="(2.4,0,0)",
            to="(dit-east)",
            width=4.0,
            height=8,
            depth=8,
            caption=r"{Velocity head\\$[B,10,12]$}",
        ),
        to_Conv(
            "task",
            s_filer="",
            n_filer="",
            offset="(0,4.5,0)",
            to="(dit-north)",
            width=4.0,
            height=4,
            depth=4,
            caption=r"{Task aux\\$[B,61]$\\training only}",
        ),
        to_connection("rgb", "dino"),
        to_connection("dino", "resampler"),
        to_connection("resampler", "visual"),
        to_connection("visual", "concat"),
        to_connection("concat", "dit"),
        to_connection("dit", "velocity"),
        r"""
\path (rgb-east) -- node[above=3pt, font=\scriptsize] {encode} (dino-west);
\path (dino-east) -- node[above=3pt, font=\scriptsize] {compress} (resampler-west);
\path (resampler-east) -- node[above=3pt, font=\scriptsize] {3 x 48} (visual-west);
\path (visual-east) -- node[above=3pt, font=\scriptsize] {concat} (concat-west);
\path (concat-east) -- node[above=3pt, font=\scriptsize] {condition} (dit-west);
\path (dit-east) -- node[above=3pt, font=\scriptsize] {12-D velocity} (velocity-west);
\draw [connection] (text-southeast) -- node {\midarrow} (concat-northeast);
\draw [connection] (state-northeast) -- node {\midarrow} (concat-southeast);
\draw [connection, densely dashed] (dit-north) -- node {\midarrow} (task-south);
\node[
  align=left,
  anchor=north west,
  text width=18cm,
  font=\small
] at (-2,-11,0) {
  \textbf{Perception:} 3 views $\rightarrow$ patch tokens $\rightarrow$ fixed visual tokens;\\
  \textbf{Shared interface:} 144 visual + 32 language + 1 state $=177$ tokens;\\
  \textbf{Action:} 177-token prefix + noisy action and $\tau$ $\rightarrow$ velocity,
  10 Euler steps $\rightarrow$ \texttt{[B,10,12]}.\\
  \textbf{Solid:} inference path \quad \textbf{Dashed:} training-only task auxiliary.
};
""",
        to_end(),
    ]
    tex_path = args.out / "corp_25d.tex"
    to_generate(arch, str(tex_path))
    print(tex_path)

    if args.compile:
        tectonic = Path("/opt/homebrew/bin/tectonic")
        if not tectonic.is_file():
            raise SystemExit("tectonic is required for --compile")
        subprocess.run(
            [
                str(tectonic),
                "--outdir",
                str(args.out),
                str(tex_path),
            ],
            check=True,
        )
        pdf_path = args.out / "corp_25d.pdf"
        print(pdf_path)

        pdftoppm = shutil.which("pdftoppm")
        if pdftoppm is None:
            raise SystemExit("pdftoppm (Poppler) is required to render corp_25d.png")
        png_prefix = args.out / "corp_25d"
        subprocess.run(
            [
                pdftoppm,
                "-png",
                "-r",
                str(args.png_dpi),
                "-singlefile",
                str(pdf_path),
                str(png_prefix),
            ],
            check=True,
        )
        print(png_prefix.with_suffix(".png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
