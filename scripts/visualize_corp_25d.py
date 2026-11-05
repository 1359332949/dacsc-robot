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
    arch = [
        to_head(str(upstream)),
        to_cor(),
        to_begin(),
        to_Conv(
            "rgb",
            s_filer=224,
            n_filer=3,
            offset="(0,0,0)",
            to="(0,0,0)",
            width=3.5,
            height=18,
            depth=18,
            caption=r"3 views\\224 x 224",
        ),
        to_Conv(
            "dino",
            s_filer=256,
            n_filer=384,
            offset="(1.6,0,0)",
            to="(rgb-east)",
            width=4.0,
            height=16,
            depth=16,
            caption=r"DINOv2\\ViT-S/14",
        ),
        to_Conv(
            "resampler",
            s_filer=48,
            n_filer=768,
            offset="(1.6,0,0)",
            to="(dino-east)",
            width=4.0,
            height=14,
            depth=14,
            caption=r"Resampler\\48 / view",
        ),
        to_Conv(
            "visual",
            s_filer=144,
            n_filer=768,
            offset="(1.6,0,0)",
            to="(resampler-east)",
            width=4.0,
            height=14,
            depth=14,
            caption=r"Visual prefix\\144 x 768",
        ),
        to_Conv(
            "concat",
            s_filer=177,
            n_filer=768,
            offset="(1.8,0,0)",
            to="(visual-east)",
            width=4.5,
            height=16,
            depth=16,
            caption=r"Fusion prefix\\177 x 768",
        ),
        to_Conv(
            "text",
            s_filer=32,
            n_filer=768,
            offset="(0,5,0)",
            to="(concat-north)",
            width=4.0,
            height=6,
            depth=6,
            caption=r"T5\\32 x 768",
        ),
        to_Conv(
            "state",
            s_filer=1,
            n_filer=768,
            offset="(0,-5,0)",
            to="(concat-south)",
            width=4.0,
            height=5,
            depth=5,
            caption=r"State\\1 x 768",
        ),
        to_Conv(
            "dit",
            s_filer=10,
            n_filer=768,
            offset="(1.8,0,0)",
            to="(concat-east)",
            width=4.5,
            height=11,
            depth=11,
            caption=r"1-layer DiT\\10 x 768",
        ),
        to_Conv(
            "velocity",
            s_filer=10,
            n_filer=12,
            offset="(1.6,0,0)",
            to="(dit-east)",
            width=4.0,
            height=8,
            depth=8,
            caption=r"Velocity\\10 x 12",
        ),
        to_Conv(
            "task",
            s_filer=1,
            n_filer=61,
            offset="(0,4.5,0)",
            to="(dit-north)",
            width=4.0,
            height=4,
            depth=4,
            caption=r"Task aux\\61-way",
        ),
        to_connection("rgb", "dino"),
        to_connection("dino", "resampler"),
        to_connection("resampler", "visual"),
        to_connection("visual", "concat"),
        to_connection("concat", "dit"),
        to_connection("dit", "velocity"),
        r"""
\draw [connection] (text-south) -- node {\midarrow} (concat-north);
\draw [connection] (state-north) -- node {\midarrow} (concat-south);
\draw [connection, densely dashed] (dit-north) -- node {\midarrow} (task-south);
\node[font=\small, anchor=west] at (-2.5,-11,0)
  {Dashed: training-only auxiliary branch};
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
