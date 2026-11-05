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
    tikz_head = to_head(str(upstream)).replace(
        r"\usepackage{import}",
        "\\usepackage{import}\n"
        "\\usepackage{xeCJK}\n"
        "\\setCJKmainfont{Songti SC}",
    )
    tikz_begin = to_begin().replace(
        r"\begin{document}",
        "\\begin{document}\n\\pagecolor{white}\n\\color{black}",
    )
    arch = [
        tikz_head,
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
            caption=r"{\small 三路相机\\$[B,3,224,224]$\\环境 + 腕部}",
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
            caption=r"{\small DINOv2\\ViT-S/14\\$[B,256,384]$\\局部视觉编码}",
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
            caption=r"{\small Resampler\\$[B,48,768]$\\固定预算}",
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
            caption=r"{\small 视觉前缀\\$[B,144,768]$\\3 $\times$ 48}",
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
            caption=r"{\small 融合前缀\\$[B,177,768]$\\144 + 32 + 1}",
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
            caption=r"{\small T5 语言\\$[B,32,768]$\\任务语义}",
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
            caption=r"{\small 状态 token\\$[B,1,768]$\\本体感知}",
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
            caption=r"{\small 1 层 DiT\\噪声 $[B,10,12]$ + $\tau$\\条件速度场}",
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
            caption=r"{\small 速度头\\$[B,10,12]$\\未来 10 步}",
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
            caption=r"{\small 任务辅助\\$[B,61]$\\仅训练}",
        ),
        to_connection("rgb", "dino"),
        to_connection("dino", "resampler"),
        to_connection("resampler", "visual"),
        to_connection("visual", "concat"),
        to_connection("concat", "dit"),
        to_connection("dit", "velocity"),
        r"""
\path (rgb-east) -- node[above=3pt, font=\scriptsize] {编码} (dino-west);
\path (dino-east) -- node[above=3pt, font=\scriptsize] {压缩} (resampler-west);
\path (resampler-east) -- node[above=3pt, font=\scriptsize] {3 $\times$ 48} (visual-west);
\path (visual-east) -- node[above=3pt, font=\scriptsize] {拼接} (concat-west);
\path (concat-east) -- node[above=3pt, font=\scriptsize] {条件化} (dit-west);
\path (dit-east) -- node[above=3pt, font=\scriptsize] {12 维速度} (velocity-west);
\draw [connection] (text-southeast) -- node {\midarrow} (concat-northeast);
\draw [connection] (state-northeast) -- node {\midarrow} (concat-southeast);
\draw [connection, densely dashed] (dit-north) -- node {\midarrow} (task-south);
\node[
  align=left,
  anchor=north west,
  text width=19cm,
  font=\normalsize
] at (-2,-11,0) {
  {\large\bfseries CoRP-RoboCasa：从多视角观察到动作生成}\\[1mm]
  \textbf{(1) 感知与压缩：} 三路相机经 DINOv2 得到每视角 256 个 patch，
  再由 Resampler 压缩为每视角 48 个视觉 token；三视角合计 144 个。\\
  \textbf{(2) 统一接口：} 144 个视觉 token + 32 个语言 token + 1 个状态 token
  $=177$ 个 token，每个 token 宽 768 维。\\
  \textbf{(3) 动作生成：} 噪声动作 \texttt{[B,10,12]} 与时间 $\tau$ 输入 DiT，
  输出 12 维速度场；推理时做 10 步 Euler 积分，得到未来 10 步动作。\\
  \textbf{(4) 训练辅助：} 任务辅助头从视觉前缀预测 61 类任务，只在训练时使用；
  虚线不进入推理动作路径。\\
  \textbf{高维理解：} 视觉编码负责“看见”，融合前缀是共享表示接口，
  DiT 负责“生成动作”。实线是推理路径，虚线是训练辅助路径。
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
