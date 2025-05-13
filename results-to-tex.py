import json
import os
import pandas as pd

# Read data from JSON files
dir_path = "./bench_res_pbbs/"
files = [f for f in os.listdir(dir_path) if f.endswith(".json")]

# Define target execs and base exec
base_exec = "EIGEN_MAILBOX_JE"
target_execs = {
    "EIGEN_MAILBOX_JE": True,
    "EIGEN_PAR_DO_STEAL": True,
    "EIGEN_FOLDED_SHARING": True,
    "EIGEN_SHARE_PROXY": True,
    "EIGEN_TIMESPAN_GRAINSIZE": True,
    "OMP_STATIC": True,
    "TBB_AFFINITY": True,
    "TBB_AUTO": True,
    "TBB_SIMPLE": True,
    "TBB_CONST_AFFINITY": True,
}

res = {}
for file in files:
    mode = file.split(".")[0]
    with open(os.path.join(dir_path, file), "r") as f:
        data = json.load(f)
    if mode in target_execs:
        res[mode] = data

# Transform into DataFrame
by_bench = {}
for exec_name, benchmarks in res.items():
    for bench_name, results in benchmarks.items():
        if bench_name not in by_bench:
            by_bench[bench_name] = {}
        by_bench[bench_name][exec_name] = results["total"]

df = pd.DataFrame.from_dict(by_bench, orient="index").fillna(0)

# Generate Heatmap LaTeX
norm_df = df.div(df.max(axis=1), axis=0)

heatmap_rows = []
for bench in norm_df.index:
    row = [bench.replace("/", "-").replace("_", "-")]
    for exec in norm_df.columns:
        norm_val = norm_df.loc[bench, exec]
        intensity = int((1 - norm_val) * 100)
        intensity = max(0, min(100, intensity))
        row.append(f"\\cellcolor{{blue!{int(100 - intensity)}}} {norm_val:.2f}")
    heatmap_rows.append(" & ".join(row) + " \\\\")

heatmap_tex = r"""
\documentclass{article}
\usepackage[table]{xcolor}
\usepackage{adjustbox}
\begin{document}
\begin{table}[ht]
\centering
\adjustbox{max width=\textwidth}{
\begin{tabular}{chosuka}
\rowcolor{white}
\textbf{Benchmark} & kalllketgondon \\
xyugavnomocha
\end{tabular}
}
\caption{Normalized benchmark results (per-row maximum)}
\label{tab:heatmap}
\end{table}
\end{document}
""".replace("xyugavnomocha", "\n".join(heatmap_rows)).replace("chosuka", 'c' * (len(norm_df.columns) + 1)).replace("kalllketgondon", ' & '.join([s.replace('_', '-') for s in norm_df.columns]))

with open("result_heatmap.tex", "w") as f:
    f.write(heatmap_tex)
