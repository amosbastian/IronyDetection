import os

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def create_table(filename):
    table = """
\\begin{center}
\\begin{tabular}{ |c|c|c|c| }
\hline
\multicolumn{4}{|c|}{filename}\\\\
\hline
1-gram & Relative Frequency & 1-gram & Relative Frequency\\\\
\hline"""
    with open(f"{DIR_PATH}/output/{filename}") as f:
        head = [next(f) for x in range(21)]
        ngrams = [(x.split("\t")[-1].replace("\n", ""), float(x.split("\t")[-2]))
                  for x in head[1:]]

    x = ngrams[:10]
    y = ngrams[10:]
    for i, j in zip(x, y):
        table += (f"\n\\textit{{{i[0]}}} & {i[1]:.2f} & \\textit{{{j[0]}}} & {j[1]:.2f}\\\\ \n"
                  "\hline")
    table += "\n\end{tabular}\n\end{center}"
    print(table)
if __name__ == "__main__":
    create_table("relative_1-gram_frequency.txt")
