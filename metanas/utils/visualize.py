""" Network architecture visualizer using graphviz """

""" 
From https://github.com/khanrc/pt.darts
which is licensed under MIT License,
cf. 3rd-party-licenses.txt in root directory.
"""

import sys

from graphviz import Digraph

from metanas.utils import genotypes as gt


def plot(genotype, file_path, caption=None):
    """ make DAG plot and save to file_path as .png """
    edge_attr = {"fontsize": "20", "fontname": "times"}
    node_attr = {
        "style": "filled",
        "shape": "rect",
        "align": "center",
        "fontsize": "20",
        "height": "0.5",
        "width": "0.5",
        "penwidth": "2",
        "fontname": "times",
    }
    g = Digraph(format="png", edge_attr=edge_attr, node_attr=node_attr, engine="dot")
    g.body.extend(["rankdir=LR"])

    # input nodes
    g.node("c_{k-2}", fillcolor="darkseagreen2")
    g.node("c_{k-1}", fillcolor="darkseagreen2")

    # intermediate nodes
    n_nodes = len(genotype)
    for i in range(n_nodes):
        g.node(str(i), fillcolor="lightblue")

    for i, edges in enumerate(genotype):
        for op, j in edges:
            if j == 0:
                u = "c_{k-2}"
            elif j == 1:
                u = "c_{k-1}"
            else:
                u = str(j - 2)

            v = str(i)
            g.edge(u, v, label=op, fillcolor="gray")

    # output node
    g.node("c_{k}", fillcolor="palegoldenrod")
    for i in range(n_nodes):
        g.edge(str(i), "c_{k}", fillcolor="gray")

    # add image caption
    if caption:
        g.attr(label=caption, overlap="false", fontsize="20", fontname="times")

    g.render(file_path, view=False)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise ValueError(f"usage:\n python {sys.argv[0]} GENOTYPE")

    genotype_str = sys.argv[1]
    try:
        genotype = gt.from_str(genotype_str)
    except AttributeError:
        raise ValueError(f"Cannot parse {genotype_str}")

    plot(genotype.normal, "normal")
    plot(genotype.reduce, "reduction")
