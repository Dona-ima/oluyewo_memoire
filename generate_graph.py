import os
os.makedirs("docs", exist_ok=True)

from app.graph.build import compiler_graphe

graphe = compiler_graphe()
image_bytes = graphe.get_graph().draw_mermaid_png()

with open("docs/graph_oluyewo.png", "wb") as f:
    f.write(image_bytes)

print("Graphe sauvegardé dans docs/graph_oluyewo.png")