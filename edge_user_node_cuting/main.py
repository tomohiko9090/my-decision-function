def edge_user_node_cuting(G, graph_type, threshold=False):
    
    if threshold:
        edges = [i for i in list(G.edges(data=True)) if i[2]["weight"] >= threshold]
    if not threshold:
        edges = [i for i in list(G.edges(data=True)) if i[2]["weight"] >= 1]
    
    if graph_type == "Graph":
        G_remove = nx.Graph()
    if graph_type == "DiGraph":
        G_remove = nx.DiGraph()
        
    G_remove.add_edges_from(edges)
    
    return G_remove