def compute_mst(points):
    """Compute the mst from the list of points"""
    n = len(points)
    heap = [(0., 0, -1)]
    tot_w = 0
    visited = [False] * n
    num_added = 0
    edges = []
    while num_added < n:
        w, curr_node, prev_node = heapq.heappop(heap)
        if visited[curr_node]:
            continue
        visited[curr_node] = True
        tot_w += w
        num_added += 1
        if prev_node != -1:
            edges.append((prev_node, curr_node, w))
        
        for j in range(n):
            if j != curr_node:
                if not visited[j]:
                    ww = math.sqrt((points[j][0] - points[curr_node][0]) ** 2 +
                                   (points[j][1] - points[curr_node][1]) ** 2)
                    heapq.heappush(heap, (ww, j, curr_node))
    return tot_w, edges


def mst(path, dataset, target_cl, points):
    """Compute the mst of the points"""
    mst_weight, mst_edges = compute_mst(points=points)
    print(len(mst_edges))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for u, v, w in mst_edges:
        line = Line2D(xdata=[points[u][0], points[v][0]], ydata=[points[u][1], points[v][1]])
        ax.add_line(line)
    scatter_out_path = path + "images/scatter/{}/".format(dataset)
    plt.show()
    # fig.savefig(scatter_out_path + "{}_mst_self.png".format(target_cl))
    plt.close()

    def cluster_and_plot(path, dataset, target_cl, points):
        """Cluster the points and plot"""
        scatter_out_path = path + "images/scatter/{}/".format(dataset)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=2, gen_min_span_tree=True,
                                    allow_single_cluster=True)
        clusterer.fit(points)
        clusterer.minimum_spanning_tree_.plot(edge_cmap='viridis',
                                              edge_alpha=0.6,
                                              node_size=80,
                                              edge_linewidth=2)
        mst = clusterer.minimum_spanning_tree_.to_networkx()
        plt.savefig(scatter_out_path + "{}_mst.png".format(target_cl))
        plt.close()
        print(mst, mst.number_of_nodes(), mst.number_of_edges())
        weights = []
        for e in list(mst.edges):
            print(e)
            break