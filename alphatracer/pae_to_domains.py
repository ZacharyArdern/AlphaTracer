# pae_to_domains.py — vendored from https://github.com/tristanic/pae_to_domains
# Copyright (c) 2021 Tristan Croll. MIT License.
# Bundled here to avoid a separate install step; no modifications made.

def parse_pae_file(pae_json_file):
    import json, numpy

    with open(pae_json_file, 'rt') as f:
        data = json.load(f)[0]

    if 'residue1' in data and 'distance' in data:
        # Legacy PAE format, keep for backwards compatibility.
        r1, d = data['residue1'], data['distance']
        size = max(r1)
        matrix = numpy.empty((size, size), dtype=numpy.float64)
        matrix.ravel()[:] = d
    elif 'predicted_aligned_error' in data:
        # New PAE format.
        matrix = numpy.array(data['predicted_aligned_error'], dtype=numpy.float64)
    else:
        raise ValueError('Invalid PAE JSON format.')

    return matrix

def domains_from_pae_matrix_networkx(pae_matrix, pae_power=1, pae_cutoff=5, graph_resolution=1):
    '''
    Takes a predicted aligned error (PAE) matrix representing the predicted error in distances between each
    pair of residues in a model, and uses a graph-based community clustering algorithm to partition the model
    into approximately rigid groups.

    Arguments:

        * pae_matrix: a (n_residues x n_residues) numpy array. Diagonal elements should be set to some non-zero
          value to avoid divide-by-zero warnings
        * pae_power (optional, default=1): each edge in the graph will be weighted proportional to (1/pae**pae_power)
        * pae_cutoff (optional, default=5): graph edges will only be created for residue pairs with pae<pae_cutoff
        * graph_resolution (optional, default=1): regulates how aggressively the clustering algorithm is. Smaller values
          lead to larger clusters. Value should be larger than zero, and values larger than 5 are unlikely to be useful.

    Returns: a series of lists, where each list contains the indices of residues belonging to one cluster.
    '''
    try:
        import networkx as nx
    except ImportError:
        print('ERROR: This method requires NetworkX (>=2.6.2) to be installed. Please install it using "pip install networkx" '
            'in a Python >=3.7 environment and try again.')
        import sys
        sys.exit()
    import numpy
    weights = 1/pae_matrix**pae_power

    g = nx.Graph()
    size = weights.shape[0]
    g.add_nodes_from(range(size))
    edges = numpy.argwhere(pae_matrix < pae_cutoff)
    sel_weights = weights[edges.T[0], edges.T[1]]
    wedges = [(i,j,w) for (i,j),w in zip(edges,sel_weights)]
    g.add_weighted_edges_from(wedges)

    from networkx.algorithms import community

    clusters = community.greedy_modularity_communities(g, weight='weight', resolution=graph_resolution)
    return clusters

def domains_from_pae_matrix_igraph(pae_matrix, pae_power=1, pae_cutoff=5, graph_resolution=1):
    '''
    Takes a predicted aligned error (PAE) matrix representing the predicted error in distances between each
    pair of residues in a model, and uses a graph-based community clustering algorithm to partition the model
    into approximately rigid groups.

    Arguments:

        * pae_matrix: a (n_residues x n_residues) numpy array. Diagonal elements should be set to some non-zero
          value to avoid divide-by-zero warnings
        * pae_power (optional, default=1): each edge in the graph will be weighted proportional to (1/pae**pae_power)
        * pae_cutoff (optional, default=5): graph edges will only be created for residue pairs with pae<pae_cutoff
        * graph_resolution (optional, default=1): regulates how aggressively the clustering algorithm is. Smaller values
          lead to larger clusters. Value should be larger than zero, and values larger than 5 are unlikely to be useful.
    Returns: a series of lists, where each list contains the indices of residues belonging to one cluster.
    '''
    try:
        import igraph
    except ImportError:
        print('ERROR: This method requires python-igraph to be installed. Please install it using "pip install python-igraph" '
            'in a Python >=3.6 environment and try again.')
        import sys
        sys.exit()
    import numpy
    weights = 1/pae_matrix**pae_power

    g = igraph.Graph()
    size = weights.shape[0]
    g.add_vertices(range(size))
    edges = numpy.argwhere(pae_matrix < pae_cutoff)
    sel_weights = weights[edges.T[0], edges.T[1]]
    g.add_edges(edges)
    g.es['weight'] = sel_weights

    vc = g.community_leiden(weights='weight', resolution_parameter=graph_resolution/100, n_iterations=-1)
    membership = numpy.array(vc.membership)
    from collections import defaultdict
    clusters = defaultdict(list)
    for i, c in enumerate(membership):
        clusters[c].append(i)
    clusters = list(sorted(clusters.values(), key=lambda l:(len(l)), reverse=True))
    return clusters


def parse_pae_batch_rust(paths: list, step: int) -> dict:
    """Parse multiple AFDB PAE JSON files using the compiled Rust binary.

    Returns {path: (n_full, matrix)} where matrix is float32, shape (n_out, n_out)
    with n_out = ceil(n_full / step).
    """
    import os, struct, subprocess, numpy as np

    _bin_dir = os.path.join(
        os.environ.get('AT_CACHE_DIR', os.path.expanduser('~/.cache/alphatracer')),
        'sketch_rs', 'release',
    )
    binary = os.path.join(_bin_dir, 'parse-pae')
    if not os.path.exists(binary):
        raise FileNotFoundError(
            f'parse-pae binary not found at {binary}. '
            'Run: cargo build --release inside alphatracer/sketch_rs/'
        )

    r = subprocess.run([binary, str(step)] + list(paths), capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f'parse-pae failed: {r.stderr.decode()[:500]}')

    buf = r.stdout
    pos = 0
    results = {}
    for path in paths:
        n_full, n_out = struct.unpack_from('<II', buf, pos)
        pos += 8
        mat = np.frombuffer(buf, dtype=np.float32,
                            count=n_out * n_out, offset=pos).reshape(n_out, n_out).copy()
        pos += n_out * n_out * 4
        results[path] = (n_full, mat)
    return results


def domains_from_pae_subsampled(pae_matrix, n_full, step,
                                 pae_power=1, pae_cutoff=5, graph_resolution=1):
    """Run igraph domain detection on a subsampled PAE matrix and return
    clusters in terms of original (full-resolution) residue indices.

    pae_matrix: shape (n_out, n_out), values in Angstroms (float32 from Rust)
    n_full: original number of residues
    step: subsampling step (pae_matrix index i → original residue i*step)
    """
    clusters_sub = domains_from_pae_matrix_igraph(
        pae_matrix, pae_power=pae_power, pae_cutoff=pae_cutoff,
        graph_resolution=graph_resolution,
    )
    expanded = []
    for cluster in clusters_sub:
        orig = []
        for idx in cluster:
            start = idx * step
            end = min(start + step, n_full)
            orig.extend(range(start, end))
        expanded.append(sorted(orig))
    return expanded


_defaults = {
    'output_file':  'clusters.csv',
    'pae_power':    1.0,
    'pae_cutoff':   5.0,
    'resolution':   1.0,
    'library':      'igraph'
}

if __name__ == '__main__':
    from time import time
    start_time = time()
    import argparse
    parser = argparse.ArgumentParser(description='Extract pseudo-rigid domains from an AlphaFold PAE matrix.')
    parser.add_argument('pae_file', type=str, help="Name of the PAE JSON file.")
    parser.add_argument('--output_file', type=str, default=_defaults['output_file'], help=f'Name of output file (comma-delimited text format. Default: {_defaults["output_file"]}')
    parser.add_argument('--pae_power', type=float, default=_defaults['pae_power'], help=f'Graph edges will be weighted as 1/pae**pae_power. Default: {_defaults["pae_power"]}')
    parser.add_argument('--pae_cutoff', type=float, default=_defaults['pae_cutoff'], help=f'Graph edges will only be created for residue pairs with pae<pae_cutoff. Default: {_defaults["pae_cutoff"]}')
    parser.add_argument('--resolution', type=float, default=_defaults['resolution'], help=f'Higher values lead to stricter (i.e. smaller) clusters. Default: {_defaults["resolution"]}')
    parser.add_argument('--library', type=str, default=_defaults['library'], help=f'Graph library to use. "igraph" is about 40 times faster; "networkx" is pure Python. Default: {_defaults["library"]}')
    args = parser.parse_args()
    pae = parse_pae_file(args.pae_file)
    lib = args.library
    if lib=='igraph':
        f = domains_from_pae_matrix_igraph
    else:
        f = domains_from_pae_matrix_networkx
    clusters = f(pae, pae_power=args.pae_power, pae_cutoff=args.pae_cutoff, graph_resolution=args.resolution)
    max_len = max([len(c) for c in clusters])
    clusters = [list(c) + ['']*(max_len-len(c)) for c in clusters]
    output_file = args.output_file
    with open(output_file, 'wt') as outfile:
        for c in clusters:
            outfile.write(','.join([str(e) for e in c])+'\n')
    end_time = time()
    print(f'Wrote {len(clusters)} clusters to {output_file}. Biggest cluster contains {max_len} residues. Run time was {end_time-start_time:.2f} seconds.')
