import pandas as pd
import nltk
from Bio import Entrez
nltk.download('stopwords')
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt



def fetch_pubmed_abstracts2(query, retmax):
    """
    Fetches abstracts and MeSH terms for a given PubMed query and returns them as a DataFrame.

    Args:
        query (str): The query to search PubMed for.
        retmax (int): The maximum number of results to retrieve (default is 1000).

    Returns:
        pandas.DataFrame: A DataFrame containing the abstracts, their MeSH terms, and the counts of the terms in the abstracts.
    """
    # Set the email address associated with your Entrez account
    Entrez.email = 'your_email@example.com'

    # Search PubMed for the query and retrieve the abstracts
    handle = Entrez.esearch(db='pubmed', retmax=retmax, term=query)
    record = Entrez.read(handle)
    handle.close()
    ids = record['IdList']

    # Retrieve the abstracts and their medical annotations
    handle = Entrez.efetch(db='pubmed', id=ids, retmode='xml')
    records = Entrez.read(handle)
    handle.close()

    # Parse the records and extract the abstracts and their MeSH terms
    abstracts = []
    mesh_terms = []
    mesh_counts = []
    for record in records['PubmedArticle']:
        try:
            abstract = record['MedlineCitation']['Article']['Abstract']['AbstractText'][0]
        except:
            abstract = ''
        try:
            terms = [m['DescriptorName'] for m in record['MedlineCitation']['MeshHeadingList']]
            counts = []
            for term in terms:
                count = abstract.count(term)
                counts.append(count)
        except:
            terms = []
            counts = []
        abstracts.append(abstract)
        mesh_terms.append(terms)
        mesh_counts.append(counts)

    # Create a DataFrame of the abstracts, their MeSH terms, and the counts of the terms in the abstracts
    df = pd.DataFrame({'abstract': abstracts, 'mesh_terms': mesh_terms, 'mesh_counts': mesh_counts})

    return df

def count_unique_values(df, col):
    """
    Given a pandas DataFrame and a column name containing lists of values, 
    returns a new DataFrame with a count of each unique value in the lists.
    
    Parameters:
        - df (pandas.DataFrame): the DataFrame containing the column of interest
        - col (str): the name of the column to count unique values in
    
    Returns:
        - pandas.DataFrame: a DataFrame with two columns:
            - unique_value: the unique value found in the lists
            - count: the number of times the unique value appears in the lists
    """
    unique_values = []
    counts = []

    for lst in df[col]:
        unique_values += lst

    unique_counts = Counter(unique_values)
    unique_values = list(unique_counts.keys())
    counts = list(unique_counts.values())

    return pd.DataFrame({'unique_value': unique_values, 'count': counts})

def count_terms(df):
    """
    Takes a DataFrame with 'abstract' and 'mesh_terms' columns, and counts the number of occurrences of each 
    mesh term in the corresponding abstract. Returns the DataFrame with an additional 'term_counts' column,
    which contains a dictionary of mesh terms and their counts for each abstract.

    Parameters:
    df (pandas.DataFrame): DataFrame containing 'abstract' and 'mesh_terms' columns.

    Returns:
    pandas.DataFrame: DataFrame with an additional 'term_counts' column containing a dictionary of mesh terms
    and their counts for each abstract.
    """
    def count_term(term, abstract):
        """
        Counts the number of occurrences of a term in a given abstract.

        Parameters:
        term (str): The term to count occurrences of.
        abstract (str): The abstract to search for occurrences of the term.

        Returns:
        int: The number of occurrences of the term in the abstract.
        """
        return abstract.count(term)
    
    term_counts = []
    for i in range(len(df)):
        abstract = df.iloc[i]['abstract']
        mesh_terms = df.iloc[i]['mesh_terms']
        term_count = {}
        for term in mesh_terms:
            count = count_term(term, abstract)
            term_count[term] = count
        term_counts.append(term_count)
        
    df['term_counts'] = term_counts
    return df 

def remove_zero_rows(df):
    """
    Removes rows from a DataFrame where all values in the 'mesh_counts' column are zero or where there is only one
    non-zero value in the 'mesh_counts' column.

    Args:
        df (pandas.DataFrame): The DataFrame to process.

    Returns:
        pandas.DataFrame: The processed DataFrame with rows removed where all values in 'mesh_counts' are zero or
        where there is only one non-zero value in 'mesh_counts'.
    """
    # Remove rows with all zeros in mesh_counts
    df = df.loc[~(df['mesh_counts'].apply(lambda x: all(v == 0 for v in x))), :]
    
    # Remove rows with only one non-zero element in mesh_counts
    df = df.loc[~(df['mesh_counts'].apply(lambda x: sum(v != 0 for v in x) == 1)), :]

    return df

def remove_rows_with_highest_count(df, counts):
    """
    Removes all rows in a dataframe where a specific term occurs the most in the 'mesh_terms' column.

    Parameters:
    df (pandas.DataFrame): The dataframe to remove rows from.
    counts (pandas.DataFrame): The result of the count_unique_values function applied to the 'mesh_terms' column in df.

    Returns:
    pandas.DataFrame: The modified dataframe with rows removed.
    """
    # get the unique_value with the highest count
    highest_count = counts['count'].max()
    unique_value = counts[counts['count'] == highest_count]['unique_value'].iloc[0]

    # iterate through every list of terms in the mesh_terms column
    for index, row in df.iterrows():
        if unique_value in row['mesh_terms']:
            # remove the row from the dataframe
            df.drop(index, inplace=True)

    return df

def create_direct_association_matrix(df):
    """
    This function takes a DataFrame with a "mesh_terms" column as input, and creates a direct association matrix
    for the mesh terms. The direct association matrix is a square matrix where each row and column represents a mesh term,
    and each cell (i,j) represents the number of documents in which both mesh term i and mesh term j appear.
    
    Args:
    - df: a pandas DataFrame containing a "mesh_terms" column
    
    Returns:
    - direct_association_matrix: a pandas DataFrame representing the direct association matrix
    """
    
    # Join the mesh terms for each document into a string and vectorize the corpus
    corpus = [" ".join(doc) for doc in df["mesh_terms"]]
    vectorizer = CountVectorizer(token_pattern=r"\b\w+\b")
    X = vectorizer.fit_transform(corpus)
    
    # Get the vocabulary and word counts for each document
    vocab = vectorizer.get_feature_names()
    word_counts = X.toarray()
    doc_word_counts = pd.DataFrame(word_counts, columns=vocab)
    
    # Calculate the direct association matrix by multiplying the document-word count matrix with its transpose
    direct_association_matrix = doc_word_counts.T.dot(doc_word_counts)
    
    return direct_association_matrix

def floyd_warshall_adj_matrix(da_matrix):
    """
    Given a direct association matrix, return the adjacency matrix for Floyd-Warshall algorithm.

    Args:
        da_matrix (pd.DataFrame): direct association matrix

    Returns:
        np.ndarray: adjacency matrix for Floyd-Warshall algorithm
    """
    # Get a list of unique nodes in the direct association matrix
    nodes = list(set(da_matrix.index) | set(da_matrix.columns))

    # Initialize the adjacency matrix with infinite distances for all pairs of nodes
    adj_matrix = np.full((len(nodes), len(nodes)), np.inf)

    # Set the weights based on the values of the direct association matrix
    for i, node1 in enumerate(nodes):
        for j, node2 in enumerate(nodes):
            if node1 == node2:
                adj_matrix[i, j] = 0
            elif (node1 in da_matrix.index and node2 in da_matrix.columns
                  and da_matrix.loc[node1, node2] > 0):
                adj_matrix[i, j] = da_matrix.loc[node1, node2]

    return adj_matrix

def check_negative_weights(matrix):
    """
    Check if a matrix has negative weights.
    
    Parameters:
    -----------
    matrix: pandas.DataFrame or numpy.array
        The matrix to check for negative weights.
        
    Returns:
    --------
    None
    """
    if (matrix < 0).any().any():
        print("The resulting matrix has negative weights.")
    else:
        print("The resulting matrix does not have negative weights.")

        
def is_direct_matrix_fully_connected(matrix):
    """
    Checks if a given direct association matrix is fully connected or not.

    Parameters:
    matrix (pandas DataFrame): The direct association matrix.

    Returns:
    None.
    """
    n = matrix.shape[0]
    visited = set()
    stack = [0]  # start at node 0

    while stack:
        node = stack.pop(0)
        visited.add(node)
        neighbors = matrix.columns[matrix.iloc[node] > 0].tolist()
        for neighbor in neighbors:
            neighbor_index = matrix.columns.get_loc(neighbor)
            if neighbor_index not in visited:
                stack.append(neighbor_index)

    if len(visited) == n:
        print("The matrix is fully connected.")
    else:
        print("The matrix is not fully connected.")

def floyd_warshall(matrix):
    """
    This function takes in an adjacency matrix representing a weighted graph, where the value in cell (i,j) represents
    the weight of the edge from node i to node j. It then applies the Floyd-Warshall algorithm to calculate the shortest
    distance between each pair of nodes in the graph, and returns the resulting distance matrix.

    Args:
    - matrix: a square numpy array representing the adjacency matrix of the graph

    Returns:
    - dist: a numpy array representing the distance matrix of the graph, where the value in cell (i,j) represents the
    shortest distance between node i and node j
    """

    n = matrix.shape[0]
    dist = np.copy(matrix)

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i, k] + dist[k, j] < dist[i, j]:
                    dist[i, j] = dist[i, k] + dist[k, j]

    return dist


def visualize_floyd_warshall(dist):
    n = dist.shape[0]
    G = {i: set() for i in range(n)}

    for i in range(n):
        for j in range(n):
            if dist[i,j] != np.inf:
                G[i].add((j, dist[i,j]))

    plt.figure(figsize=(8,8))
    pos = {i: (np.cos(2*np.pi*i/n), np.sin(2*np.pi*i/n)) for i in range(n)}
    for i in range(n):
        for j, w in G[i]:
            plt.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]], 'k-', linewidth=0.5)
            plt.text((pos[i][0] + pos[j][0])/2, (pos[i][1] + pos[j][1])/2, str(w), fontsize=8)
    plt.xlim(-1.2, 1.2)
    plt.ylim(-1.2, 1.2)
    plt.axis('off')
    plt.show()
    

def edmonds_karp(matrix, source, sink):
    """
    This function implements the Edmonds-Karp algorithm for finding the maximum flow in a flow network represented
    by an adjacency matrix.

    Args:
        matrix (numpy.ndarray): the adjacency matrix representing the flow network
        source (int): the index of the source node
        sink (int): the index of the sink node

    Returns:
        int: the maximum flow in the flow network
    """
    n = matrix.shape[0]
    parent = np.zeros(n, dtype=int)
    max_flow = 0

    while True:
        visited = np.zeros(n, dtype=bool)
        visited[source] = True
        queue = [source]

        while queue:
            u = queue.pop(0)
            for v in range(n):
                if not visited[v] and matrix[u, v] > 0:
                    visited[v] = True
                    parent[v] = u
                    queue.append(v)
                    if v == sink:
                        path = []
                        curr = v
                        while curr != source:
                            path.append(curr)
                            curr = parent[curr]
                        path.append(source)
                        path.reverse()
                        min_flow = min(matrix[u, v] for u, v in zip(path[:-1], path[1:]))
                        for u, v in zip(path[:-1], path[1:]):
                            matrix[u, v] -= min_flow
                            matrix[v, u] += min_flow
                        max_flow += min_flow
                        break
            else:
                continue
            break
        else:
            break

    return max_flow

def find_maximum_flow(matrix):
    '''
    Given an input matrix representing a weighted graph, the function calculates the maximum flow between all pairs 
    of nodes in the graph using the Edmonds-Karp algorithm. The algorithm works by repeatedly finding the shortest 
    augmenting path in the residual graph (i.e., the graph with the residual capacities), which is the path from the 
    source node to the sink node with the minimum remaining capacity along the path. The function returns a matrix 
    where each element [i, j] is the maximum flow that can be sent from node i to node j.

    Parameters:
    matrix (numpy array): A square matrix representing a weighted graph.

    Returns:
    max_flow_matrix (numpy array): A square matrix where each element [i, j] is the maximum flow that can be sent 
                                   from node i to node j.
    '''
    n = matrix.shape[0]
    max_flow_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if matrix[i, j] != np.inf:
                for k in range(n):
                    if matrix[i, k] != np.inf and matrix[k, j] != np.inf:
                        if matrix[i, j] == matrix[i, k] + matrix[k, j]:
                            max_flow_matrix[i, j] = edmonds_karp(matrix, i, j)
                            break

    return max_flow_matrix

def max_flow_df(max_flow_matrix):
    """
    This function takes the resulting max_flow_matrix and stores in a Pandas DataFrame, which nodes have paths and what 
    the resulting max flow is.

    Args:
        max_flow_matrix (numpy.ndarray): the matrix of maximum flows between all pairs of nodes in the graph

    Returns:
        pandas.DataFrame: a DataFrame with three columns: 'source', 'sink', and 'max_flow', where each row represents 
                           a pair of nodes with a path and the resulting maximum flow between them
    """
    n = max_flow_matrix.shape[0]
    df = pd.DataFrame(columns=['source', 'sink', 'max_flow'])
    for i in range(n):
        for j in range(n):
            if max_flow_matrix[i, j] != np.inf and max_flow_matrix[i, j] > 0:
                df = df.append({'source': i, 'sink': j, 'max_flow': max_flow_matrix[i, j]}, ignore_index=True)
    return df

def find_transitive_cuts(distance_matrix):
    n = distance_matrix.shape[0]
    results = []

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if np.isnan(distance_matrix[i, k]) or np.isnan(distance_matrix[k, j]):
                    continue  # No transitive path from i to j through k
                if np.isnan(distance_matrix[i, j]) or distance_matrix[i, j] > distance_matrix[i, k] + distance_matrix[k, j]:
                    # Found a shorter path from i to j through k
                    # Calculate the minimum cut for the path from i to j
                    # using the Edmonds-Karp algorithm
                    matrix = np.zeros((n, n))
                    for u in range(n):
                        for v in range(n):
                            if not np.isnan(distance_matrix[u, v]):
                                matrix[u, v] = distance_matrix[u, v]
                    cut = edmonds_karp(matrix, i, j)
                    results.append((i, j, k, cut))
                    # Separate the nodes i and j by removing the minimum cut from the distance matrix
                    for u in range(n):
                        for v in range(n):
                            if not np.isnan(distance_matrix[u, v]):
                                distance_matrix[u, v] = max(distance_matrix[u, v] - cut, 0)

    return results

#--------------------------------------------------------------------------------------

terms = ['beta-catenin', 'zyxin', 'p130Cas', 'PTH', 'PTHR1', 'ECM', 'BCAR1', 'Breast cancer anti-estrogen resistance protein 1',
         'rage', 'HMGB1', 'High-Mobility Group', 'High-Mobility Group Box 1', 'ZYX', 'ESP-2', 'HED-2', 'NMP4/CIZ', 'NMP4 CIZ'
         'atf', 'r-smad', 'SMAD2', 'SMAD3', 'SMAD1', 'SMAD5', 'SMAD8', 'Smad4', 'Nmp4', 'actin', 'alpha-actinin', 'nuclear matrix protein 4']

email = 'your.email@example.com'
retmax = 20

dfs = []
for term in terms:
    df = fetch_pubmed_abstracts2(f'"{term}"', retmax)
    dfs.append(df)

# Concatenate the list of dataframes into a single dataframe
df_all = pd.concat(dfs, ignore_index=True)
df_all = df_all[df_all['mesh_terms'].apply(lambda x: len(x) > 0)]

# We are interested in transitive associations so any row with all zero term counts and any row with only 1 are deleted
df_all = remove_zero_rows(df_all)

# replace β with beta
df_all['abstract'] = df_all['abstract'].str.replace('β', 'beta')
# replace α with alpha
df_all['abstract'] = df_all['abstract'].str.replace('α', 'alpha')

# Unpacking mesh terms from StringElement object
df_all['mesh_terms'] = df_all['mesh_terms'].apply(lambda x: [str(term) for term in x])

"""
needed for big matrices
counts = count_unique_values(df_all, 'mesh_terms')
df_copy = df_all.copy()

df_updated = remove_rows_with_highest_count(df_copy, counts)
counts_updated = count_unique_values(df_updated, 'mesh_terms')
"""
df_updated = df_all
da_matrix = create_direct_association_matrix(df_updated)
check_negative_weights(da_matrix) # thank the lord
is_direct_matrix_fully_connected(da_matrix) # it's not!
adj_matrix = floyd_warshall_adj_matrix(da_matrix)
FW_graph = floyd_warshall(adj_matrix)
#visualize_floyd_warshall(FW_graph)
max_flow = find_maximum_flow(FW_graph)

which_ones = max_flow_df(max_flow)

min_cut = find_transitive_cuts(FW_graph)
