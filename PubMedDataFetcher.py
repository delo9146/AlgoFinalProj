from pymed import PubMed
import pandas as pd
import nltk
import string
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('stopwords')
import requests
from bs4 import BeautifulSoup
from itertools import combinations
import random
import math
from collections import defaultdict
from collections import Counter
import matplotlib.pyplot as plt

def pubmed_query(query, email, retmax=100):
    """
    Retrieve PubMed data for a given query.

    Args:
        query (str): Query to search PubMed with.
        email (str): Email address to use for PubMed API.
        retmax (int): Maximum number of records to retrieve.

    Returns:
        List of dictionaries containing the PubMed data for each article retrieved.
    """
    pubmed = PubMed(tool="MyTool", email=email)

    # Add "osteoporosis[AB]" to the query to search for articles related to osteoporosis in the abstract field
    query = f"{query} [AB]"

    results = pubmed.query(query=query, max_results=retmax)

    articles = []
    for article in results:
        if article.abstract is not None:
            authors = []
            for author in article.authors:
                if author.get('firstname') is not None and author.get('lastname') is not None:
                    name = author['firstname'] + ' ' + author['lastname']
                    authors.append(name)

            article_data = {
                "pmid": article.pubmed_id,
                "title": article.title,
                "abstract": article.abstract,
                "date": str(article.publication_date),
                "authors": authors,
            }
            articles.append(article_data)

    return articles



def process_abstracts(results_dict, terms):
    """
    Tokenize and remove punctuation from the abstracts in the results_dict.

    Args:
        results_dict (dict): Dictionary containing the PubMed data for each combination of terms searched.
        terms (list): List of terms searched.

    Returns:
        pandas dataframe with processed abstracts.
    """
    processed_abstracts = []
    for key, value in results_dict.items():
        for article in value:
            if article['abstract'] is not None:
                # replace Greek characters with their English counterparts
                article['abstract'] = article['abstract'].replace('β', 'beta').replace('α', 'alpha')
                # check if term is in abstract and tokenize the term if found
                found_terms = []
                for term in terms:
                    term_count = 0
                    if term.lower() in article['abstract'].lower() or term.lower().replace('-', ' ') in article['abstract'].lower():
                        term_count = article['abstract'].lower().count(term.lower()) + article['abstract'].lower().count(term.lower().replace('-', ' '))
                        found_terms.append({'term': term.lower(), 'count': term_count})
                # tokenize the abstract and remove punctuation
                text = article['abstract'].lower()
                table = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
                text = text.translate(table)
                tokens = nltk.word_tokenize(text)
                # add the processed abstract and found terms to the list
                processed_abstracts.append({'key': key, 'abstract': ' '.join(tokens), 'found_terms': found_terms})
    df = pd.DataFrame(processed_abstracts)
    # count the unique terms in the "found_terms" column
    term_counts = Counter()
    for row in df['found_terms']:
        for term in row:
            term_counts[term['term']] += term['count']
    # create a visualization displaying the counts
    term_counts_df = pd.DataFrame.from_dict(term_counts, orient='index', columns=['count'])
    term_counts_df.sort_values(by='count', ascending=False, inplace=True)
    plt.figure(figsize=(10, 6))
    plt.bar(x=term_counts_df.index, height=term_counts_df['count'])
    plt.xticks(rotation=90)
    plt.xlabel('Term')
    plt.ylabel('Count')
    plt.title('Counts of Unique Terms')
    plt.show()
    return df



def find_synonyms(term):
    """
    Find synonyms for a medical term using the UMLS Metathesaurus.

    Args:
        term (str): Medical term to find synonyms for.

    Returns:
        List of strings containing synonyms for the term.
    """
    # Construct URL for UMLS Metathesaurus search
    base_url = "https://metamap.nlm.nih.gov/Docs/SemanticTypes_2018AB.txt"
    search_url = f"https://metamap.nlm.nih.gov/metamaplite/rest/semanticType/{term}?version=2018AB"

    # Send HTTP GET request to UMLS Metathesaurus search API
    response = requests.get(search_url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Parse HTML response and extract synonyms
    synonyms = []
    for tag in soup.find_all('a'):
        name = tag.get('name')
        if name and name.startswith('result'):
            text = tag.get_text()
            synonyms.append(text)

    return synonyms

def search_terms(terms, email, retmax):
    """
    Search PubMed for each term in the list and store the results in a dictionary.

    Args:
        terms (list): List of terms to search PubMed for.
        email (str): Email address to use for PubMed API.
        retmax (int): Maximum number of records to retrieve per term.
    
    Returns:
        Dictionary containing the PubMed data for each term searched.
    """
    results_dict = {}
    for term in terms:
        # Add single quotes before and after the term
        query = f'"{term}"'
        results = pubmed_query(query=query, email=email, retmax=retmax)
        results_dict[term] = results
    return results_dict



terms = ['beta-catenin', 'zyxin', 'p130Cas', 'PTH', 'PTHR1', 'ECM', 'BCAR1', 'Breast cancer anti-estrogen resistance protein 1',
         'rage', 'HMGB1', 'High-Mobility Group', 'High-Mobility Group Box 1', 'ZYX', 'ESP-2', 'HED-2',
         'atf', 'r-smad', 'SMAD2', 'SMAD3', 'SMAD1', 'SMAD5', 'SMAD8', 'Smad4', 'Nmp4', 'actin', 'alpha-actinin']

email = 'your.email@example.com'
retmax = 300
results_dict = search_terms(terms, email, retmax)
tok_abstracts = process_abstracts(results_dict, terms)
tok_abstracts = tok_abstracts[tok_abstracts['found_terms'].apply(len) != 0]

# Initialize a dictionary to keep track of the counts
counts = {}

# Loop over the rows in the "found_terms" column
for row in tok_abstracts['found_terms']:
    # Get the length of the list in the current row
    length = len(row)
    # If the length is not yet in the counts dictionary, initialize it to zero
    if length not in counts:
        counts[length] = 0
    # Increment the count for this length
    counts[length] += 1

# Print the counts
for length, count in counts.items():
    print(f"Number of lists with length {length}: {count}")


