# Datasets

Arxiv HEP-TH (high energy physics theory) citation graph is from the e-print arXiv and covers all the citations within a dataset of 27,770 papers with 352,807 edges. If a paper i cites paper j, the graph contains a directed edge from i to j. If a paper cites, or is cited by, a paper outside the dataset, the graph does not contain any information about this.

The data covers papers in the period from January 1993 to April 2003 (124 months). It begins within a few months of the inception of the arXiv, and thus represents essentially the complete history of its HEP-TH section.

Brief explanation of the files included in this folder:

### Network files
The High-energy physics theory citation network [dataset](https://snap.stanford.edu/data/cit-HepTh.html) can be retrieved from [SNAP](https://snap.stanford.edu/index.html) (Stanford Network Analysis Project) 
- cit-HepTh.txt: Paper citations
- cit-HepTh-abstracts.tar.gz: All paper abstracts
- cit-HepTh-dates.txt.gz: Papers with published date. Note: This was not used as not all papers in our network have a date in this dataset. We end up using the date listed in the abstracts.

### Preprocessed files

Due to the inconsistencies found in our dataset (duplicates, cycles, self-referencing papers), it was required to do some preprocessing of it before carrying out out analysis.

Files used to perform our network analysis:
- cit-HepTh-nodes.csv: Contains all the unique nodes of our network with the date the paper was published and the last time it was cited.
- cit-HepTh-with-dates.csv: Edges dataset containing the dates the from and to nodes where published.
- domains_with_countries: Contains the locations of all the institutions that have published a paper in our network.


## Enrichment datasets

- countries.tsv: Latitude, Longitude of all the countries in the world.
- abstracts: Contains all the paper abstracts by year.