# Learning to Prune with Linear Context Free Rewriting Systems (LCFRS)


## Requirements
The following packages have to be installed before using ltplcfrs:
- [disco-dop](https://github.com/andreasvc/disco-dop) : library for
discontinuous data oriented parsing
- [pandas](https://pandas.pydata.org/) : used for dataset aggregation
- [scipy](https://www.scipy.org/) : necessary for pandas data sets
- [sklearn](http://scikit-learn.org/) : used for training/prediction


## Preparations
Prepare the tiger corpus:
```bash
$ iconv -f <current encoding> -t UTF-8 <corpus-file> > <utf8-corpus-file>
$ iconv -f ISO-8859-1 -t UTF-8 "path/to/corpus.xml" > "path/to/corpus.utf8.xml"
```


Usefull calls for building the project in the development phase:

```bash
$ pipenv --python 3.5.2
$ pipenv install -r requirements.txt
$ pipenv shell
$ python # starting interpreter for testing purposes
```


## Execution
```bash
$ python3 -m ltplcfrs
$ ltplcfrs
```

## Usage
Training of a pruning policy after inducing a grammar of a given corpus,
a number of trainings iterations and a accuracy-runtime trade off factor
(weight):
```bash
$ ltplcfrs train <corpus-file> <iterations> <weight>
$ ltplcfrs train "path/to/corpus.xml" 10 0.03 < initialpolicy.txt > policy.txt
```
The initial pruning policy can be omitted.


Parsing of a sentence with a pruning policy after inducing a grammer of a given
corpus:
```bash
$ ltplcfrs parse <corpus-file> <sentence>
$ ltplcfrs parse "path/to/corpus.xml" "Hello there ." < policy.txt
```
The pruning policy can be omitted.
