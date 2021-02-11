This repo is forked from https://github.com/balajiln/mondrianforest .

Install the following python packages 
(and possibly other packages) to run the scripts:

* numpy
* scipy
* matplotlib (for plotting Mondrian partitions)
* pydot and graphviz (for printing Mondrian trees)
* sklearn (for reading libsvm format files)

Some of the packages (e.g. pydot, matplotlib) are necessary only for '--draw_mondrian 1' option. If you just want to run experiments
without plotting the Mondrians, these packages may not be necessary.
To easily install the packages use 'pip install -r requirements.txt'.

Help on usage for mondrian settings can be obtained by typing the following commands on the terminal:

./mondrianforest.py -h

**Example usage**:

./mondrianforest_demo.py --dataset unary-feat --n_mondrians 100 --budget -1 --normalize_features 1 --optype class

**Example on a dataset**:

*assuming you have the csv files (for unary features and labels) divided into two subfolders called 'unary_csv' and 'labels_csv' from the path given in input*

./unary_features_mf.py --n_mondrians 40 --budget -1 --normalize_features 1 --optype class --data_path [path to the directory containing the dataset] --n_minibatch 4

