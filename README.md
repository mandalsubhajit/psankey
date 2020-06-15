<img src="https://github.com/mandalsubhajit/psankey/blob/master/pSankey.png" width="100">


# psankey - A module for plotting Sankey flow diagrams in Python

Inspired by **d3-sankey** package for d3js (https://github.com/d3/d3-sankey)

## Brief description

In data science, we often require to visualize flows in the form of a Sankey diagram. This module helps with that. Usage is very straightforward and customizable.

**Note:** Does not work for cyclical graphs.

## Getting started

### Installation

Directly from the source - clone this repo on local machine, open the shell, navigate to this directory and run:
```
python setup.py install
```
or through pip:
```
pip install psankey
```

### Documentation

**Input Data Format**

A dataframe of links with the following columns (first 3 mandatory, rest optional):

source, target, value, color(optional), alpha(optional)

Example:
```
data1.csv

source,target,value,color,alpha
B,E,20,,
C,E,20,,
C,D,20,,
B,A,20,,
E,D,20,,
E,A,20,,
D,A,40,orange,0.85
```

**Usage**

```python
from psankey.sankey import sankey
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/data1.csv')
fig, ax = sankey(df, aspect_ratio=4/3, nodelabels=True, linklabels=True, labelsize=5, nodecmap='copper', nodealpha=0.5, nodeedgecolor='white')
plt.show()
```

**Parameters**

To be updated...

### Sample Output

<img src="https://github.com/mandalsubhajit/psankey/blob/master/output/sankey1.png" width="1000">

<img src="https://github.com/mandalsubhajit/psankey/blob/master/output/sankey2.png" width="1000">

## Citing **pSankey**

To cite the library if you use it in scientific publications (or anywhere else, if you wish), please use the link to the GitHub repository (https://github.com/mandalsubhajit/pSankey). Thank you!
