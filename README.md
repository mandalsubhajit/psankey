<img src="https://github.com/mandalsubhajit/venndata/blob/master/pSankey.png" width="200">


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

**Usage**

Start by importing the modules.
```python
from psankey.sankey import sankey
import pandas as pd
import matplotlib.pyplot as plt
```

## Citing **pSankey**

To cite the library if you use it in scientific publications (or anywhere else, if you wish), please use the link to the GitHub repository (https://github.com/mandalsubhajit/pSankey). Thank you!
