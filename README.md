This repository contains the code for my Machine Learning Algorithms and Applications Project.

## Installation

The simplest way to install the required packages is to use the provided `environment.yml` file and conda:

```bash
conda env create -f environment.yml
conda activate wlclf
```

Once installed you can experiment with the code in the notebooks.

## Functionality

Currently, two main functionalities are implemented, the representation and evaluation of C2 formulas and a simple graph classifier based on color refinement and decision trees.

### C2 Formulas

In c2.py the class Formula is used to represent and evaluate formulas in the logic C2, two-variable first order logic with counting quantifiers. The formulas can be evaluated on graphs:

```python
from c2 import *
import networkx as nx

formula = Exists(Var.x, (Exists(Var.y, E(Var.x, Var.y), 7)))
formulas.evaluate(nx.fast_gnp_random_graph(10, 0.5))
```
### Graph Classification

In wlclassifier.py a simple graph classifier is implemented:
```python
phi = Exists(Var.x, (Exists(Var.y, E(Var.x, Var.y), 7)))
X = [nx.fast_gnp_random_graph(10, 0.5) for _ in range(100)]
y = [phi.evaluate(g) for g in X]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = WLClassifier()
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
```

