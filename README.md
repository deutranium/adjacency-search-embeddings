# Adjacency based embeddings

## File structure:
1. `MAS.py`: implementation of the MAS algorithm
2. `TAS.py`: implementation of TAS algorithm
3. `main.ipynb`: Sample notebook demonstrating execution of TAS and MAS to get the embeddings

## How to run?
Please refer to main.ipynb to check the execution process. We have provided the code to execute MAS (implemented in MAS.py) and TAS (implemented in TAS.py).

## Data format
Please provide the data as a pickle file in the below format:
```
[g,
[train_X, train_Y],
[val_X, val_Y],
[test_X, test_Y]]
```
As a sanity check, plese ensure that the pickle file can be loaded using the code below:
``` python
with open(DATA_PATH, "rb") as f:
    (
        g,
        [train_X, train_y],
        [val_X, val_y],
        [test_X, test_y],
    ) = pickle.load(f)
```
Here,
- `g` is a `networkx.classes.graph.Graph` object containing the graph
- `train_X`, `val_X` and `test_X` are lists containing train, val and test splits respectively. Every element of the list represents the node ID of a node in the respective split
- `train_Y`, `val_Y` and `test_Y` are lists containing the node labels for the respective splits.