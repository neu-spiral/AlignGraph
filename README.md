## Documentation




#### AlignGraph: A group of generative models that combine fast and efficient tractable graph alignment methods with a family of deep generative models and are thus invariant to node permutations.


## Dependencie

- python 3.6.10


## Usage
Install packages:

```py
pip install -r requirements.txt
```
To create graphs:

```py
python graph_creation.py --graph  "graph-type"
```

To compute the center graph and graph alignment, depending on the graph alignment method use either G_Parallel_galign.py, G_Parallel_fermat.py, CG_Parallel_galign.py or CG_Parallel_fermat.py.

Example:
```py
python G_Parallel_galign.py --center  True
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.
