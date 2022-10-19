## AlignGraph:

A group of generative models that combine fast and efficient tractable graph alignment methods with a family of deep generative models and are thus invariant to node permutations.


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
python G_Parallel_galign.py --center  True --data "graph-name"
```

### To train generative models:


G-align-Single (VAE):
```py
python parallel_single.py  --model gae --data "graph-name"  
```

G-align-Single (GraphRNN):
```py
python main.py   
```

G-align-Single (GRAN):
```py
python run_exp.py -c config/dataset_info   
```

G-align-Double and Fermat-Double (VAE):
```py
python parallel_double.py  --model gae --data "graph-name"  
```

G-align-Double and Fermat-Double (GraphRNN):
```py
python main_double.py   
```

G-align-Double and Fermat-Double (GRAN):
```py
python run_exp_double.py -c config/dataset_info
```
## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.
