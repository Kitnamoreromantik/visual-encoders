# Vision encoders topology

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Study of the topological properties of vision encoders.

### Results
- The report is located in the `/reports` directory.
- The experimental outcomes, along with the experimental settings and logs, can be found in the `data/results` folder.


### How to run

1. Recreate the conda environment:
```
conda env create -f environment.yml
conda activate visual-encoders-env
```

2. Create in the project root a `/data` folder then download and unzip there the [dataset](https://figshare.com/articles/dataset/ImageNet_7x500/27097807?file=49389991) needed for the Part 1 experiments.

3. To reproduce the Part 1 experiment (anisotropy and intrinsic dimension of layers), run `/src/topology/main.py`. If needed, experimental setting might be adjusted at `/src/topology/setup.py`. Then the plots cam be reproduced bu running `/src/topology/plot.py` as well as `notebooks/topology_plot.ipynb`.

4. To reproduce the Part 2 experiment (internal representations via linear probing), refer to `/src/probing/probing.py`.