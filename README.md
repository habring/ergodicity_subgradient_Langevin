# ergodicity_subgradient_Langevin



<!-- GETTING STARTED -->
## Getting Started

In this repository you find the source code to reproduce the results of the paper ["Ergodicity of Langevin Dynamics and its Discretizations for Non-smooth potentials" by L. Fruehwirth and A. Habring](https://arxiv.org/pdf/2411.12051).

### Usage

### Structure of the repository

The repository contains files to run the actual simulation/sampling and files to evaluate the results/create the plots appearing in the paper. The `2d_example.py` is to run the 2D experiments and `imaging_example.py` to run the image denoising and deconvolution experiments from the paper. In each of those files at the top you can adapt the settings (choose the sampling method [explicit, semi-implicit, MYULA], choose the experiment [with or withour a linear operator `K` in the 2D experiments; denoising or deconvolution in the imaging experiments; see paper for more info], define hyperparameters). The file `evaluate_2d.ipynb` contains a jupyter notebook to create all figuers for the 2D experiments and the file `evaluate_imaging.ipynb` for the imaging experiments. Note that you need to run all the corresponding sampling experiments before the evaluation (or exclude some of the evaluation to focus on certain experiments).

### Reproducing the results

We recommend simply using conda as an environment management system.
1. Create a new conda environment with python 3.9.
  ```sh
  conda create -n subgrad-ergodic-env python=3.9
  ```
2. Activate the environment
   ```
   conda activate subgrad-ergodic-env
   ```
2. Install the relevant packages
   ```
   conda install numpy matplotlib scipy imageio ipython tqdm seaborn ipykernel
   conda install conda-forge::pot
   ```
3. Run `2d_example.py` and `imaging_example.py` with all relevant settings using `ipython 2d_example.py` and `ipython imaging_example.py` after adapting the corresponding lines of code at the top of the files.
4. In order to be able to choose the conda environment in your jupyter notebook run
   ```
   python -m ipykernel install --user --name=subgrad-ergodic-env
   ```
6. Open the jupyter notebooks and evaluate the results.



## Citation

If you use this code, please consider citing the paper
```
@article{fruehwirth2024ergodicity,
title={Ergodicity of Langevin Dynamics and its Discretizations for Non-smooth Potentials},
author={Fruehwirth, Lorenz and Habring, Andreas},
journal={arXiv preprint arXiv:2411.12051},
year={2024}
}
```

## Contact

Please contact me under `habring<dot>andreas<at>gmail<dot>com` if there are anny issues or you have questions regarding the work.
