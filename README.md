# SyllableBifurcations

Repository with code and data for the project. 

## Structure of the repo

- Root folder:
    - `PhysicalModelAirflow.ipynb`: differential equations for the physical quantities involved in the airflow production, and their numerical solution, which resembles a competition between raising and decaying sigmoids.
    - `{Hopf,VdP,SNILC}-Syllables.ipynb`: calculation of a single solution, plotting it, saving it to `Output/`, computing its spectrum and displaying an audio snippet of the soundwave.
    - `FourierSpectrograms.ipynb`: this notebook displays the solution accompanied by the sliding window Fourier decomposition of each solution, and saves the resulting figure to `Output/`.
    - Python files `*.py`: containing manual numerical solvers, functions for the differential equations for each bifurcation, plotting helpers, etc.
    
- `SolutionProperties/`: contains notebooks with computations of the envelope profiles (and their fits to the sigmoidal airflow model) and its frequency spectra (and the indices computed from its discretization) for each of the calculated solutions.

- `IndexComputations/`:
    - `{Hopf,VdP,SNILC}_solutions.py`: iteratively calculate the solution to for each bifurcation, varying $\lambda_1/\lambda_2$, saving them to `Solutions/`.
    - `SolutionChecker.ipynb`: this notebook enables visualizing each solution and the manual computation of each index.
    - `solution_indices.py`: compute the indices associated to each of the solutions stored in `Solutions/`, saving them to a dictionary in `Indices/`
    - `IndexSpace.ipynb`: the indices computed from the previous script are loaded here and plotted in 3D along three 2D projections.