# SyllableBifurcations

## Structure of the repo

- `PhysicalModelAirflow.ipynb`: differential equations for the physical quantities involved in the airflow production, and their numerical solution, which resembles a competition between raising and decaying sigmoids.
- `{Hopf,VdP,SNILC}-Syllables.ipynb`: calculation of a single solution, plotting it, saving it to `Output/`, computing its spectrum and displaying an audio snippet of the soundwave.
- `FourierSpectrograms.ipynb`: this notebook displays the solution accompanied by the sliding window Fourier decomposition of each solution, and saves the resulting figure to `Output/`.
- `SolutionProperties/`: contains notebooks with computations of the envelope profiles (and their fits to the sigmoidal airflow model) and its frequency spectra (and the indices computed from its discretization).