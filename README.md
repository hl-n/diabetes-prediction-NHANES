# Diabetes Prediction With NHANES Data


## Author

Hi there! 👋 I'm hl-n, the developer behind this project. I'm passionate about leveraging data to gain insights and solve real-world problems. If you have any questions, suggestions, or just want to connect, feel free to reach out to me [@hl-n](https://github.com/hl-n)!


## Overview

This repository contains a machine learning project for predicting the presence of diabetes (defined as glycohemoglobin levels greater than or equal to 6.5%)  using [data from the National Health and Nutrition Examination Survey (NHANES)](https://hbiostat.org/data/repo/nhgh.tsv). The project leverages various machine learning models, including logistic regression, decision tree, and random forest, to analyse and predict the probability that an individual currently has diabetes based on relevant health indicators.


## Project Structure

The project is organised as follows:

- `data/`: Directory to store the data.
    - `processed/`: Directory to store the processed data.
    - `raw/`: Directory to store the raw data.

- `notebooks/`: Jupyter notebooks for analysis and visualisation.
    - [`EDA.ipynb`](notebooks/EDA.ipynb): Jupyter notebook for exploratory data analysis.
    - [`modelling.ipynb`](notebooks/modelling.ipynb): Jupyter notebook for training, optimising, and evaluating machine learning models.

- [`README.md`](README.md): Project documentation.

- `results/`: Directory to store the results of model evaluations and visualisations.

- [`requirements.txt`](requirements.txt): File listing project dependencies.

- `src/`: Source code directory containing modules for data preparation and machine learning.
    - `data_preparation/`: Module for preparing the dataset.
        - [`data_ingestion.py`](src/data_preparation/data_ingestion.py): Module for data ingestion from the data source.
        - [`data_processing.py`](src/data_preparation/data_processing.py): Module for data processing.

    - `ml_models/`: Module for machine learning models.
        - [`decision_tree.py`](src/ml_models/decision_tree.py): Module for decision tree model.
        - [`logistic_regression.py`](src/ml_models/logistic_regression.py): Module for logistic regression model.
        - [`random_forest.py`](src/ml_models/random_forest.py): Module for random forest model.


## Getting Started

To run the notebooks and reproduce the results:

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/hl-n/diabetes-prediction-nhanes.git
    ```
2. **Set up the Virtual Environment:**
    ```bash
    # Navigate to the project directory
    cd diabetes-prediction-nhanes

    # Create and activate a virtual environment
    python -m venv diabetes_ml_env
    # On Windows:
    .\diabetes_ml_env\Scripts\activate
    # On macOS/Linux:
    source diabetes_ml_env/bin/activate

    # Upgrade pip and install required dependencies
    pip install --upgrade pip
    pip install -r requirements.txt
    ```
3. **Run the Notebooks:**

    ```bash
    # Navigate to notebooks directory
    cd notebooks
    # Run Jupyter Notebook
    jupyter notebook
    ```
    The above command will launch Jupyter Notebook in your web browser, and you'll be able to see the list of available notebooks.

    1. Click on the notebook you want to run:
        - Run EDA.ipynb for exploratory data analysis.
        - Run modelling.ipynb for developing, optimising, and interpreting the machine learning models.

    2. Inside the notebook, use the Jupyter interface to run individual cells or run all cells.

4. **Shut Down Jupyter Notebook:**
    1. Once you have completed your work and reviewed the notebook results, go back to the terminal where Jupyter Notebook is running.
    2. Press `Ctrl` + `C` to stop the Jupyter Notebook server.
    3. Confirm by typing `y` and pressing `Enter`.
    4. Close the browser tabs associated with Jupyter Notebook.


5. **Deactivate Virtual Environment:**
    ```bash
    deactivate
    ```

## Acknowledgements

### NHANES Dataset

The [dataset](https://hbiostat.org/data/repo/nhgh.tsv) used in this project is sourced from the National Health and Nutrition Examination Survey.

Data obtained from http://hbiostat.org/data courtesy of the Vanderbilt University Department of Biostatistics.


## Licensing

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Contributing

If you have suggestions for improvements, discover issues, or want to contribute to the project, feel free to:
- [Open an Issue](https://github.com/hl-n/diabetes-prediction-nhanes/issues)
- [Submit a Pull Request](https://github.com/hl-n/diabetes-prediction-nhanes/pulls)
