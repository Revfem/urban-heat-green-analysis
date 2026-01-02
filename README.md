# Urban Heat & Green Analysis

**Urban Heat & Green Analysis** is a Machine Learning and data analysis project that explores the *Urban Heat Island (UHI)* effect and the influence of green spaces using satellite imagery and spatial datasets.

This repository contains code, models, and notebooks to preprocess data, analyze spatial relationships between vegetation and localized surface temperature, and visualize results.

---

## 📂 Table of Contents

- [About the Project](#about-the-project)  
- [Dataset](#dataset)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Project Structure](#project-structure)  
- [Examples](#examples)  
- [Dependencies](#dependencies)  
- [Contributing](#contributing)  

---

## About the Project

Urban Heat Islands are urbanized areas that experience higher temperatures than their rural surroundings due to human activities and reduced vegetation. This project aims to:

- quantify the UHI effect from satellite data,
- relate surface temperature to vegetation indices,
- build models that can predict heat distribution given green coverage levels,
- provide visualizations and maps for interpretation.

---

## Dataset

Data used in this project includes:

- Satellite imagery (e.g., thermal bands),
- Vegetation indices (e.g., NDVI),
- Urban area masks or shapefiles,
- Any related CSV or raster files.

🔗 **Datasets are shared in the Google Drive folder:**  
https://drive.google.com/drive/folders/1sMxe-8vzlGqlBF_GYc6lNSmnJkfIq7Ea?usp=sharing

*You must download and place these in the `data/` directory before running analysis.*

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Revfem/urban-heat-green-analysis.git
cd urban-heat-green-analysis
```
2. Create your Python environment:
  ```bash
conda env create -f environment.yml
conda activate urban-heat-env
```
3. Install additional dependencies if needed:
  ```bash
  pip install -r requirements.txt
```

## Usage
Run preprocessing scripts:
```bash
python scripts/preprocess_data.py
```
Run analysis
```bash
python scripts/run_analysis.py
```
Visualize results
```bash
python scripts/visualize_results.py
```

## Example (run full workflow)
python scripts/preprocess_data.py --input data/raw --output data/processed
python scripts/run_analysis.py
python scripts/visualize_results.py

## Project Structure
```graphql
urban-heat-green-analysis/
├── data/                     # raw and processed datasets
├── notebooks/                # exploratory analysis notebooks
├── scripts/                  # python scripts for each workflow
├── outputs/                  # figures, maps, and models
├── environment.yml          
├── requirements.txt
├── .gitignore
└── README.md
```

## Dependencies
Major Python libraries used:
- numpy, pandas, xarray
- matplotlib, seaborn
- scikit‑learn
- rasterio, geopandas
…and more listed in requirements.txt or environment.yml.

## Contributing
Contributions are welcome!
Please open an issue or submit a pull request with clear descriptions of improvements.
If adding new scripts or notebooks, add documentation or comments so others understand the intent.






