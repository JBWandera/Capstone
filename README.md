# AfroViewz

**Pothole detection and road condition mapping, tested on Ugandan roads.**

AfroViewz finds potholes in photos, videos and a live phone camera, grades how severe each one is, and puts them on a map that road engineers can export. It started as a data science capstone (archived in [`archive/capstone-v1`](archive/capstone-v1)) and is being rebuilt as a product for a presentation on 1 December 2026.

> Status: **in progress**, data stage. See the plan below.

## What is different from the v1 capstone

| | v1 capstone | AfroViewz |
|---|---|---|
| Data | One Roboflow dataset, its own split | Several sources incl. South African dashcam data, near duplicates removed, leak free grouped split |
| Ugandan test | None | Holdout set of Ugandan street photos (Mapillary), never trained on |
| Model | One RF-DETR Base run | RF-DETR Nano / Small / Medium compared, a YOLO baseline, tuning, a segmentation model for severity |
| Evaluation | mAP after a 0.5 confidence cut (understated) | COCO mAP done right, tuned threshold, confidence intervals, speed on a real phone |
| Product | Notebook only | Web app, installable phone app, API, pothole map, docs site |

## Repository layout

```
configs/            data sources and training settings
src/afroviewz/      the Python package (data, training, evaluation, export)
notebooks/          the data science workflow, one notebook per stage, runs on Colab
tests/              unit tests (pytest)
docs/               documentation site (Quarto) and presentation
archive/capstone-v1 the original capstone notebook, README and proposal
```

## The data science workflow

| # | Notebook | Stage | Open |
|---|---|---|---|
| 01 | Data acquisition | Collect and convert every source | [Colab](https://colab.research.google.com/github/JBWandera/Capstone/blob/afroviewz/notebooks/01_data_acquisition.ipynb) |
| 02 | Cleaning, de-duplication, split | Data preparation | [Colab](https://colab.research.google.com/github/JBWandera/Capstone/blob/afroviewz/notebooks/02_data_cleaning_and_split.ipynb) |
| 03 | Exploratory data analysis | Data understanding | coming |
| 04 | Uganda holdout labelling | Test set construction | coming |
| 05 | Baselines | Modelling | coming |
| 06 | Training and tuning | Modelling | coming |
| 07 | Evaluation and error analysis | Evaluation | coming |
| 08 | Optimization and export | Deployment | coming |

## Data sources

| Source | Region | Use | License |
|---|---|---|---|
| [Road Pothole Images](https://scholar.sun.ac.za/handle/10019.1/97191), Stellenbosch University | South Africa | train | CC0 1.0 |
| [pothole-detection-6vpbs](https://universe.roboflow.com/project-etnb9/pothole-detection-6vpbs) (Roboflow Universe) | Mixed | train, masks | CC BY 4.0 |
| [potholes-detection-jbnou](https://universe.roboflow.com/muhammadmoin-arxtl/potholes-detection-jbnou) (v1 capstone data) | Mixed | train, comparison | CC BY 4.0 |
| [Mapillary](https://www.mapillary.com) street photos | Uganda | holdout test only | CC BY-SA 4.0, credited per image |

## Running the code

```bash
pip install -e ".[dev]"
pytest
```

Training and data download run on Google Colab (free tier); open a notebook from the table above.

## License

Code: Apache 2.0. Datasets keep their own licenses (see above).

## Author

Jude Bwire Wandera
