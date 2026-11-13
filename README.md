# AfroViewz

**Pothole detection and road condition mapping, tested on Ugandan roads.**

AfroViewz finds potholes in photos, videos and a live phone camera, grades how severe each one is, and puts them on a map that road engineers can export. It started as a data science capstone (archived in [`archive/capstone-v1`](archive/capstone-v1)) and is being rebuilt as a product for a presentation on 1 December 2026.

> Status: **in progress**. Data stage done (notebooks 01 to 03 run, outputs committed). Next: training on Colab.

## The dataset so far (`afroviewz-v1`)

| | images | potholes | pothole free |
|---|---|---|---|
| train | 4,769 | 6,683 | 1,872 |
| valid | 529 | 919 | 160 |
| test | 594 | 840 | 238 |
| **Uganda holdout** (to label) | 3,200 candidates from Kampala, Entebbe, Masaka, Mbale, Lira | | |

Built from 7,452 cleaned source images (after dropping 1,622 baked-in augmented copies and capping pothole free dashcam frames at 2,500): 1,560 exact repeats removed (pixel checked), 267 near duplicate clusters each kept inside one split. Rebuilding gives the same split image for image (checked against `configs/manifests/afroviewz-v1_splits.csv`).

Main EDA finding: at a 560 px model input, **98% of the dashcam potholes are under 32 px** (median 9 px), while the phone photo sources are mostly medium or large. This, more than any training setting, explains the v1 model's 0.04 small-object AP.

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
| 03 | Exploratory data analysis | Data understanding | [Colab](https://colab.research.google.com/github/JBWandera/Capstone/blob/afroviewz/notebooks/03_eda.ipynb) |
| 04 | Uganda holdout labelling | Test set construction | [Colab](https://colab.research.google.com/github/JBWandera/Capstone/blob/afroviewz/notebooks/04_uganda_holdout.ipynb) |
| 05 | Baselines (v1 model, YOLO26) | Modelling | [Colab](https://colab.research.google.com/github/JBWandera/Capstone/blob/afroviewz/notebooks/05_baselines.ipynb) |
| 06 | Training RF-DETR (resumable) | Modelling | [Colab](https://colab.research.google.com/github/JBWandera/Capstone/blob/afroviewz/notebooks/06_train_rfdetr.ipynb) |
| 07 | Evaluation and error analysis | Evaluation | [Colab](https://colab.research.google.com/github/JBWandera/Capstone/blob/afroviewz/notebooks/07_evaluation.ipynb) |
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
