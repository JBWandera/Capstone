# AfroViewz

**Pothole detection and road condition mapping, tested on Ugandan roads.**

AfroViewz finds potholes in photos, videos and a live phone camera, grades how severe each one is, and puts them on a map that road engineers can export. It started as a data science capstone (archived in [`archive/capstone-v1`](archive/capstone-v1)) and is being rebuilt as a product for a presentation on 1 December 2026.

> Status: **in progress**. Data stage done; first training round done (notebook 07). Round 2 (input resolution, road band crop) running on Modal.

## Results so far (test split, 592 images, 846 potholes)

| Run | Model | Input | mAP50 (95% CI) | mAP50:95 | Precision | Recall | ms per image |
|---|---|---|---|---|---|---|---|
| E5 | RF-DETR Small, road band crop | 704 | **0.730** (0.689 to 0.778) | **0.413** | 0.755 | 0.701 | 12.4 |
| E5 | RF-DETR Medium, road band crop | 576 | 0.700 (0.653 to 0.747) | 0.387 | 0.709 | 0.695 | 8.6 |
| E4 | RF-DETR Small, road band crop | 512 | 0.694 (0.655 to 0.742) | 0.377 | 0.743 | 0.651 | 8.5 |
| E3 | RF-DETR Small | 704 | 0.667 (0.624 to 0.715) | 0.364 | 0.725 | 0.619 | 13.9 |
| E5 | RF-DETR Nano, road band crop (**app model**) | 384 | 0.643 (0.599 to 0.694) | 0.344 | 0.724 | 0.599 | 4.6 |
| E2 | RF-DETR Medium | 576 | 0.628 (0.584 to 0.679) | 0.334 | 0.700 | 0.602 | 9.7 |
| E1 | YOLO26 small (comparison only) | 640 | 0.600 (0.556 to 0.648) | 0.313 | 0.641 | 0.585 | 6.9 |
| E2 | RF-DETR Small | 512 | 0.578 (0.542 to 0.627) | 0.313 | 0.659 | 0.582 | 8.0 |
| E2 | RF-DETR Nano | 384 | 0.522 (0.476 to 0.575) | 0.273 | 0.664 | 0.511 | 5.5 |

Precision and recall at the threshold chosen on the validation split; speed on an NVIDIA L4.

Round 1 showed the dashcam frames were the weak spot (0.46 mAP50 for RF-DETR Small) because their potholes are tiny. Round 2 changed one thing at a time on the same Small model:

* **Road band crop (E4):** dashcam frames are cut to the strip of road between 35% and 75% of the image height before training and inference, so the potholes take up more of the model's input. Dashcam mAP50 rose from 0.46 to 0.66 and overall mAP50 from 0.578 to 0.694, with confidence intervals that no longer overlap. It costs nothing in speed.
* **Higher input resolution (E3, 704 px):** dashcam mAP50 rose to 0.61 and overall to 0.667, but each image takes 75% longer.

Round 3 kept the crop everywhere:

* **Crop plus 704 px (E5 Small):** the two gains add up. Best model so far at 0.730 mAP50, dashcam 0.71, small pothole recall 0.67 (from 0.47).
* **Medium with the crop:** 0.700, no better than Small with the crop at 512, so model size is not the bottleneck.
* **Nano with the crop:** 0.643, up from 0.522, better than round 1's Medium while running in 4.6 ms. This is the model the app ships.

## The dataset so far (`afroviewz-v1`)

| | images | potholes | pothole free |
|---|---|---|---|
| train | 4,766 | 6,689 | 1,872 |
| valid | 534 | 907 | 160 |
| test | 592 | 846 | 237 |
| **Uganda holdout** (to label) | 3,200 candidates from Kampala, Entebbe, Masaka, Mbale, Lira | | |

Built from 7,452 cleaned source images (after dropping 1,622 baked-in augmented copies and capping pothole free dashcam frames at 2,500): 1,560 exact repeats removed (pixel checked), 267 near duplicate clusters each kept inside one split. The split is frozen in `configs/manifests/afroviewz-v1_frozen_splits.csv` (the exact split the models were trained on); rebuilding reproduces it image for image.

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
