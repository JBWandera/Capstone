# Training on Modal

Free Colab needs a person to keep a browser tab open. Modal runs the same
training code on a cloud GPU from one command, so runs continue unattended.

One time setup (done once per machine):

```bash
pip install modal
modal token set --token-id <id> --token-secret <secret>
modal secret create huggingface HF_TOKEN=<hugging face token with read access>
```

Run an experiment from `configs/experiments.yaml` (it resumes if restarted):

```bash
modal run --detach modal_app/train_modal.py --exp E2_rfdetr_nano
```

Checkpoints, learning curves and predictions land in the Modal volume
`afroviewz` under `runs/<exp>/`. Copy them locally for notebook 07:

```bash
modal volume get afroviewz runs/E2_rfdetr_nano ./runs/E2_rfdetr_nano
```

Cost guide: an L4 GPU is about USD 0.80 per hour on Modal; one run is about
1 to 1.5 hours.
