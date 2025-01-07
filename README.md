
## Requirements

### Python Environment
- Python 3.x
- Install dependencies via `requirements.txt`:
  ```bash
  pip install -r requirements.txt
  ```

### CUDA and PyTorch
- Install PyTorch and CUDA support:
  ```bash
  conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch
  ```
### Running the Pipeline

For Linux:
On Linux

To run the experiments on Linux, use the provided shell scripts:

Sleep-EDF experiments:
 ```bash
./sleepedf.sh
 ```
Epilepsy experiments:
 ```bash
./epilepsy.sh
 ```
HAR experiments:
 ```bash
  ./har.sh
  ```
For windows:
```bash
run.bat
```

## Datasets
### Download Links
- [Sleep-EDF](https://gist.github.com/emadeldeen24/a22691e36759934e53984289a94cb09b)
- [HAR](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones)
- [Epilepsy](https://archive.ics.uci.edu/ml/datasets/Epileptic+Seizure+Recognition)  
---

## Training Instructions

### Modes
- **Random Initialization**
- **Supervised training**
- **Self-supervised training**
- **Fine-tuning**
- **Training a linear classifier**

### Example Command
```bash
python main.py --experiment_description exp1 --run_description run_1 --seed 123 --training_mode self_supervised --selected_dataset HAR
```

---

## Results
- Experiment logs and classification reports are stored in the `experiments_logs` folder by default.

---
### Directory Structure
```bash
.gitattributes
config_files/
    __init__.py
    Epilepsy_Configs.py
    HAR_Configs.py
    pFD_Configs.py
    sleep_Configs.py
data/
    epilepsy/
    HAR/
    sleep/
data_preprocessing/
    epilepsy/
    sleep-edf/
    uci_har/
dataloader/
    augmentations.py
    dataloader.py
epilepsy.sh
experiments_logs/
    epilepsy_experiment/
    HAR_experiment/
    sleepEDF_experiment/
har.sh
main.py
misc/
models/
    attention.py
requirements.txt
run.bat
sleepedf.sh
trainer/
utils.py
```