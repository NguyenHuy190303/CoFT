@echo off
call conda activate TS-TCC

set exp=HAR_experiment
set run=HAR
set dataset=HAR

set start=0
set end=0

for /L %%i in (%start%,1,%end%) do (
    python main.py --experiment_description %exp% --run_description %run% --seed %%i --selected_dataset %dataset% --training_mode "self_supervised"
    python main.py --experiment_description %exp% --run_description %run% --seed %%i --selected_dataset %dataset% --training_mode "train_linear_1p"
    python main.py --experiment_description %exp% --run_description %run% --seed %%i --selected_dataset %dataset% --training_mode "ft_1p"
    python main.py --experiment_description %exp% --run_description %run% --seed %%i --selected_dataset %dataset% --training_mode "gen_pseudo_labels"
    python main.py --experiment_description %exp% --run_description %run% --seed %%i --selected_dataset %dataset% --training_mode "SupCon"
    python main.py --experiment_description %exp% --run_description %run% --seed %%i --selected_dataset %dataset% --training_mode "train_linear_SupCon_1p"
)
pause