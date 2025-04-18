# SMARTe: Slot-based Method for Accountable Relational Triple extraction

Code for our paper [SMARTe: Slot-based Method for Accountable Relational Triple extraction]([https://arxiv.org/abs/2010.11304](http://arxiv.org/abs/2504.12816
).

<img width="828" alt="Architecture" src="https://github.com/Chen-XueWen/SMARTe/blob/main/Architecture.png" />

If you make use of this code in your work, please kindly cite the following paper:

```bibtex
@misc{tan2025smarteslotbasedmethodaccountable,
      title={SMARTe: Slot-based Method for Accountable Relational Triple extraction}, 
      author={Xue Wen Tan and Stanley Kok},
      year={2025},
      eprint={2504.12816},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2504.12816}, 
}
```

Before proceeding, ensure that all required packages are installed. The necessary dependencies for SMARTE are listed below:


## Requirements (Tested on the Version below)
| Package    | Version |
| -------- | ------- |
|python|3.11.3|
|pytorch|2.2.2|
|transformers|4.39.3|
|wandb|0.16.6|
|scipy|1.13.0|
|numpy|1.26.4|

Please create it under the environment named SMARTE.
```bash
conda create -n "SMARTE" python=3.11.3
conda activate SMARTE
```
Install the following dependencies and make adjustments based on your system:
```bash
conda install pytorch==2.2.2 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers==4.39.3
pip install wandb==0.16.6
pip install scipy==1.13.0
pip install numpy==1.26.4
```

To run the code:

a. Process the dataset
```bash
unzip 111_SMARTe_Slot_based_Method_f_Data.zip
mv data ./SMARTE
cd ./SMARTE
bash process_data.sh
```

b. Run the Experiments

**NYT Partial Matching**
```bash
python main-partial.py --data_type NYT-Partial --epochs 80 --encoder_lr 2e-5 --decoder_lr 8e-5 --num_iterations 6 --project NYTPartial-ACL --name smarte42 --seed 42
```
**WebNLG Partial Matching**
```bash
python main-partial.py --data_type WebNLG-Partial --epochs 340 --encoder_lr 2e-5 --decoder_lr 6e-5 --num_iterations 3 --project WebNLGPartial-ACL --name smarte42 --seed 42
```

**NYT Exact Matching**
```bash
python main-exact.py --data_type NYT-Exact --epochs 100 --encoder_lr 1e-5 --decoder_lr 6e-5 --num_iterations 3 --project NYTExact-ACL --name smarte42 --seed 42
```
**WebNLG Exact Matching**
```bash
python main-exact.py --data_type WebNLG-Exact --epochs 340 --encoder_lr 2e-5 --decoder_lr 6e-5 --num_iterations 3 --project WebNLGExact-ACL --name smarte42 --seed 42
```


Thank you! 
