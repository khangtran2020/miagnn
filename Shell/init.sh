mkdir results
mkdir results/models/
mkdir results/dict/
mkdir results/logs/

pip install wandb loguru numpy scipy rich tqdm matplotlib 
pip install torchmetrics
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
