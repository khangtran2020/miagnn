mkdir results
mkdir results/models/
mkdir results/dict/
mkdir results/logs/

pip install wandb
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

pip install  dgl -f https://data.dgl.ai/wheels/cu118/repo.html
pip install  dglgo -f https://data.dgl.ai/wheels-test/repo.html

pip install torch_geometric
pip install pygraphviz
pip install numpy scipy rich tqdm matplotlib
pip install torchmetrics
pip install networkx
pip install ogb
