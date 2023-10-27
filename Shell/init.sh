mkdir results
mkdir results/models/
mkdir results/dict/
mkdir results/logs/

pip install wandb
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install  dgl -f https://data.dgl.ai/wheels/cu118/repo.html
pip install  dglgo -f https://data.dgl.ai/wheels-test/repo.html

pip install torch_geometric
pip install pygraphviz
pip install numpy scipy rich tqdm matplotlib
pip install torchmetrics
pip install networkx
pip install ogb
# pip install \
#     --extra-index-url=https://pypi.nvidia.com \
#     cudf-cu11 dask-cudf-cu11 cuml-cu11 cugraph-cu11 cuspatial-cu11 cuproj-cu11 cuxfilter-cu11 cucim
