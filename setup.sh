# Update conda environment.
conda update -n base conda;
conda update --all;

# Create conda environment.
conda create -n kals python=3.8 -y;
conda activate kals;
conda install -c conda-forge notebook -y;
python -m ipykernel install --user --name kals;

# install pytorch (1.9.0 version)
conda install numpy -y;
# conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch -y;
# Only CPU
# conda install pytorch torchvision cudatoolkit=10.2 -c pytorch -y;
# With GPU
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install packages from conda-forge.
conda install -c conda-forge matplotlib -y;
conda install -c conda-forge pandas networkx scipy -y;
conda install conda-forge::imageio

# Install pip packages
pip3 install -U scikit-learn;

# Create environment.yml.
conda env export > environment.yml;
