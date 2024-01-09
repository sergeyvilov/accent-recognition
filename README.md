# MCV-spoken-language-recognition

conda env create -f environment.yml
conda activate accents
brew install libsndfile
flask run



conda create -n accents python=3.10.4 librosa=0.10.1 pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 flask==3.0.0 -c pytorch -c conda-forge

