

conda create --name e2exp python=3.6
source activate e2exp

conda install -y pytorch=1.3 torchvision cudatoolkit=10.0 -c pytorch
conda install -y -c conda-forge matplotlib
conda install -y scipy=1.5 pandas scikit-learn=0.23
conda install -y -c anaconda sqlite

#pip install e2cnn

mkdir tmp_e2cnn
cd tmp_e2cnn
git clone --single-branch --branch legacy_py3.6 https://github.com/QUVA-Lab/e2cnn
mv e2cnn/e2cnn ../e2cnn
cd ..
rm -rf tmp_e2cnn

