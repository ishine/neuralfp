# neuralfp Audio Fingerprinter
A PyTorch audio fingerprinting framework inspired from "Neural Audio Fingerprint for High-Specific Audio Retrieval Based on Contrastive Learning" by S. Chang et al.

## Creating fingerprint database using neuralfp
Fingerprints can be generated from an audio dataset using the pre-trained model or framework can be trained in the dataset provided by the user. 

### Setting up the environment and the dataset
For using the framework, clone the repository:
```shell

git clone https://github.com/chymaera96/neuralfp.git
cd neuralfp
```

Install the requirements in a virtual environment:
```shell

python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```
For comparison to the state-of-the-art, it is required to do the query matching on a large scale fingerprint dataset. For this purpose we use the ```fma_large.zip``` as the test dataset, which is published by [mdeff](https://github.com/mdeff/fma). As advised in the repository, the extraction of the test dataset should be performed using [7zip](https://www.7-zip.org/download.html).
```shell
cd data
curl -O https://os.unil.cloud.switch.ch/fma/fma_large.zip
echo "497109f4dd721066b5ce5e5f250ec604dc78939e  fma_large.zip"    | sha1sum -c -
7z e fma_large.zip

```
Instead of the above, validation can be performed on a smaller dataset such as ```fma_medium.zip```
``` shell
cd data
curl -O https://os.unil.cloud.switch.ch/fma/fma_medium.zip
echo "c67b69ea232021025fca9231fc1c7c1a063ab50b  fma_medium.zip"   | sha1sum -c -
7z e fma_medium.zip
```

### Computing fingerprint dataset
```shell
cd ../model
gdown https://drive.google.com/uc?id=1udaJj13tZnj2rHXB8OUDB2ctIjrkY0Zr
cd ..
```
To specify CUDA device ```1``` for example, you would set the ```CUDA_VISIBLE_DEVICES``` using
```shell
export CUDA_VISIBLE_DEVICES=1
```
If you want to specify more than one device, use
```shell
export CUDA_VISIBLE_DEVICES=0,1
```
Finally, run the scripts to compute fingerprints
```shell
python test.py --test_dir=/PATH/TO/TEST/DATASET --model_path=model/model_au_epoch_480.pth --clean=True
```
## Query Matching

Query searching and matching is performed using the ```faiss``` [library](https://github.com/facebookresearch/faiss). Let us create a noisy query dataset from a subset of the test dataset.
```shell
cd data
gdown https://drive.google.com/uc?id=1CDUoyK4nHdpPyuOG7zJ6MLcRY4F_2jml
unzip augmentation_datasets.zip
cd ..
python create_query_data.py --length 5 --test_dir=/PATH/TO/TEST/DATASET --noise_dir=data/augmentatation_datasets/noise
```
Calculating top-1 hit rate on the query dataset
```shell
python test.py --fp_path=fingerprints/fma_large_au.pt --query_dir=data/fma_5sec_2K --model_path=model/model_au_epoch_480.pth --eval=True
```
or 
```
python test.py --fp_path=fingerprints/fma_medium_au.pt --query_dir=data/fma_5sec_2K --model_path=model/model_au_epoch_480.pth --eval=True
```

NOTE: The repository is currently in development. The framework would be soon made available as a package along with the performance report against the state-of-the-art. 



