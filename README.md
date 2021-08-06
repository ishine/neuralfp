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
For comparison to the state-of-the-art, it is required to do the query matching on a large scale fingerprint dataset. For this purpose we use the ```fma_large.zip``` dataset published by [mdeff](https://github.com/mdeff/fma). As advised in the repository, the extraction of the dataset should be performed using [7zip](https://www.7-zip.org/download.html).
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
gdown https://drive.google.com/uc?id=12sCEepSwqbw6jbeJPO1CoTgRULYUj7Gg
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
python test.py --test_dir=/PATH/TO/DATASET --model_path=model/model_au_epoch_480.pth --clean=True
```



