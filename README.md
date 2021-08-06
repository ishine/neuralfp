# neuralfp Audio Fingerprinter
A PyTorch audio fingerprinting framework inspired from "Neural Audio Fingerprint for High-Specific Audio Retrieval Based on Contrastive Learning" by S. Chang et al.

## Creating fingerprint database using neuralfp
Fingerprints can be generated from a audio dataset using the pre-trained model or framework can be trained in the dataset provided by the user. The trained model is available for download here [LINK].
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
Using the FMA dataset to create the fingerprint dataset:
```shell

