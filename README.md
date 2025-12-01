# nanoTabPFN

Train your own small TabPFN in less than 500 LOC and a few minutes.

The purpose of this repository is to be a good starting point for students and researchers that are interested in learning about how TabPFN works under the hood.

Clone the repository, afterwards install dependencies via:
```
pip install numpy torch schedulefree h5py scikit-learn openml seaborn
```

### Our Code

- `model.py` contains the implementation of the architecture and a sklearn-like interface in less than 200 lines of code. 
- `train.py` implements a simple training loop and prior dump data loader in under 200 lines
- `experiment.ipynb` will recreate the experiment from the [paper](https://arxiv.org/pdf/2511.03634) (requires `pip install tabpfn==2.2.1`)


### Pretrain your own nanoTabPFN

To pretrain your own nanoTabPFN, you need to first download a prior data dump from [here](http://ml.informatik.uni-freiburg.de/research-artifacts/nanoTabPFN/300k_150x5_2.h5), then run `train.py`.

```bash
cd nanoTabPFN

# download data dump
curl http://ml.informatik.uni-freiburg.de/research-artifacts/nanoTabPFN/300k_150x5_2.h5 --output 300k_150x5_2.h5

python train.py
```

#### Step by Step explanation:

First we import our code from model.py and train.py
```py
from model import NanoTabPFNModel
from model import NanoTabPFNClassifier
from train import PriorDumpDataLoader
from train import train, get_default_device
```
Then we instantiate our model
```py
model = NanoTabPFNModel(
    embedding_size=96,
    num_attention_heads=4,
    mlp_hidden_size=192,
    num_layers=3,
    num_outputs=2
)
```
and our dataloader
```py
prior = PriorDumpDataLoader(
    "300k_150x5_2.h5",
    num_steps=2500,
    batch_size=32,
)
```
Now we can train our model:
```py
device = get_default_device()
model, _ = train(
    model,
    prior,
    lr = 4e-3,
    device = device
)
```
and finally we can instantiate our classifier:
```py
clf = NanoTabPFNClassifier(model, device)
```
and use its `.fit`, `.predict` and `.predict_proba`:
```py
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

clf.fit(X_train, y_train)
prob = clf.predict_proba(X_test)
pred = clf.predict(X_test)
print('ROC AUC', roc_auc_score(y_test, prob))
print('Accuracy', accuracy_score(y_test, pred))
```

### TFM-Playground

The nanoTabPFN repository is supposed to stay ultra small and simple, but we created another repository,
the [TFM-Playground](https://github.com/automl/TFM-Playground/) which we are building out to have a lot more features,
like regression, multiple prior interfaces, multiple architectures, ensembling of different pre-processings and more,
so check it out if you are interested!

### Paper

Check out our [paper](https://arxiv.org/pdf/2511.03634).
Please use the following BibTex to cite:

```
@article{pfefferle2025nanotabpfn,
  title={nanoTabPFN: A Lightweight and Educational Reimplementation of TabPFN},
  author={Pfefferle, Alexander and Hog, Johannes and Purucker, Lennart and Hutter, Frank},
  journal={arXiv preprint arXiv:2511.03634},
  year={2025}
}
```
