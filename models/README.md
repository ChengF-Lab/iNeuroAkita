# iNeuroAkita final model (7 fold models)

Multi-head (3 targets) Akita model, initialised from the published Akita weights and trained with
7-fold chromosome-disjoint cross-validation (model "A4" in the evaluation scripts).

```
models/
├── chrom_to_fold.tsv        chromosome -> fold whose model held it out (use for out-of-fold scoring)
└── foldK/
    ├── model_best.h5        weights (basenji / TensorFlow 2), best validation epoch
    ├── params.json          model + training parameters (identical in all folds = configs/params_iNeuroAkita.json)
    ├── targets.txt          target table used for training (head order; cooler paths are placeholders)
    ├── statistics.json      dataset geometry and number of train / valid / test windows
    └── split_chroms.txt     number of windows per split and chromosome
```

| | |
|---|---|
| Input | 1,048,576 bp one-hot DNA (hg38) |
| Output | upper triangle (diagonal offset 2) of a 448 x 448 map at 2,048 bp (512 bins, 32 cropped on each side) = 99,681 values x 3 targets, log(observed / expected) |
| Head order | 0 = NPC_P16, 1 = Mic_AD, 2 = Neuron_P16 |
| SuPreMo names (patched SuPreMo) | HFF = NPC_P16, H1hESC = Mic_AD, GM12878 = Neuron_P16 |

Test (held-out) chromosomes: fold1 10,17 · fold2 8,9,21 · fold3 1,11,20 · fold4 2,3,19 · fold5 4,5,18,22 ·
fold6 6,7,12,13 · fold7 14,15,16,X (validation chromosomes: `configs/folds.tsv`). chrY is in no fold.

Load in Python:

```python
import json
from basenji import seqnn
params = json.load(open("models/fold1/params.json"))
model = seqnn.SeqNN(params["model"]); model.restore("models/fold1/model_best.h5")
```

To score a sequence or variant without train/test leakage, use the fold model whose test set contains its chromosome
(`chrom_to_fold.tsv`); for SuPreMo set `SUPREMO_MODEL_DIR=models/foldK`.
