# UNIFY: Unified Learning with One Model via Batch Norm Switching

**A memory-efficient framework for joint multi-dataset learning using a single neural network.**

UNIFY is an architecture-flexible deep learning framework that enables a single Convolutional Neural Network to jointly learn and perform inference across multiple image classification datasets through **Batch Normalization Switching (BNS)** and dataset-specific classification heads.

Instead of independently training and deploying a model for each dataset, UNIFY shares a common backbone while maintaining lightweight dataset-specific components. This reduces model redundancy and deployment memory requirements while providing comparable or
superior performance compared to independently trained, dataset-specific models.

## Key Features

- **Unified multi-dataset learning:** Trains one model jointly across multiple datasets.
- **Batch Normalization Switching:** Maintains independent BN statistics and parameters for each dataset.
- **Dataset-specific classification heads:** Supports datasets with different numbers of classes.
- **Shared backbone:** Re-uses convolutional parameters across datasets.
- **Memory-efficient deployment:** Consolidates multiple independently trained models into one UNIFIED model.
- **Architecture-flexible design:** Applicable to various BN-based CNN architectures, without fundamentally changing their backbones.

## Framework Overview

UNIFY has three components:
1. **Shared backbone:** A common feature extractor optimized jointly across datasets.
2. **Switchable Batch Normalization:** Dataset-specific BN configurations selected using the dataset identifier.
3. **Dataset-specific classification heads:** Independent output layers for each dataset's classification task.

During training and inference, the dataset identifier selects the corresponding BN configuration and classification head. The backbone parameters remain shared.

## Datasets

| Dataset | Classes | Original image resolution |
|---|---:|---|
| MNIST | 10 | 28 × 28 × 1 |
| Fashion-MNIST | 10 | 28 × 28 × 1 |
| CIFAR-10 | 10 | 32 × 32 × 3 |
| CIFAR-100 | 100 | 32 × 32 × 3 |
| Tiny-ImageNet | 200 | 64 × 64 × 3 |

## Evaluated CNN Architectures

- SqueezeNet 1.1
- WideResNet-40-2
- WideResNet-16-8
- ResNet-18
- ResNet-34
- ResNet-50

## Classification Performance

Classification accuracy (%) of the UNIFIED models jointly trained across all five datasets:

| UNIFIED model | MNIST | Fashion-MNIST | CIFAR-10 | CIFAR-100 | Tiny-ImageNet |
|---|---:|---:|---:|---:|---:|
| SqueezeNet 1.1 | 99.41 | 93.12 | 91.06 | 69.18 | 56.76 |
| WideResNet-40-2 | 99.62 | 94.22 | 94.39 | 75.29 | 61.13 |
| WideResNet-16-8 | 99.66 | 94.44 | 96.00 | 79.41 | 65.19 |
| ResNet-18 | 99.56 | 94.84 | 95.44 | 78.21 | 66.22 |
| ResNet-34 | 99.53 | 94.84 | 96.12 | 80.32 | 67.67 |
| ResNet-50 | 99.63 | 94.48 | 95.64 | 80.22 | 69.67 |

**Key result:** UNIFIED models outperform independently trained dataset-specific counterparts in **21 of 30 architecture–dataset combinations (70% of the evaluated combinations)**.

## Memory Efficiency

UNIFY reduces deployment memory by replacing five independent models with a single UNIFIED model.

| Architecture | Five independent models | UNIFIED model | Size reduction |
|---|---:|---:|---:|
| SqueezeNet 1.1 | 15.70 MB | 3.82 MB | 4.11× |
| WideResNet-40-2 | 45.85 MB | 9.68 MB | 4.74× |
| WideResNet-16-8 | 221.45 MB | 44.88 MB | 4.93× |
| ResNet-18 | 225.80 MB | 45.87 MB | 4.92× |
| ResNet-34 | 428.30 MB | 86.71 MB | 4.94× |
| ResNet-50 | 479.80 MB | 99.11 MB | 4.84× |

*Memory footprints are reported for Float32 model parameters.*

For example, UNIFIED ResNet-34 reduces deployment memory from **428.30 MB** (five separate models) to **86.71 MB** (one unified model), a **4.94× reduction**, with approximately **0.61% additional parameters** compared to one original ResNet-34 architecture.

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Epochs | 200 |
| Batch size | 64 |
| Optimizer | SGD |
| Initial learning rate | 0.1 |
| Momentum | 0.9 |
| Weight decay | 5 × 10⁻⁴ |
| Learning-rate schedule | Cosine annealing |
| Loss | Cross-entropy |
| Data Augmentation | Nill |

The dataloaders of smaller datasets are repeated to match the length of the Tiny-ImageNet dataloader. Independent-model baselines use consistent training settings.

## Ablation: Why Batch Normalization Switching Matters

Ablation experiments show that using one shared BN configuration instead of dataset-specific BN configurations can substantially degrade accuracy, even when separate classification heads are used. This underscores the importance of Batch Norm Switching (BNS) for UNIFIED learning.

## Future Directions

- Evaluate larger datasets, including ImageNet.
- Extend UNIFY to object detection and semantic segmentation.
- Investigate automatic dataset identification and BN selection.
- Explore continual integration of new datasets.
- Combine UNIFY with model compression techniques such as pruning, quantization, and knowledge distillation.

## Citation

If UNIFY is useful in your research, please cite the associated paper:
```bibtex
@article{tareen2026unify,
  title={UNIFY: Unified Learning with One Model via Batch Norm Switching},
  author={Tareen, Shaharyar Ahmed Khan and Yuan, Xiaojing},
  journal={arXiv preprint arXiv:2611.28746},
  year={2026}
}
```
## Authors

- **Shaharyar Ahmed Khan Tareen** — University of Houston, TX, USA
- **Xiaojing Yuan** — University of Houston, TX, USA

## License

See [LICENSE](LICENSE) for the repository's licensing terms.

---

**UNIFY — One Model, Multiple Datasets, Shared Learning, Lower Memory Footprint, and Efficient Deployment.**
