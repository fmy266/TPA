# Transferability Bound Theory: Exploring Relationship between Adversarial Transferability and Flatness (NeurIPS 2024)

This repository contains the source codes of TPA, accepted as poster at NeurIPS 2024. [Click here to access the preprint](https://arxiv.org/abs/2311.06423).

## 1. Preliminary

1) Please install dependent libraries listed in **requirements.txt** and ensure consistent versions.

2) **./toolkit** contains the source codes of FAA and **./FAA** is main evaluation scripts. Reproducing experiments see STEP2 (produce adversarial examples) and STEP3 (evaluate transferability).

## 2. Produce Adversarial Examples

Produce adversarial examples with our method and save the examples into "./our_advs". **save_dir** denotes the store path of produced adversarial examples.

> python produce_advs.py --device 0 --save_dir our_advs

## 3. Evaluate Transferability

Evaluate the attack performance of the examples against various models. Some target models can be automatically downloaded with Torch and Torchvision, e.g., VGG19. Some models are scattered and you need to manually download them (see below for download urls). Manually-downloaded models should be stored in **./FAA/defense_models/**. You can make suitable changes in **acc_validate.py** to evaluate specified models and we believe this is easy. By default, the code evaluates the target models presented in Table 1 (ResNet50, VGG19, etc.).

[Augmix Models](https://drive.google.com/file/d/1z-1V3rdFiwqSECz7Wkmn4VJVefJGJGiF/view)

[SIN Models](https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/6f41d2e86fc60566f78de64ecff35cc61eb6436f/resnet50_train_60_epochs-c8e5653e.pth.tar)

[SIN-IN Models](https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/60b770e128fffcbd8562a3ab3546c1a735432d03/resnet50_train_45_epochs_combined_IN_SF-2a0d100e.pth.tar)

[Linf and L2 Adversarially-trained Models](https://github.com/microsoft/robust-models-transfer)

> python acc_validate.py --device 0 --adv_dir our_advs

## Reference

If you find this repository helpful, please cite as:

```
@misc{fan2024transferabilityboundtheoryexploring,
      title={Transferability Bound Theory: Exploring Relationship between Adversarial Transferability and Flatness}, 
      author={Mingyuan Fan and Xiaodan Li and Cen Chen and Wenmeng Zhou and Yaliang Li},
      year={2024},
      eprint={2311.06423},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2311.06423}, 
}
```
