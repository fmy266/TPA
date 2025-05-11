# Transferability Bound Theory: Exploring Relationship between Adversarial Transferability and Flatness (NeurIPS 2024)

**Update (2025-05-11)**: Recently, some have reported that the code in this repository achieves only 60-70% ASR on their servers, while some individuals successfully reproduce the results. I can currently only attribute this discrepancy to environmental differences, as this issue has exceeded my capabilities (you can see the Issues for more details). Below, I've detailed my experimental environment, which may help with reproducibility. I wish that someone could figure out what causes these variations.

Some observations that might be helpful:
- The code ran successfully on the company servers last year
- The code also works well on my lab servers
- System versions and GPU hardware are unlikely to be the root cause
- Please prioritize aligning the remaining environment components, including CUDA version

For researchers using TPA as a baseline but unable to fully reproduce results due to unknown factors:
1. Find collaborators who can successfully reproduce the results
2. Use my pre-generated adversarial examples (ResNet50, 10 iterations, perturbation budget $\epsilon$=16/255)

Reproduction Environment:
- Python: 3.11.5
- Dependencies: requirements.txt
- GPU: NVIDIA 4090
- CUDA: 12.2
- OS: CentOS 7

Please note that minor version differences in dependencies might still affect reproducibility. If you discover any specific environmental factors affecting the results, please leave a message in our issue.

---

This repository contains the source codes of TPA (Theoretically Provable Attack), accepted as a poster at NeurIPS 2024. TPA optimizes a surrogate of the derived bound craft adversarial examples. The crafted adversarial examples can transfer across state-of-the-art normal and defense models. [Click here to access the preprint for more information of TPA](https://arxiv.org/abs/2311.06423).

## 1. Preliminary

1) Please install dependent libraries listed in **requirements.txt** and ensure consistent versions.

2) **./toolkit** contains the source codes of FAA and **./FAA** is main evaluation scripts. Reproducing experiments see STEP2 (produce adversarial examples) and STEP3 (evaluate transferability).

3) Regarding the dataset, you can refer to [this link](https://github.com/cleverhans-lab/cleverhans/tree/master/cleverhans_v3.1.0/examples/nips17_adversarial_competition/dataset). Alternatively, you can extract some images from ImageNet.

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
@inproceedings{fan2025bad,
      title={Transferability Bound Theory: Exploring Relationship between Adversarial Transferability and Flatness}, 
      author={Mingyuan Fan and Xiaodan Li and Cen Chen and Wenmeng Zhou and Yaliang Li},
  booktitle    = {Proc. of NeurIPS},
  year         = {2024},
}
```
