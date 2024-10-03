import torch, csv, os, sys, argparse, pretrainedmodels, torchvision, warnings
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms.functional import to_pil_image
# from robustness import datasets, defaults, model_utils, train
# from robustness.tools import helpers
from densenet import *
from resnet import *
from vgg import *
sys.path.append("..")
from toolkit.adv import attack


def load_model(archstr, device):

    net = pretrainedmodels.__dict__[archstr](num_classes=1000, pretrained=None)
    net = torch.nn.DataParallel(net, device_ids=[device,])
    ckpt = torch.load("./model/" + archstr + ".pth")
    if "model_state_dict" in ckpt:
        net.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        net.load_state_dict(ckpt)

    modelsdir = {
        'resnet50': resnet50(),
        'resnet152': resnet152(),
        'densenet121': densenet121(),
        'densenet201': densenet201(),
    }

    model = modelsdir[archstr]
    model = nn.DataParallel(model, device_ids=[device,])
    model_dict = model.state_dict()
    pre_dict = net.state_dict()
    state_dict = {k: v for k, v in pre_dict.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict, strict=False)
    model.eval()

    return model


def load_ground_truth(csv_filename):
    image_id_list = []
    label_ori_list = []
    label_tar_list = []

    with open(csv_filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            image_id_list.append( row['ImageId'] )
            label_ori_list.append( int(row['TrueLabel']) - 1 )
            label_tar_list.append( int(row['TargetClass']) - 1 )

    return image_id_list ,label_ori_list ,label_tar_list


class SubsetImageNet(Dataset):
    def __init__(self, root="./target_data/images", transform=None, targeted=False):
        super(SubsetImageNet, self).__init__()
        self.root = root
        self.transform = transform
        image_id_list, label_ori_list, label_tar_list = load_ground_truth('./target_data/images.csv')
        img_path = [img+".png" for img in image_id_list]
        self.img_path = [item for item in img_path if 'png' in item]
        if targeted:
            self.mapping = {i:j for i,j in zip(image_id_list, label_tar_list)}
        else:
            self.mapping = {i:j for i,j in zip(image_id_list, label_ori_list)}

    def __getitem__(self, item):
        filepath = os.path.join(self.root, self.img_path[item])
        sample = Image.open(filepath, mode='r')

        if self.transform:
            sample = self.transform(sample)

        label = self.mapping[self.img_path[item].split(".")[0]]

        return sample, label

    def __len__(self):
        return len(self.img_path)


train_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


def unnormal_func(image):
    image[:, 0, :, :] = image[:,0, :, :] * 0.229 + 0.485
    image[:, 1, :, :] = image[:, 1, :, :] * 0.224 + 0.456
    image[:, 2, :, :] = image[:, 2, :, :] * 0.225 + 0.406


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=7)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--noise", type=float, default=16.)
    parser.add_argument("--copies", type=int, default=10)
    parser.add_argument("--budget", type=float, default=16.)
    parser.add_argument("--target", type=bool, default=False)
    parser.add_argument("--proxy", type=str, default="resnet50")
    parser.add_argument("--save_dir", type=str, default="our_advs")
    args = parser.parse_args()
    device = torch.device("cuda:{}".format(args.device))
    epsilon = 0.27 / 16 * args.budget # approxi 4 6 8 10 12 14 16/255

    dataset = SubsetImageNet(transform=train_transform, targeted=False)
    mini_imagenet = torch.utils.data.DataLoader(dataset, batch_size=16)

    proxy_model_zoo = {
        "resnet50": torchvision.models.resnet50(pretrained=True).to(device).eval(),
        # "densenet121": torchvision.models.densenet121(pretrained=True).to(device).eval(),
        # "resnet152": torchvision.models.resnet152(pretrained=True).to(device).eval(),
        # "densenet201": torchvision.models.densenet201(pretrained=True).to(device).eval(),
    }

    eval_attack = attack.TAEFEP(epsilon=epsilon, step_size=epsilon / 10, device=device, iter_num=20,
                                  alpha=args.alpha, noise_magn=args.noise, forward_step_size=args.step,
                                  copies=args.copies)

    if not os.path.exists(f"./{args.save_dir}"):
        os.mkdir(f"./{args.save_dir}")

    proxy_model = load_model(args.proxy, args.device)

    save_idx = 0
    for data, label in mini_imagenet:

        data, label = data.to(device), label.to(device)

        adv_noise = eval_attack.produce_adv(data, label, proxy_model, torch.nn.CrossEntropyLoss()).detach().clone()

        adv_img = data + adv_noise

        unnormal_func(adv_img)
        unnormal_func(data)

        adv_img.clamp_(0., 1.)

        assert (data-adv_img).abs().max().item() <= 16. / 255.

        for idx in range(label.size(0)):
            to_pil_image(adv_img[idx]).save(f"./{args.save_dir}/{save_idx}_{label[idx].item()}.png")
            save_idx += 1