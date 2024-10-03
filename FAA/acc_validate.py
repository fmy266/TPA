import torch, argparse, torchvision, timm, os
from PIL import Image
# from robustness import datasets, model_utils
from produce_advs import SubsetImageNet
import warnings


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
def unnormalize(t):
    t[:, 0, :, :] = (t[:, 0, :, :]*std[0] + mean[0])
    t[:, 1, :, :] = (t[:, 1, :, :]*std[1] + mean[1])
    t[:, 2, :, :] = (t[:, 2, :, :]*std[2] + mean[2])
    return t

class AdvDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        super(AdvDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.img_path = os.listdir(root)

    def __getitem__(self, item):
        filepath = os.path.join(self.root, self.img_path[item])
        sample = Image.open(filepath, mode='r')

        if self.transform:
            sample = self.transform(sample)

        label = int(self.img_path[item].split(".")[0].split("_")[1])

        return sample, label

    def __len__(self):
        return len(self.img_path)


@torch.no_grad()
def validate(val_loader, model, robustness_flag, device):
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()
    for i, raw_data in enumerate(val_loader):

        input = raw_data[0]
        target = raw_data[1]

        input = input.to(device)
        target = target.to(device)

        if robustness_flag == 1:
            input = unnormalize(input)
            output, _ = model(input)
        else:
            output = model(input)

        with torch.no_grad():
            probs = torch.softmax(output, dim=1)

        prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

    return top1.avg, top5.avg


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


train_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument("--adv_dir", type=str, default="our_advs")
    parser.add_argument("--device", type=int, default=5)
    parser.add_argument("--target", type=bool, default=False)
    args = parser.parse_args()
    device = torch.device("cuda:{}".format(args.device))

    weights = torchvision.models.Swin_T_Weights.DEFAULT
    model_zoo = {
        "resnet50": torchvision.models.resnet50(pretrained=True).to(device).eval(),
        "densenet121": torchvision.models.densenet121(pretrained=True).to(device).eval(),
        "mobilenet": torchvision.models.mobilenet_v3_small(pretrained=True).to(device).eval(),
        "efficientnet": torchvision.models.efficientnet_b0(pretrained=True).to(device).eval(),
        "vgg19": torchvision.models.vgg19_bn(pretrained=True).to(device).eval(),
        "inception": torchvision.models.inception_v3(pretrained=True).to(device).eval(),
        "regnet": torchvision.models.regnet_x_16gf(pretrained=True).to(device).eval(),
        "wideresnet50": torchvision.models.wide_resnet50_2(pretrained=True).to(device).eval(),
        "swin_t": torchvision.models.swin_t(weights=weights).to(device).eval(),
        "squeezenet": torchvision.models.squeezenet1_1(pretrained=True).to(device).eval(),
        "mnasnet": torchvision.models.mnasnet1_0(pretrained=True).to(device).eval(),
        "vit": torchvision.models.vit_b_16(pretrained=True).to(device).eval(),
        "convnext": torchvision.models.convnext_tiny(pretrained=True).to(device).eval(),
        "shufflenet": torchvision.models.shufflenet_v2_x1_0(pretrained=True).to(device).eval(),
    }

    adv_loader = torch.utils.data.DataLoader(AdvDataset(f"./{args.adv_dir}", transform=train_transform), batch_size=16, Shuffle=False)
    clean_loader = torch.utils.data.DataLoader(SubsetImageNet(transform=train_transform, targeted=False), batch_size=16)


    for model_name in model_zoo.keys():
        model = model_zoo[model_name]
        adv_acc, _ = validate(adv_loader, model, 0, device)
        clean_acc, _ = validate(clean_loader, model, 0, device)
        print(f"target model: {model_name}\tclean accuracy:{clean_acc:.2f}%\tattack success rate:{100-adv_acc:.2f}%")
