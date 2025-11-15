import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision import datasets

from core.dataset import CelebAHQDataset, BDD100k, NamedImageFolder, BINARYDATASET, HFImageNetDataset
from models import get_classifier

import os
import tqdm
import argparse
import pandas as pd


def arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', required=True,
                        choices=['BDD', 'CelebAHQ', 'ImageNet'],
                        help='Dataset name')
    parser.add_argument('--partition', type=str, default='train',
                        help='Dataset partition')
    parser.add_argument('--data-dir', required=True, type=str,
                        help='dataset path')
    parser.add_argument('--label-query', type=int, default=0,
                        help='Query label to check. Only applies for binary datasets')
    parser.add_argument('--image-size', default=256, type=int,
                        help='dataset image size')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Inference batch size')
    parser.add_argument('--classifier-path', required=True,
                        help='path to classifier')
    parser.add_argument('--dataset_source', type=str, default='torchvision',
                        choices=['torchvision', 'hf'],
                        help='Where to load the dataset from (only used for ImageNet)')
    parser.add_argument('--hf_token', type=str, default='',
                        help='HuggingFace token (defaults to HF_TOKEN env var)')
    parser.add_argument('--hf_cache', type=str, default='',
                        help='HF datasets cache dir (defaults to DATASET_CACHE env var)')

    return parser.parse_args()


if __name__ == '__main__':

    torch.set_grad_enabled(False)

    args = arguments()
    device = torch.device('cuda:0')
    os.makedirs('utils', exist_ok=True)

    if args.dataset == 'CelebAHQ':
        dataset = CelebAHQDataset(
            image_size=args.image_size,
            data_dir=args.data_dir,
            partition=args.partition,
            normalize=False,
            random_crop=False,
            random_flip=False,
            return_filename=True,
            label_query=args.label_query
        )
    elif args.dataset == 'BDD':
        dataset = BDD100k(
            image_size=256,
            data_dir=args.data_dir,
            partition=args.partition,
            normalize=False,
            padding=False
        )
    elif args.dataset == 'ImageNet':
        if args.dataset_source == 'hf':
            split = args.partition if args.partition != 'val' else 'validation'
            # Keep [0,1] here; Normalizer will handle ImageNet statistics
            ds = HFImageNetDataset(
                image_size=args.image_size,
                split=split,
                hf_token=args.hf_token or os.environ.get('HF_TOKEN', None),
                cache_dir=args.hf_cache or os.environ.get('DATASET_CACHE', None),
                normalize=False
            )
            class HFWithIndex(torch.utils.data.Dataset):
                def __init__(self, base):
                    self.base = base
                def __len__(self):
                    return len(self.base)
                def __getitem__(self, i):
                    x, y = self.base[i]
                    return x, y, str(i)
            dataset = HFWithIndex(ds)
        else:
            # For predictions, we do not normalize to [-1, 1]; instead we keep [0,1]
            # and let the Normalizer wrapper handle ImageNet statistics.
            tfm = transforms.Compose([
                transforms.Resize(args.image_size),
                transforms.CenterCrop(args.image_size),
                transforms.ToTensor(),
            ])
            root = os.path.join(args.data_dir, args.partition)
            dataset = NamedImageFolder(root=root, transform=tfm)

    loader = data.DataLoader(dataset,
                             batch_size=args.batch_size,
                             num_workers=5,
                             shuffle=False)

    classifier = get_classifier(args)
    classifier.to(device).eval()

    d = {'idx': [],
         'prediction': []}
    n = 0
    acc = 0

    for img, lab, img_file in tqdm.tqdm(loader):

        img = img.to(device)
        lab = lab.to(device)
        logits = classifier(img)
        if args.dataset in BINARYDATASET:
            pred = (logits > 0).int()
        else:
            pred = logits.argmax(dim=1).int()
        acc += (pred == lab).float().sum().item()
        n += lab.size(0)

        d['prediction'] += [p.item() for p in pred]
        d['idx'] += list(img_file)

    print(acc / n)
    df = pd.DataFrame(data=d)
    df.to_csv(
        'utils/{}-{}-prediction-label-{}.csv'.format(args.dataset.lower(), args.partition, args.label_query),
        index=False
    )
