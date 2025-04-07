import argparse

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    parser = argparse.ArgumentParser(description='Pytorch Detecting Out-of-distribution examples in neural networks')

    in_dataset = 'gastrovision'
    n_classes =11
    model_arch ='deit'
    # weights = f'/scratch/achhetri/experimentalResults/g-ood/{model_arch}/{in_dataset}.pt'
    weights = f'/work/FAC/HEC/DESI/yshresth/aim/achhetri/checkpoints_2/{model_arch}/medical/{in_dataset}_224.pt'

  
    base_dir = f'/users/achhetri/myWork/NERO_raw/repo_test.txt'

    id_path_train = f"/work/FAC/HEC/DESI/yshresth/aim/achhetri/medical/{in_dataset}/ID/train"
    id_path_valid = f"/work/FAC/HEC/DESI/yshresth/aim/achhetri/medical/{in_dataset}/ID/test"
    ood_path = f"/work/FAC/HEC/DESI/yshresth/aim/achhetri/medical/{in_dataset}/OOD"

    parser.add_argument('--in-dataset', default=in_dataset, type=str, help='kvasir dataset')
    parser.add_argument('--num_classes', default=n_classes, type=int, help='number of classes' )

    parser.add_argument('--model-arch', default=model_arch, type=str, help='model architecture available: [resnet18, vit]')
    parser.add_argument('--name', default=model_arch, type=str, help='neural network name and training set')
    parser.add_argument('--weights', default=weights, help='model weights to load')
    
    parser.add_argument('--seed', default=0, type=int, help='seed')
    parser.add_argument('-b', '--batch-size', default=32, type=int, help='mini-batch size')


    parser.add_argument('--base-dir', default=base_dir, type=str, help='result directory')

    
    parser.add_argument('--id_path_train', default=id_path_train, help='path to id train dataset')
    parser.add_argument('--id_path_valid', default=id_path_valid, help='path to id valid dataset')
    parser.add_argument('--ood_path', default=ood_path, help='path to ood dataset')
    
    #remove
    parser.add_argument("--label", default = '0', type=str, help="Label to process (0-9)")

    parser.set_defaults(argument=True)
    args = parser.parse_args()
    return args