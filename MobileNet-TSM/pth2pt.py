import torch
import torchvision
from torch.utils.mobile_optimizer import optimize_for_mobile
# from model_v3 import mobilenet_v3_large

# from train import n_classes  #此变量是调的train.py种类超参数

from ops.models import TSN
from opts import parser
from ops import dataset_config

global args, best_prec1
args = parser.parse_args()

num_class, args.train_list, args.val_list, args.root_path, prefix = dataset_config.return_dataset(args.dataset,
                                                                                                  args.modality)
# model = mobilenet_v3_large(num_classes=n_classes)
model = TSN(num_class, args.num_segments, args.modality,
            base_model=args.arch,
            consensus_type=args.consensus_type,
            dropout=args.dropout,
            img_feature_dim=args.img_feature_dim,
            partial_bn=not args.no_partialbn,
            pretrain=args.pretrain,
            is_shift=args.shift, shift_div=args.shift_div, shift_place=args.shift_place,
            fc_lr5=not (args.tune_from and args.dataset in args.tune_from),
            temporal_pool=args.temporal_pool,
            non_local=args.non_local)

tune_from = "log\\TSM_HockeyFights_RGB_mobilenetv2_shift8_blockres_avg_segment8_e100\\88\ckpt.best.pth.tar"

if tune_from:
    print(("=> fine-tuning from '{}'".format(tune_from)))
    sd = torch.load(tune_from)
    sd = sd['state_dict']
    model_dict = model.state_dict()
    # print('=====MODEL=====')
    # for k, v in model_dict.items():
    #     print(k)
    # print(model_dict)
    # print('=====MODEL=====')
    replace_dict = []

    sd2 = {}
    sd1 = {}

    for k, v in sd.items():
        # print(k)
        o = 'module.' + k
        sd2[o] = v

    # print('=====load=====')

    for k, v in sd2.items():
        # print(k)
        ch = '.net'
        if '.net' in k:
            # print(k)
            # o = k.replace(ch, '')
            # sd1[o] = v
            sd1[k] = v
        else:
            sd1[k] = v

    # print('=====load=====')
    for k, v in sd1.items():
        if k not in model_dict and k.replace('.net', '') in model_dict:
            print('=> Load after remove .net: ', k)
            replace_dict.append((k, k.replace('.net', '')))
    for k, v in model_dict.items():
        if k not in sd1 and k.replace('.net', '') in sd1:
            print('=> Load after adding .net: ', k)
            replace_dict.append((k.replace('.net', ''), k))

    # print('replace_dict:', replace_dict)
    for k, k_new in replace_dict:
        sd1[k_new] = sd1.pop(k)
    keys1 = set(list(sd1.keys()))
    keys2 = set(list(model_dict.keys()))
    # print('keys1:', keys1)
    # print('keys2:', keys2)
    set_diff = (keys1 - keys2) | (keys2 - keys1)
    print('#### Notice: keys that failed to load: {}'.format(set_diff))
    if args.dataset not in args.tune_from:  # new dataset
        print('args.dataset:', args.dataset)
        print('args.tune_from:', args.tune_from)
        print('=> New dataset, do not load fc weights')
        # sd1 = {k: v for k, v in sd1.items() if 'fc' not in k}
        for k, v in list(sd1.items()):
            if 'classifier' in k:
                print('k:', k)
        sd1 = {k: v for k, v in sd1.items() if 'classifier' not in k}
    if args.dataset in args.tune_from:  # new dataset
        print('args.dataset:', args.dataset)
        print('args.tune_from:', args.tune_from)
        print('=> load fc weights')

    if args.modality == 'Flow' and 'Flow' not in args.tune_from:
        sd1 = {k: v for k, v in sd1.items() if 'conv1.weight' not in k}
    model_dict.update(sd1)
    # print(type(model_dict))
    # print(model_dict)
    model.load_state_dict(model_dict)

# model.load_state_dict(torch.load("log/TSM_HockeyFights_RGB_mobilenetv2_shift8_blockres_avg_segment8_e100/88/ckpt.best.pth.tar"))
# model = torchvision.models.mobilenet_v3_small(pretrained=True)
model.eval()
example = torch.rand(1, 3, 224, 224)
traced_script_module = torch.jit.trace(model, example)
optimized_traced_model = optimize_for_mobile(traced_script_module)
# optimized_traced_model._save_for_lite_interpreter("model/m24.pt")
optimized_traced_model._save_for_lite_interpreter("MODEL/m24.pt")
