import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

# from model_v3 import mobilenet_v3_large
# from train import n_classes  #此变量是调的train.py种类超参数
from ops.models import TSN
from opts import parser
from ops import dataset_config
from archs.mobilenet_v2 import mobilenet_v2




def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    global args, best_prec1
    args = parser.parse_args()

    num_class, args.train_list, args.val_list, args.root_path, prefix = dataset_config.return_dataset(args.dataset,
                                                                                                      args.modality)
    # model = mobilenet_v3_large(num_classes=n_classes)
    # model = TSN(num_class, args.num_segments, args.modality,
    #     #             base_model=args.arch,
    #     #             consensus_type=args.consensus_type,
    #     #             dropout=args.dropout,
    #     #             img_feature_dim=args.img_feature_dim,
    #     #             partial_bn=not args.no_partialbn,
    #     #             pretrain=args.pretrain,
    #     #             is_shift=args.shift, shift_div=args.shift_div, shift_place=args.shift_place,
    #     #             fc_lr5=not (args.tune_from and args.dataset in args.tune_from),
    #     #             temporal_pool=args.temporal_pool,
    #     #             non_local=args.non_local)

    model = mobilenet_v2()
    model = model.cuda()

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    img_path = "test_image/00065.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model
    # model = mobilenet_v3_large(num_classes=n_classes).to(device)
    # load model weights
    # tune_from = True
    tune_from = False
    model_weight_path = "G:\learn\\2022.6\shixian\TSM-test\pretrained\TSM_kinetics_RGB_mobilenetv2_shift8_blockres_avg_segment8_e100_dense.pth"
    #1
    # model.load_state_dict(torch.load(model_weight_path, map_location=device))

    if  tune_from == True:
        print(("=> fine-tuning from '{}'".format(model_weight_path)))
        sd = torch.load(model_weight_path)
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
            # o = 'module.' + k
            o = k
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
    # 2
    # torch.load(model_weight_path, map_location=device)
    model.eval()

    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "fileName:{}    class: {}   prob: {:.3}".format(img_path,class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    print(print_res)
    plt.show()


if __name__ == '__main__':
    main()
