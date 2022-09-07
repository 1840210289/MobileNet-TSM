import torch  # 命令行是逐行立即执行的

# sd = torch.load('G:\learn\\2022.5\\temporal-shift-module-master-5.6\pretrained\\resnext50_32x4d-7cdf4587.pth')
sd = torch.load(
    'G:\learn\\2022.5\TSM-newmobile\pretrained\TSM_kinetics_RGB_mobilenetv2_shift8_blockres_avg_segment8_e100_dense.pth')
# sd = torch.load('G:\learn\\2022.5\TSM-newmobile\pretrained\mobilenetv2_1.0-f2a8633.pth.tar')
# sd = torch.load('./pretrained/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e50.pth')
# content = torch.load('G:\learn\\2022.5\\temporal-shift-module-master-5.6\pretrained\TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e50.pth')
# print(sd)
sd = sd['state_dict']
sd1 = {}
for k, v in sd.items():
    # print(k)
    # o = 'base_model.' + k
    # list_str = list(k)
    # ch = 'base_model.'
    # o = k.replace(ch, "")
    # list_str = ''.join(list_str)
    # o = list_str
    # sd1[o] = v
    # print(o)
    # print(k)
    # print(v)
    if '.net' in k:

        ch = 'net.'
        o = k.replace(ch, "")
        print(o)
# print(sd1)
# print(sd1.keys())   # keys()
# 之后有其他需求比如要看 key 为 model 的内容有啥
# print(content['model'])



