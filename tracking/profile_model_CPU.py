import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import sys

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

import argparse
import torch
from lib.utils.misc import NestedTensor
from thop import profile
from thop.utils import clever_format
import time
import importlib

# from lib.models.FERMT.vit import vit_base_patch16_224, vit_tiny_patch16_224

torch.set_num_threads(60)

def parse_args():
    """
    args for training.
    """
    parser = argparse.ArgumentParser(description='Parse args for training')
    # for train
    parser.add_argument('--script', type=str, default='FERMT', choices=['FERMT'],
                        help='training script name')
    parser.add_argument('--config', type=str, default='vit_tiny_got10k', help='yaml configure file name')
    args = parser.parse_args()

    return args


# def evaluate_vit(model, template, search, search_last):
def evaluate_vit(model, template, search):
    '''Speed Test'''
    macs1, params1 = profile(model, inputs=(template, search),
                             custom_ops=None, verbose=False)
    macs, params = clever_format([macs1, params1], "%.3f")
    print('overall macs is ', macs)
    print('overall params is ', params)

    # T_w = 100
    # T_t = 500
    T_w = 500
    T_t = 1000
    print("testing speed ...")
    
    with torch.no_grad():
        # overall
        for i in range(T_w):
            # _ = model(template, search)
            _ = model.backbone(template, search)
            # _ = model.forward_head(search_last)
    with torch.no_grad():
        start = time.time()
        for i in range(T_t):
            # _ = model(template, search)
            _ = model.backbone(template, search)
            # _ = model.forward_head(search_last)
        end = time.time()
        avg_lat = (end - start) / T_t
        print("The average overall latency is %.2f ms" % (avg_lat * 1000))
        print("FPS is %.2f fps" % (1. / avg_lat))
        # for i in range(T_w):
        #     _ = model(template, search)
        # start = time.time()
        # for i in range(T_t):
        #     _ = model(template, search)
        # end = time.time()
        # avg_lat = (end - start) / T_t
        # print("The average backbone latency is %.2f ms" % (avg_lat * 1000))


def evaluate_vit_separate(model, template, search):
    '''Speed Test'''
    T_w = 50
    T_t = 1000
    print("testing speed ...")
    z = model.forward_backbone(template, image_type='template')
    x = model.forward_backbone(search, image_type='search')
    with torch.no_grad():
        # overall
        for i in range(T_w):
            _ = model.forward_backbone(search, image_type='search')
            _ = model.forward_cat(z, x)
        start = time.time()
        for i in range(T_t):
            _ = model.forward_backbone(search, image_type='search')
            _ = model.forward_cat(z, x)
        end = time.time()
        avg_lat = (end - start) / T_t
        print("The average overall latency is %.2f ms" % (avg_lat * 1000))


def get_data(bs, sz):
    img_patch = torch.randn(bs, 3, sz, sz)
    att_mask = torch.rand(bs, sz, sz) > 0.5
    return NestedTensor(img_patch, att_mask)


if __name__ == "__main__":
    device = 'cpu'
    # torch.cuda.set_device(device)
    # Compute the Flops and Params of our STARK-S model
    args = parse_args()
    '''update cfg'''
    yaml_fname = 'experiments/%s/%s.yaml' % (args.script, args.config)
    config_module = importlib.import_module('lib.config.%s.config' % args.script)
    cfg = config_module.cfg
    config_module.update_config_from_file(yaml_fname)
    '''set some values'''
    bs = 1
    z_sz = cfg.TEST.TEMPLATE_SIZE
    x_sz = cfg.TEST.SEARCH_SIZE

    if args.script == "FERMT":
        model_module = importlib.import_module('lib.models')
        model_constructor = model_module.build_FERMT
        model = model_constructor(cfg, training=False)
        # get the template and search
        # template = torch.randn(bs, 3, z_sz, z_sz, dtype=torch.float32)
        # search = torch.randn(bs, 3, x_sz, x_sz, dtype=torch.float32)
        template = torch.randn(bs, 3, z_sz, z_sz)
        search = torch.randn(bs, 3, x_sz, x_sz)
        # transfer to device
        model = model.to(device)
        template = template.to(device)
        search = search.to(device)
        model.eval()

        merge_layer = cfg.MODEL.BACKBONE.MERGE_LAYER
        if merge_layer <= 0:
            evaluate_vit(model, template, search)
        else:
            evaluate_vit_separate(model, template, search)

    else:
        raise NotImplementedError
