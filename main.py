# encoding:utf-8
import os

import Image
import ipdb
import numpy
import torch as t
import torchvision as tv
import tqdm
from model import NetG, NetD,NetmyG
from torchnet.meter import AverageValueMeter
import dataset as mydataset
import time
import pandas as pd


class Config(object):
    data_path = '/home/graydove/Datasets/AADB/originalSize_train/'  # 数据集存放路径
    lable_path = '/home/graydove/LXQ/AADB合并'
    alltestdata_path = "/home/graydove/Datasets/AADB/originalSize_test/"
    alltestlabel_path = "/home/graydove/LXQ/AADB_test"
    num_workers = 4  # 多进程加载数据所用的进程数
    image_size = 224  # 图片尺寸
    batch_size = 16
    max_epoch = 1000
    lr1 = 0.0004  # 生成器的学习率
    lr2 = 0.0002  # 判别器的学习率
    beta1 = 0.5  # Adam优化器的beta1参数
    gpu = True  # 是否使用GPU
    nz = 1  # 噪声维度
    ngf = 16  # 生成器feature map数
    ndf = 16  # 判别器feature map数

    save_path = 'imgs/'  # 生成图片保存路径
    test_path = "/home/graydove/Datasets/AADB/originalSize_test/farm1_286_20013434330_d99ab6b9a0_b.jpg"
    vis = True  # 是否使用visdom可视化
    env = 'meixueGAN'  # visdom的env
    plot_every = 20  # 每间隔20 batch，visdom画图一次

    debug_file = '/tmp/debuggan'  # 存在该文件则进入debug模式
    d_every = 1  # 每1个batch训练一次判别器
    g_every = 5  # 每5个batch训练一次生成器
    save_every = 10  # 没10个epoch保存一次模型
    netd_path = None  # 'checkpoints/netd_.pth' #预训练模型
    netg_path = None  # 'checkpoints/netg_211.pth'
    weidu = 12
    # 只测试不训练
    gen_img = 'result.png'
    # 从512张生成的图片中保存最好的64张
    gen_num = 64
    gen_search_num = 512
    gen_mean = 0  # 噪声的均值
    gen_std = 1  # 噪声的方差


opt = Config()


def train(**kwargs):
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)

    device = t.device('cuda') if opt.gpu else t.device('cpu')
    if opt.vis:
        from visualize import Visualizer
        vis = Visualizer(opt.env, server="172.16.6.194")

    # 数据
    transforms = tv.transforms.Compose([
        tv.transforms.Resize(opt.image_size),
        tv.transforms.CenterCrop(opt.image_size),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = mydataset.AADBDataset(opt.data_path, opt.lable_path, transforms=transforms)
    # dataset = tv.datasets.ImageFolder(opt.data_path, transform=transforms)
    dataloader = t.utils.data.DataLoader(dataset,
                                         batch_size=opt.batch_size,
                                         shuffle=True,
                                         num_workers=opt.num_workers,
                                         drop_last=True
                                         )

    # 网络
    netg, netd = NetmyG(opt), NetD(opt)
    map_location = lambda storage, loc: storage
    if opt.netd_path:
        netd.load_state_dict(t.load(opt.netd_path, map_location=map_location))
    if opt.netg_path:
        netg.load_state_dict(t.load(opt.netg_path, map_location=map_location))
    netd.to(device)
    netg.to(device)

    # 定义优化器和损失
    optimizer_g = t.optim.Adam(netg.parameters(), opt.lr1, betas=(opt.beta1, 0.999))
    optimizer_d = t.optim.Adam(netd.parameters(), opt.lr2, betas=(opt.beta1, 0.999))
    criterion = t.nn.BCELoss().to(device)
    # huber损失
    huber = t.nn.SmoothL1Loss().to(device)

    # 真图片label为1，假图片label为0
    # noises为生成网络的输入
    # TODO
    true_labels = t.ones([opt.batch_size, opt.weidu]).to(device)
    fake_labels = t.zeros([opt.batch_size, opt.weidu]).to(device)

    # noises 为12维
    fix_noises = t.randn(opt.batch_size, opt.nz, 1, opt.weidu).to(device)
    noises = t.randn(opt.batch_size, opt.nz, 1, opt.weidu).to(device)

    errord_meter = AverageValueMeter()
    errorg_meter = AverageValueMeter()
    errorgs_meter = AverageValueMeter()

    epochs = range(opt.max_epoch)
    for epoch in iter(epochs):
        for ii, (img, _) in tqdm.tqdm(enumerate(dataloader)):
            real_img = img.to(device)

            if ii % opt.d_every == 0:
                # 训练判别器
                optimizer_d.zero_grad()
                ## 尽可能的把真图片判别为正确

                output = netd(real_img)

                error_d_real = criterion(output, true_labels)
                error_d_real.backward(retain_graph=True)

                ## 尽可能把假图片判别为错误
                noises.data.copy_(t.randn(opt.batch_size, opt.nz, 1, opt.weidu))
                fake_img = netg(noises).detach()  # 根据噪声生成假图
                output = netd(fake_img)

                error_d_fake = criterion(output, fake_labels)
                error_d_fake.backward(retain_graph=True)

                optimizer_d.step()

                error_d = error_d_fake + error_d_real

                errord_meter.add(error_d.item())

            if ii % opt.g_every == 0:
                # 训练生成器
                optimizer_g.zero_grad()
                noises.data.copy_(t.randn(opt.batch_size, opt.nz, 1, opt.weidu))
                fake_img = netg(noises)
                output = netd(fake_img)
                error_g = criterion(output, true_labels)
                print(output.shape, true_labels.shape)
                error_g.backward(retain_graph=True)
                bb = noises.view(opt.batch_size,-1)

                print (bb.shape)
                error_s = huber(output, bb)
                error_s.backward(retain_graph=True)

                error_gs = error_s + error_g
                optimizer_g.step()
                errorg_meter.add(error_g.item())
                errorgs_meter.add(error_gs.item())

            if opt.vis and ii % opt.plot_every == opt.plot_every - 1:
                ## 可视化
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()
                fix_fake_imgs = netg(fix_noises).detach()
                vis.images(fix_fake_imgs.detach().cpu().numpy()[:64] * 0.5 + 0.5, win='fixfake')
                vis.images(real_img.data.cpu().numpy()[:64] * 0.5 + 0.5, win='real')
                vis.plot('errord', errord_meter.value()[0])
                vis.plot('errorg', errorg_meter.value()[0])
                vis.plot('errorgs', errorgs_meter.value()[0])

        if (epoch + 1) % opt.save_every == 0:
            # 保存模型、图片
            localtime = time.asctime(time.localtime(time.time()))
            print("要保存了")
            t.save(netd.state_dict(), 'chenck20190529/netd_%s.pth' % epoch)
            t.save(netg.state_dict(), 'chenck20190529/netg_%s.pth' % epoch)
            errord_meter.reset()
            errorg_meter.reset()
            errorgs_meter.reset()

@t.no_grad()
def generate(**kwargs):
    """
    随机生成动漫头像，并根据netd的分数选择较好的
    """
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)

    device = t.device('cuda') if opt.gpu else t.device('cpu')

    netg, netd = NetG(opt).eval(), NetD(opt).eval()
    noises = t.randn(opt.gen_search_num, opt.nz, 1, opt.weidu).normal_(opt.gen_mean, opt.gen_std)
    noises = noises.to(device)

    map_location = lambda storage, loc: storage
    netd.load_state_dict(t.load(opt.netd_path, map_location=map_location))
    netg.load_state_dict(t.load(opt.netg_path, map_location=map_location))
    netd.to(device)
    netg.to(device)

    # 生成图片，并计算图片在判别器的分数
    fake_img = netg(noises)
    fake_img = fake_img.view(opt.gen_search_num, 3, 224, 224)
    scores = netd(fake_img).detach()
    tv.utils.save_image(t.stack(fake_img), opt.gen_img, normalize=True, range=(-1, 1))
    # 挑选最好的某几张
    indexs = scores.topk(opt.gen_num)[1]
    result = []
    for ii in indexs:
        result.append(fake_img.data[ii])
    # 保存图片
    tv.utils.save_image(t.stack(result), opt.gen_img, normalize=True, range=(-1, 1))


@t.no_grad()
def singletest(**kwargs):
    """
    python main.py singletest --nogpu --vis=False --netd-path=checkpoints/netd_2199.pth

    :param kwargs:
    :return:
    """
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)

    device = t.device('cuda') if opt.gpu else t.device('cpu')
    pil_img = Image.open(opt.test_path)
    pil_img = pil_img.convert('RGB')
    transforms = tv.transforms.Compose([
        tv.transforms.Resize(opt.image_size),
        tv.transforms.CenterCrop(opt.image_size),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    data = transforms(pil_img)
    array = numpy.asarray(data)

    array = t.from_numpy(array)
    array = array.view(1, 3, 224, 224).to(device)

    print (array.shape)
    netd = NetD(opt).eval()
    map_location = lambda storage, loc: storage
    netd.load_state_dict(t.load(opt.netd_path, map_location=map_location))
    netd.to(device)
    getlabel = netd(array).double().detach()
    print (getlabel)


@t.no_grad()
def test(**kwargs):
    """
    python main.py test --nogpu --vis=False --netd-path=checkpoints/netd_2199.pth    --gen-num=1 --batch_size=1

    """
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)

    device = t.device('cuda') if opt.gpu else t.device('cpu')
    # 数据
    transforms = tv.transforms.Compose([
        tv.transforms.Resize(opt.image_size),
        tv.transforms.CenterCrop(opt.image_size),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    datasettest = mydataset.AADBDatasetTest(opt.alltestdata_path, opt.alltestlabel_path, transforms=transforms)
    # dataset = tv.datasets.ImageFolder(opt.data_path, transform=transforms)
    dataloadertest = t.utils.data.DataLoader(datasettest,
                                             batch_size=opt.batch_size,
                                             shuffle=True,
                                             num_workers=opt.num_workers,
                                             drop_last=True
                                             )
    netd = NetD(opt).eval()
    map_location = lambda storage, loc: storage
    netd.load_state_dict(t.load(opt.netd_path, map_location=map_location))

    netd.to(device)
    error = 0
    criterion = t.nn.MSELoss().to(device)
    for ii, (img, label) in tqdm.tqdm(enumerate(dataloadertest)):
        img224 = img.to(device)
        getlabel = netd(img224).double().detach()
        label = label.double()
        error = error + criterion(label, getlabel)

    print (error / 1000)

    #
    # map_location = lambda storage, loc: storage
    # netd.load_state_dict(t.load(opt.netd_path, map_location=map_location))
    #
    # netd.to(device)
    #
    #
    # # 生成图片，并计算图片在判别器的分数
    # fake_label = netd(noises).detach()
    # #scores = netd(fake_img).detach()
    # data = pd.DataFrame(fake_label)
    # data.to_csv("score.csv")

    # print (fake_label)


@t.no_grad()
def discriminator(**kwargs):
    """
    随机生成动漫头像，并根据netd的分数选择较好的
    """
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)

    device = t.device('cuda') if opt.gpu else t.device('cpu')

    netd = NetD(opt).eval()
    noises = t.randn(opt.gen_search_num, 3, 224, 224).normal_(opt.gen_mean, opt.gen_std)
    # noises = t.randn(3, 3, 224).normal_(opt.gen_mean, opt.gen_std)
    noises = noises.to(device)

    map_location = lambda storage, loc: storage
    netd.load_state_dict(t.load(opt.netd_path, map_location=map_location))

    netd.to(device)

    # 生成图片，并计算图片在判别器的分数
    fake_label = netd(noises).detach()
    # scores = netd(fake_img).detach()
    data = pd.DataFrame(fake_label)
    data.to_csv("score.csv")

    print (fake_label)
    # # 挑选最好的某几张
    # indexs = scores.topk(opt.gen_num)[1]
    # result = []
    # for ii in indexs:
    #     result.append(fake_img.data[ii])
    # # 保存图片
    # tv.utils.save_image(t.stack(result), opt.gen_img, normalize=True, range=(-1, 1))


if __name__ == '__main__':
    import fire

    fire.Fire()
