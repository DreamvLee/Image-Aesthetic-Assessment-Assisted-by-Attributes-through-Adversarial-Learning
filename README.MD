论文实现 [AAAI-BowenP.4115]() 
参考电子工业出版社的《深度学习框架PyTorch：入门与实践》配套代码写的

## 环境准备

- 本程序需要安装[PyTorch](https://pytorch.org/)
- 还需要通过`pip install -r requirements.txt` 安装其它依赖


## 数据准备

美学评价使用数据集AADB来训练。

```
transforms = tv.transforms.Compose([
        tv.transforms.Resize(opt.image_size),
        tv.transforms.CenterCrop(opt.image_size),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
```

## 用法
如果想要使用visdom可视化，请先运行`python2 -m visdom.server`启动visdom服务
基本用法：
```
Usage： python main.py FUNCTION --key=value,--key2=value2 ..
```

- 训练
```bash
python main.py train --gpu --vis=False
```

- 生成图片

[点此](http://pytorch-1252820389.cosbj.myqcloud.com/netg_200.pth)可下载预训练好的生成模型，如果想要下载预训练的判别模型，请[点此](http://pytorch-1252820389.cosbj.myqcloud.com/netd_200.pth)
```bash
python main.py generate --nogpu --vis=False --netd-path=checkpoints/netd_109.pth --netg-path=checkpoints/netg_109.pth --gen-img=result.png --gen-num=64
```
- 单张测试

```
    python main.py singletest --nogpu --vis=False --netd-path=checkpoints/netd_2199.pth

```

-全测试集测试

```
   python main.py test --nogpu --vis=False --netd-path=checkpoints/netd_2199.pth    --gen-num=1 --batch_size=1

``` 


完整的选项及默认值



### 兼容性测试
train 
- [x] GPU  
- [] CPU  
- [] Python2
- [x] Python3

test: 

- [x] GPU
- [x] CPU
- [] Python2
- [x] Python3
