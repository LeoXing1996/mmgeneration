<div align="center">
    <img src="https://user-images.githubusercontent.com/12726765/114528756-de55af80-9c7b-11eb-94d7-d3224ada1585.png" width="400"/>
      <div>&nbsp;</div>
   <div align="center">
     <b><font size="5">OpenMMLab 官网</font></b>
     <sup>
       <a href="https://openmmlab.com">
         <i><font size="4">HOT</font></i>
       </a>
     </sup>
     &nbsp;&nbsp;&nbsp;&nbsp;
     <b><font size="5">OpenMMLab 开放平台</font></b>
     <sup>
       <a href="https://platform.openmmlab.com">
         <i><font size="4">TRY IT OUT</font></i>
       </a>
     </sup>
   </div>
   <div>&nbsp;</div>
</div>

[![PyPI](https://img.shields.io/pypi/v/mmgen)](https://pypi.org/project/mmgen)
[![docs](https://img.shields.io/badge/docs-latest-blue)](https://mmgeneration.readthedocs.io/en/latest/)
[![badge](https://github.com/open-mmlab/mmgeneration/workflows/build/badge.svg)](https://github.com/open-mmlab/mmgeneration/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmgeneration/branch/1.x/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmgeneration)
[![license](https://img.shields.io/github/license/open-mmlab/mmgeneration.svg)](https://github.com/open-mmlab/mmgeneration/blob/1.x/LICENSE)
[![open issues](https://isitmaintained.com/badge/open/open-mmlab/mmgeneration.svg)](https://github.com/open-mmlab/mmgeneration/issues)
[![issue resolution](https://isitmaintained.com/badge/resolution/open-mmlab/mmgeneration.svg)](https://github.com/open-mmlab/mmgeneration/issues)

[📘使用文档](https://mmgeneration.readthedocs.io/en/latest/) |
[🛠️安装教程](https://mmgeneration.readthedocs.io/en/latest/get_started.html#installation) |
[👀模型库](https://mmgeneration.readthedocs.io/en/latest/modelzoo_statistics.html) |
[🆕更新记录](https://github.com/open-mmlab/mmgeneration/blob/1.x/docs/en/changelog.md) |
[🚀进行中的项目](https://github.com/open-mmlab/mmgeneration/projects) |
[🤔提出问题](https://github.com/open-mmlab/mmgeneration/issues)

[English](README.md) | 简体中文

## 简介

MMGeneration 是一个基于 PyTorch 和[MMCV](https://github.com/open-mmlab/mmcv/tree/2.x)的强有力的生成模型工具箱，尤其专注于 GAN 模型。
主分支目前支持 **PyTorch 1.5** 以上的版本。

<div align="center">
    <img src="https://user-images.githubusercontent.com/12726765/114534478-9a65a900-9c81-11eb-8087-de8b6816eed8.png" width="800"/>
</div>

## 主要特性

- **高质量高性能的训练:** 我们目前支持 Unconditional GANs, Internal GANs, Image Translation Models 和 Diffusion Models 的训练。
- **强有力的应用工具箱:** 为用户提供了丰富的工具箱，包含 GANs 中的多种应用。我们的框架集成了 GANs 的插值，投影和编辑。请用你的 GANs 尽情尝试！([应用教程](docs/zh_cn/advanced_guides/applications.md))
- **生成模型的高效分布式训练:** 对于生成模型中的高度动态训练，我们采用 `MMDDP` 的新方法来训练动态模型。([DDP教程](docs/tutorials/ddp_train_gans.md))
- **灵活组合的新型模块化设计:** 针对复杂的损失模块，我们提出了一种新的设计，可以自定义模块之间的链接，实现不同模块之间的灵活组合。 ([新模块化设计教程](docs/zh_cn/advanced_guides/losses.md))

<table>
<thead>
  <tr>
    <td>
<div align="center">
  <b> 训练可视化</b>
  <br/>
  <img src="https://user-images.githubusercontent.com/12726765/114509105-b6f4e780-9c67-11eb-8644-110b3cb01314.gif" width="200"/>
</div></td>
    <td>
<div align="center">
  <b> GAN 插值</b>
  <br/>
  <img src="https://user-images.githubusercontent.com/12726765/114679300-9fd4f900-9d3e-11eb-8f37-c36a018c02f7.gif" width="200"/>
</div></td>
    <td>
<div align="center">
  <b> GAN 投影</b>
  <br/>
  <img src="https://user-images.githubusercontent.com/12726765/114524392-c11ee200-9c77-11eb-8b6d-37bc637f5626.gif" width="200"/>
</div></td>
    <td>
<div align="center">
  <b> GAN 编辑</b>
  <br/>
  <img src="https://user-images.githubusercontent.com/12726765/114523716-20302700-9c77-11eb-804e-327ae1ca0c5b.gif" width="200"/>
</div></td>
  </tr>
</thead>
</table>

## 亮点

- **Positional Encoding as Spatial Inductive Bias in GANs (CVPR2021)** 已在 `MMGeneration` 中发布.  [\[配置文件\]](configs/positional_encoding_in_gans/README.md), [\[项目主页\]](https://nbei.github.io/gan-pos-encoding.html)
- 我们已经支持训练目前主流的 Conditional GANs 模型，更多的方法和预训练权重马上就会发布，敬请期待。
- 混合精度训练已经在 `StyleGAN2` 中进行了初步支持，请到[这里](configs/styleganv2/README.md)查看各种实现方式的详细比较。

## 更新日志

2022/8/31 发布了 v1.0.0rc0 版本

这个版本引入一个全新的，可扩展性强的训练和测试引擎，但目前仍在开发中。欢迎根据[文档](https://mmgeneration.readthedocs.io/zh_cn/1.x/)进行试用。

同时，新版本中存在一些与旧版本不兼容的修改。请查看[迁移文档](https://mmgeneration.readthedocs.io/zh_cn/1.x/migration.html)来详细了解这些变动。

新版本的公测将持续到 2022 年末，在此期间，我们将基于 `1.x` 分支进行更新，不会合入到 `master` 分支。另外，至少
到 2023 年末，我们会保持对 0.x 版本的维护。

发布历史和更新细节请参考 [更新日志](https://mmgeneration.readthedocs.io/zh_CN/1.x/notes/changelog.html)

## 安装

MMGeneration 依赖 [PyTorch](https://pytorch.org/) 和 [MMCV](https://github.com/open-mmlab/mmcv/tree/2.x)，以下是安装的简要步骤。

**步骤 1.**
依照[官方教程](https://pytorch.org/get-started/locally/)安装PyTorch，例如

```python
pip3 install torch torchvision
```

**步骤 2.**
使用 [MIM](https://github.com/open-mmlab/mim) 安装 MMCV

```
pip install -U openmim
# wait for more pre-compiled pkgs to release
mim install 'mmcv>=2.0.0rc1'
```

**步骤 3.**
从源码安装 MMGeneration

```
git clone -b 1.x https://github.com/open-mmlab/mmgeneration.git
cd mmgeneration
pip3 install -e .[all]
```

更详细的安装指南请参考 [get_started.md](docs/zh_cn/get_started.md) .

## 快速入门

对于 `MMGeneration` 的基本使用请参考 [快速入门](docs/zh_cn/get_started.md)。其他细节和教程，请参考我们的[文档](https://mmgeneration.readthedocs.io/zh_cn/1.x)。

## 模型库

这些算法在我们的框架中得到了认真研究和支持。

<details open>
<summary>Unconditional GANs (点击折叠)</summary>

- ✅ [DCGAN](configs/dcgan/README.md) (ICLR'2016)
- ✅ [WGAN-GP](configs/wgan-gp/README.md) (NIPS'2017)
- ✅ [LSGAN](configs/lsgan/README.md) (ICCV'2017)
- ✅ [GGAN](configs/ggan/README.md) (arXiv'2017)
- ✅ [PGGAN](configs/pggan/README.md) (ICLR'2018)
- ✅ [StyleGANV1](configs/styleganv1/README.md) (CVPR'2019)
- ✅ [StyleGANV2](configs/styleganv2/README.md) (CVPR'2020)
- ✅ [StyleGANV3](configs/styleganv3/README.md) (NeurIPS'2021)
- ✅ [Positional Encoding in GANs](configs/positional_encoding_in_gans/README.md) (CVPR'2021)

</details>

<details open>
<summary>Conditional GANs (点击折叠)</summary>

- ✅ [SNGAN](configs/sngan_proj/README.md) (ICLR'2018)
- ✅ [Projection GAN](configs/sngan_proj/README.md) (ICLR'2018)
- ✅ [SAGAN](configs/sagan/README.md) (ICML'2019)
- ✅ [BIGGAN/BIGGAN-DEEP](configs/biggan/README.md) (ICLR'2019)

</details>

<details open>
<summary>Image2Image Translation (点击折叠)</summary>

- ✅ [Pix2Pix](configs/pix2pix/README.md) (CVPR'2017)
- ✅ [CycleGAN](configs/cyclegan/README.md) (ICCV'2017)

</details>

<details open>
<summary>Internal Learning (点击折叠)</summary>

- ✅ [SinGAN](configs/dcgan/README.md) (ICCV'2019)

</details>

<details open>
<summary>Denoising Diffusion Probabilistic Models (点击折叠)</summary>

- ✅ [Improved DDPM](configs/improved_ddpm/README.md) (arXiv'2021)

</details>

## 相关应用

- ✅ [MMGEN-FaceStylor](https://github.com/open-mmlab/MMGEN-FaceStylor)

## 贡献指南

我们感谢所有的贡献者为改进和提升 MMGeneration 所作出的努力。请参考[贡献指南](https://github.com/open-mmlab/mmcv/blob/2.x/CONTRIBUTING.md)来了解参与项目贡献的相关指引。

## 引用

如果您发现此项目对您的研究有用，请考虑引用：

```BibTeX
@misc{2021mmgeneration,
    title={{MMGeneration}: OpenMMLab Generative Model Toolbox and Benchmark},
    author={MMGeneration Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmgeneration}},
    year={2020}
}
```

## 开源许可证

该项目采用 [Apache 2.0 license](LICENSE) 开源许可证。`MMGeneration` 中的一些操作使用了其他许可证。如果您使用我们的代码进行商业事务，请参考 [许可证](LICENSES.md) 并仔细检查。

## OpenMMLab 的其他项目

- [MMEngine](https://github.com/open-mmlab/mmengine): OpenMMLab 深度学习模型训练基础库
- [MMCV](https://github.com/open-mmlab/mmcv/tree/2.x): OpenMMLab 计算机视觉基础库
- [MIM](https://github.com/open-mmlab/mim): MIM 是 OpenMMlab 项目、算法、模型的统一入口
- [MMClassification](https://github.com/open-mmlab/mmclassification/tree/1.x): OpenMMLab 图像分类工具箱
- [MMDetection](https://github.com/open-mmlab/mmdetection/tree/3.x): OpenMMLab 目标检测工具箱
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d/tree/1.x): OpenMMLab 新一代通用 3D 目标检测平台
- [MMRotate](https://github.com/open-mmlab/mmrotate/tree/1.x): OpenMMLab 旋转框检测工具箱与测试基准
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation/tree/1.x): OpenMMLab 语义分割工具箱
- [MMOCR](https://github.com/open-mmlab/mmocr/tree/1.x): OpenMMLab 全流程文字检测识别理解工具箱
- [MMPose](https://github.com/open-mmlab/mmpose/tree/1.x): OpenMMLab 姿态估计工具箱
- [MMHuman3D](https://github.com/open-mmlab/mmhuman3d/tree/1.x): OpenMMLab 人体参数化模型工具箱与测试基准
- [MMSelfSup](https://github.com/open-mmlab/mmselfsup/tree/1.x): OpenMMLab 自监督学习工具箱与测试基准
- [MMRazor](https://github.com/open-mmlab/mmrazor/tree/1.x): OpenMMLab 模型压缩工具箱与测试基准
- [MMFewShot](https://github.com/open-mmlab/mmfewshot/tree/1.x): OpenMMLab 少样本学习工具箱与测试基准
- [MMAction2](https://github.com/open-mmlab/mmaction2/tree/1.x): OpenMMLab 新一代视频理解工具箱
- [MMTracking](https://github.com/open-mmlab/mmtracking/tree/1.x): OpenMMLab 一体化视频目标感知平台
- [MMFlow](https://github.com/open-mmlab/mmflow/tree/1.x): OpenMMLab 光流估计工具箱与测试基准
- [MMEditing](https://github.com/open-mmlab/mmediting/tree/1.x): OpenMMLab 图像视频编辑工具箱
- [MMGeneration](https://github.com/open-mmlab/mmgeneration/tree/1.x): OpenMMLab 图片视频生成模型工具箱
- [MMDeploy](https://github.com/open-mmlab/mmdeploy): OpenMMLab 模型部署框架

## 欢迎加入 OpenMMLab 社区

扫描下方的二维码可关注 OpenMMLab 团队的 [知乎官方账号](https://www.zhihu.com/people/openmmlab)，加入 OpenMMLab 团队的 [官方交流 QQ 群](https://jq.qq.com/?_wv=1027&k=aCvMxdr3)

<div align="center">
<img src="https://user-images.githubusercontent.com/22982797/115827101-66874200-a43e-11eb-9abf-831094c27ef4.JPG" height="400" />  <img src="https://user-images.githubusercontent.com/22982797/115827162-7c950280-a43e-11eb-8270-2f2ce7e79f93.JPG" height="400" />
</div>

我们会在 OpenMMLab 社区为大家

- 📢 分享 AI 框架的前沿核心技术
- 💻 解读 PyTorch 常用模块源码
- 📰 发布 OpenMMLab 的相关新闻
- 🚀 介绍 OpenMMLab 开发的前沿算法
- 🏃 获取更高效的问题答疑和意见反馈
- 🔥 提供与各行各业开发者充分交流的平台

干货满满 📘，等你来撩 💗，OpenMMLab 社区期待您的加入 👬
