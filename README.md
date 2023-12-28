# STRIVER: Spatiotemporally regularized inversion for motion-resolved computational imaging

**Authors:** [Yunhui Gao](https://github.com/Yunhui-Gao) (gyh21@mails.tsinghua.edu.cn) and Liangcai Cao (clc@tsinghua.edu.cn)

*[HoloLab](http://www.holoddd.com/), Tsinghua University*

_______

<p align="left">
<img src="figs/fig_comparison.png", width='600'>
</p>

Holography is a powerful technique that records the amplitude and phase of an optical field simultaneously, enabling a variety of applications such as label-free biomedical analysis and coherent diffraction imaging. Holographic recording without a reference wave has been long pursued because it obviates the high experimental requirements of conventional interferometric methods. However, due to the ill-posed nature of the underlying phase retrieval problem, reference-free holographic imaging is faced with an inherent tradeoff between imaging fidelity and temporal resolution. Here, we propose a general computational framework, termed **spatiotemporally regularized inversion (STRIVER)**, to achieve motion-resolved, reference-free holographic imaging with high fidelity. Specifically, STRIVER leverages signal priors in the spatiotemporal domain to jointly eliminate phase ambiguities and motion artifacts, and, when combined with diversity measurement schemes, produces a physically reliable, time-resolved holographic video from a series of intensity-only measurements. We experimentally demonstrate STRIVER in near-field ptychography, where dynamic holographic imaging of freely swimming paramecia is performed at a framerate-limited speed of 112 fps. The proposed method can be potentially extended to other measurement schemes, spectral regimes, and computational imaging modalities, pushing the temporal resolution toward higher limits.

**Holographic video of live paramecia:**
<p align="left">
<img src="figs/vid_paramecia_1.gif", width="500">
<img src="figs/vid_paramecia_2.gif", width="500">
<img src="figs/vid_paramecia_3.gif", width="500">
</p>

## Requirements
The code has been implemented using Matlab 2022b. Older visions may be sufficient but have not been tested.

## Quick Start
- **Phase retrieval using simulated data.** Run [`demo_sim.m`](https://github.com/THUHoloLab/STRIVER/blob/master/main/demo_sim.m) with default parameters.

## Accelerated Implementations
The basic demo codes provide intuitive and proof-of-concept implementations for beginners, but are far from efficient. To facilitate faster reconstruction, we provide an optimized version based on CPU or GPU, which can be found at [`demo_sim_fast.m`](https://github.com/THUHoloLab/STRIVER/blob/master/main/demo_sim_fast.m). To enable GPU usage, simply set `gpu = true;` in the code.

The following figure shows the runtime (200 iterations) for different image dimensions. The results are obtained using a laptop computer with Intel&reg; Core&trade; i7-12700H (2.30 GHz) CPU and Nvidia GeForce RTX&trade; 3060 GPU.

<p align="left">
<img src="figs/fig_runtime.png", width='400'>
</p>

## Theories and References
For algorithm derivation and implementation details, please refer to our paper:

[Yunhui Gao and Liangcai Cao, "Motion-resolved, reference-free holographic imaging via spatiotemporally regularized inversion," Optica 11(1), XXXX-XXXX (2023).](https://doi.org/10.1364/OPTICA.506572)