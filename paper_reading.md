# 2022 VoiceFixer（44.1k):    Mel修复+Vocoder, 2d

### Background

- Derevb， **2021 A new weighted magnitude-phase loss, CRNN**
  
  - $L=MSE(|M|,|\hat{M}|)+\alpha\times|M|^2\sin^2((\theta-\hat{\theta})/2)$
  
  - 即：只有当幅度较大时，才有必要精确估计相位。且相位的权重可以调整，$\alpha=1$
  
  - RIRs of different rooms, Fixed 1m distance, Random place, Different T60.

- **Super-Resolution， 2017**， 波形到波形，Unet
  
  - *Bandwidth Extension*
  
  - Zero Interpolation & lowpass filtering: 相当于高频区域全部补零，
  
  - Cubic Interpolation：稍微恢复了接近Nyquist频率的部分。
  
  - 低分辨率信号预先用cubic插值，作为input，以确保模型输入输出同等大小
    即：模型*不需要*预测上采样倍数，倍数已经事先给出。

- Declipping, **Deep Filter(2019**): filtering instead of cIRM
  
  - Hard Clipping会在高频产生谐波分量。
  
  - Temporal Clipping detection is important. Histogram differences & Maximum counting

### Structure

- Degradation
  
  - 四种失真是Sequential的

- Analysis stage：估计mel谱的masking，以达到修复目的
  
  - Unet，**编码器解码器各六个，包含四个卷积块（BN+prelu+conv2d），池化+反卷积**。
  - target是mel的mask
  - loss是mel的绝对值损失

- Synthesis
  
  - **Pretrained** vocoder

- Baseline
  
  - Unet直接估计幅度谱

### Result

- 客观
  
  - Baseline（幅度谱）> 声码器理论上界 > 声码器实际

- 主观
  
  - 差不多，声码器略好一点
  
  - **原因：生成模型的结果往往时间上不对齐，导致客观指标偏低**

- 单独任务没有重新训练

# 2021 SDDNet（16k):    Mag去噪+Mag去混响+stft细化, 2d+1d

- 训练后面的时候，前面冻结
- Unet（补零，conv2d， IN， relu），中间用Gated-TCN（256-双路64-256）

# 2021 S-DCCRN（32k）：子带+全带DCCRN（单阶段），2d

### Structure

- 不用Bark、Mel，也不直接用STFT，用DPT-FSNet中的DenseNet编码器解码器。
  子带、全带的Target都是复数mask
  
  - **DenseNet**
    
    - 是ResNet的升级版
    
    - $x_l=Net_l(concat(x_1,x_2...x_{l-1}))$, #Param = $lk^2<<C^2$
    
    - $k$是每层新添加的通道数，*膨胀率*，k<<普通卷积通道数C
    
    - 参数更少，还保留了低层次的特征（特征复用）
  
  - Conv2d(k=1) + DenseNet + Conv2d(k=1)

- **Group = 2， 实现低高频分开处理**

- loss
  
  - 时域上，SISNR
  
  - 频域上，复平面的平方损失 + KL散度
  
  - 直接相加，权重为1

# 2019 MelGAN 和 2020 HiFi-GAN （22k）

### Background

- WaveNet(2016) as an Autoregressive Model
- WaveGlow(2019) as a flow-based model, very big

### netG

- melgan和hifigan都是1d

- melgan反卷积上采样$8*8*2*2=256$倍，
  k16s8, o256
  k16s8, o128
  k4s2, o64
  k4s2, o32
  *卷积核是步长倍数，减轻棋盘效应*

- melgan(conv1d) 反卷积后接residual块
  k3d1, k3, add
  k3d3, k3, add
  k3d9, k3, add
  *空洞率按幂指数增加，减轻棋盘效应*

- hifigan(conv1d)反卷积后接并联的
  **多感受野融合**
  一路k3d1, k3d3, k3d5, k3, k3, k3
  一路k7d1, k7d3, k7d5, k7, k7, k7
  一路k11d1, k11d3, k11d5, k11, k11, k11
  三路直接加起来取平均

### netD

#### Multi Scale Discriminator used by both

同样的结构工作在三个不同scale上，

1x，2x，4x，平均池化

k15o16,
k41o64, s4g4
k41o256, s4g16
k41o1024, s4g64,
k41o1024, s4g256,
k5o1024,
k3o1,
*分组卷积，以换来大卷积核*

#### Multi Period Discriminator used by hifigan

按照步长p，1d音频重排为2d，p取素数排列2，3，5，7，11

接下来用2d的卷积，核为$k\times1$，只在同一次筛出来的样本方向上延申

- 多尺度：Smooth waveforms，应对语音问题里特有的超长数据

- 多周期：Disjoint samples，能捕捉更多信号的周期模式

### Loss for netG

melgan：hinge + 10 * Feature Match（G应当保证，通过D后，每只D的每层都要匹配，L1损失）

hifigan：LSGAN + 2 * Feature Match + 45 * Mel损失（要求G生成的音频，与真实音频有类似的mel谱，L1损失，可以在训练早期稳定训练）



# 2023 HiFi++，与Voicefixer的异同点

#### 任务

- voicefixer
  一次完成四个任务，44.1k的采样率

- hifi++
  一次只能完成两个任务中的一个
  对band extension是2/4/8k变到16k，对降噪是16k

#### 结构

* voicefixer
  
  1. 先对*mel域*做mask，**监督学习问题**
  
  2. 再用预训练的vocoder生成音频只有第一阶段需要损失函数：Mel的L1损失

* hifi++
  
  1. 先用**2d Unet + hifigan + 1d Unet**生成音频
  
  2. 再转到*STFT域*用2d Unet估计IAM
  
  3. 再转回音频这三个阶段都作为netG，损失函数还是hifigan中的三个部分，LSGAN+FM+Mel损失

# 2023 TDANet(16k)（CNN ver.) 大量的特征重用

### 特征重用

- encoder下采样中每一层的feature map都抽出来，池化后加在一起，送进bottleneck

- bottleneck的输出仅仅是为了生成feature从encoder到decoder的mask，每一层对应地上采样回去（用最近邻）

- decoder上采样中每一层的输出都来自于前一层和对应encoder，不来自bottleneck

- **大量重用了encoder中的feature**

### 应用到48k

- 参数量7M（原16k时2.3M） 

Params

|     | Time         | Freq                                                           |
| --- | ------------ | -------------------------------------------------------------- |
| sr  | 即：长度增加↓      | 即：频谱变宽<br/>无关（conv2d）因为卷积核是固定大小的<br/>二次（conv1d）因为卷积通道与fft点数成正比 |
| 长度  | 无关，因为卷积核固定大小 | 即：频谱变长<br/>无关，因为卷积核沿着时间轴是固定大小                                  |

MACs

|     | Time     | Freq                                                         |
| --- | -------- | ------------------------------------------------------------ |
| sr  | 即：长度增加↓  | 即：频谱变宽<br/>一次（conv2d）沿着频谱多卷几次，故复杂度+1<br/>二次（conv1d）没有再增加复杂度了 |
| 长度  | 一次（conv） | 即：频谱变宽<br/>一次                                                |


