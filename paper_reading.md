# 2022   44.1k  VoiceFixer:    Mel修复+Vocoder

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

### Detail

- Degradation
  
  - 四种失真是Sequential的

- Analysis
  
  - Unet，**编码器解码器各六个，包含四个卷积块（BN+prelu+conv2d），下采样用池化，上采样用反卷积**。
  - target是mel的mask
  - loss是mel的绝对值损失

- Synthesis
  
  - 预训练的vocoder

- Baseline
  
  - Unet直接估计幅度谱

### Result

- 客观
  
  - Baseline（幅度谱）> 声码器理论上界 > 声码器实际

- 主观
  
  - 差不多，声码器略好一点
  
  - *原因：生成模型的结果往往时间上不对齐，导致客观指标偏低*

- 单独任务没有重新训练

# 2021   16k， SDDNet:    Mag去噪+Mag去混响+stft细化



- 训练后面的时候，前面冻结
- Unet（补零，conv2d， IN， relu），中间用Gated-TCN（256-双路64-256）

# 2021    32k， S-DCCRN：子带+全带DCCRN（单阶段），

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


