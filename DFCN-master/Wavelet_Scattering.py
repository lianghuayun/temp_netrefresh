import matplotlib.pyplot as plt
import torch
from kymatio.numpy import Scattering1D

# scattering = Scattering2D(J=2, shape=(32, 32))
import numpy as np
import torch
from kymatio import Scattering1D
# 创建小波散射变换实例
scattering=Scattering1D(J=2,shape=(4096,))
#创建一个示例信号并进行小波散射变换
t = np.linspace(0, 1, 4096, endpoint=False)
# 生成频率为 5 Hz 的正弦波
sin_wave = np.sin(2 * np.pi * 5 * t)
# 将其转换为形状为 (1, 4096) 的张量
x = sin_wave.reshape(1, -1)
# x = torch.randn(1, 4096)
# plt.imshow(x, cmap='viridis', aspect='auto')
# plt.title('Heatmap of Tensor (1, 4096)')
# plt.colorbar()  # 显示颜色条
# plt.show()

x_flat = x.flatten()
plt.plot(x_flat)
plt.title('Plot of Array (4096,)')
plt.xlabel('Index')
plt.ylabel('Value')
plt.show()


plt.figure(2)
Sx = scattering(x)[0]
print(Sx.shape)  #输出小波散射系数的形状
plt.imshow(Sx, cmap='viridis', aspect='auto')
plt.title('Heatmap')
plt.colorbar()  # 显示颜色条

plt.figure(3)
Sx0 = scattering(x)[0][0].reshape(1,-1)
Sx1 = scattering(x)[0][1].reshape(1,-1)
Sx2 = scattering(x)[0][2].reshape(1,-1)
Sx3 = scattering(x)[0][3].reshape(1,-1)
#输出小波散射系数的形状
print(Sx.shape)

plt.imshow(Sx3, cmap='viridis', aspect='auto')
plt.title('Heatmap')
plt.colorbar()  # 显示颜色条
plt.show()

