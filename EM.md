​	从不完全数据或有数据丢失的数据集（存在隐变量（hidden variable））中求解概率模型参数的极大似然估计，或极大后验概率估计方法。
$$
完全数据(complete\ data)=观测随机变量的数据Y+隐随机变量的数据Z
$$
​	迭代算法，没有解析解

​	EM算法与初始值的选取有关，初值不同可能得到不同的参数估计值	

**<u>算法流程</u>**

- 随机初始化分布参数$\theta=\{...\}$

- 循环下述步骤直至收敛

  - E步（求Q函数）：对每个i，根据上一次迭代的模型参数来计算出隐变量（z）的后验概率（也就是隐变量的期望），作为隐变量的现估计值
    $$
    Q_i(z^{(i)}):=p(z^{(i)}|x^{(i)};\theta)
    $$

  - M步（求使Q函数极大时的参数取值）：将似然函数最大化以获得新的参数

  $$
  \theta := arg\ max_\theta \sum_i \sum_{z^{(i)}}Q_i(z^{(i)})log\frac{p(x^{(i)},z^{(i)};\theta)}{Q_i(z^{(i)})}
  $$

### 案例导入

假设有3枚硬币，分别记作$A,B,C$，它们的正面出现的概率分别是$\pi,p,q$，进行如下试验：先抛硬币A，根据其结果选出硬币B或C，正面选硬币B，反面选硬币C，然后掷选出的硬币，出现正面记作1，反面记作0；独立地重复n次试验。

假设只能观测到掷硬币的结果，不能观测其过程，问如何估计三硬币正面出现的概率，即三硬币模型的参数：$\theta = \{\pi,p,q\}$ 

### 算法

输入：观测变量数据$Y$；隐变量数据$Z$，联合分布$P(Y,Z|\theta)$；条件分布$P(Z|Y,\theta)$ 

输出：模型参数$\theta$ 

- 选择参数的初始值，开始迭代

- E-Step：记$\theta^{(i)}$为第$i$次迭代参数$\theta$的估计值，在第$i+1$次迭代的$E$步，计算
  $$
  \begin{align}
  Q(\theta,\theta^{(i)})&=E_Z[logP(Y,Z|\theta)|Y,\theta^{(i)}] \\
  &=\sum_ZlogP(Y,Z|\theta)P(Z|Y,\theta^{(i)})
  \end{align}
  $$

- M-Step：求使$Q(\theta,\theta^{(i)})$极大化的$\theta$，确定第$i+1$次迭代的参数的估计值$\theta^{(i+1)}$ 
  $$
  \theta^{(i+1)}=arg\ max_\theta Q(\theta,\theta^{(i)})
  $$

- 重复E-Step 和 M-Step，直到收敛

  给出停止迭代的条件，一般是对较小的正数$\epsilon_1,\epsilon_2$，若满足
  $$
  ||\theta^{(i+1)}-\theta^{(i)}||<\epsilon_1 \ \ \ 或\ \  \ ||Q(\theta^{(i+1)},\theta^{(i)})-Q(\theta^{(i)},\theta^{(i)})||<\epsilon_2
  $$
  则停止迭代。

> Q函数：
>
> 完全数据的对数似然函数$logP(Y,Z|\theta)$ 关于在给定观测数据$Y$和当前参数$\theta^{(i)}$下对未观测数据$Z$的条件概率分布$P(Z|Y,\theta^{(i)})$的期望称为$Q$函数

### EM算法的导出

<u>方法：近似求解观测数据的对数似然函数的极大化问题来导出EM算法</u>

面对一个含有隐变量的概率模型，目标是极大化观测数据（不完全数据）

极大化$Y$关于参数$\theta$的对数似然函数，即：
$$
\begin{align}
max \ \ L(\theta)&=logP(Y|\theta)=log\sum_ZP(Y,Z|\theta) \\
&=log(\sum_ZP(Y|Z,\theta)P(Z|\theta))
\end{align}
$$
<u>EM算法就是通过迭代逐步近似极大化$L(\theta)$</u> 

假设在第$i$次迭代后$\theta$的估计值是$\theta^{(i)}$，我们希望新估计值$\theta$能使$L(\theta)$增加，即$L(\theta)>L(\theta^{(i)})$，并逐步达到极大值，为此，考虑两者之差：
$$
L(\theta)-L(\theta^{(i)})=log(\sum_Z P(Y|Z,\theta)P(Z|\theta))-logP(Y|\theta^{(i)})
$$
利用$Jensen \ \  inequality$ 得到其下界：
$$
\begin{align}
L(\theta)-L(\theta^{(i)})&=log(\sum_Z P(Y|Z,\theta^{(i)}) \frac{P(Y|Z,\theta)P(Z|\theta)}{P(Y|Z,\theta^{(i)})} )-logP(Y|\theta^{(i)})  \\
& \geq \sum_ZP(Z|Y,\theta^{(i)})log\frac{P(Y|Z,\theta)P(Z|\theta)}{P(Z|Y,\theta^{(i)})}-logP(Y|\theta^{(i)}) \\
&=\sum_ZP(Z|Y,\theta^{(i)})log\frac{P(Y|Z,\theta)P(Z|\theta)}{P(Z|Y,\theta^{(i)})P(Y|\theta^{(i)})}
\end{align}
$$
令
$$
B(\theta,\theta^{(i)})=L(\theta^{(i)})+\sum_ZP(Z|Y,\theta^{(i)})log\frac{P(Y|Z,\theta)P(Z|\theta)}{P(Z|Y,\theta^{(i)})P(Y|\theta^{(i)})}
$$
则
$$
L(\theta) \geq B(\theta,\theta^{(i)})
$$
即函数$B(\theta,\theta^{(i)})$是$L(\theta)$的一个下界，且
$$
L(\theta^{(i)})=B(\theta^{(i)},\theta^{(i)})
$$
因此，任何可以使$B(\theta,\theta^{(i)})$增大的$\theta$，也可以使$L(\theta)$增大，为了使$L(\theta)$有尽可能大的增长，选择$\theta^{(i+1)}$使$B(\theta,\theta^{(i)})$达到极大，即
$$
\theta^{(i+1)}=arg \ max_\theta\ B(\theta,\theta^{(i)})
$$
省去对$\theta$的极大化而言是常数的项：
$$
\begin{align}
\theta^{(i+1)}&=arg\ max_\theta(L(\theta^{(i)})+\sum_ZP(Z|Y,\theta^{(i)})log\frac{P(Y|Z,\theta)P(Z|\theta)}{P(Z|Y,\theta^{(i)})P(Y|\theta^{(i)})}) \\
&=arg\ max_\theta(\sum_ZP(Z|Y,\theta^{(i)})log P(Y|Z,\theta)P(Z|\theta))) \\
&=arg\ max_\theta(\sum_ZP(Z|Y,\theta^{(i)})log P(Y,Z,|\theta)) \\
&=arg\ max_\theta Q(\theta,\theta^{(i)})
\end{align}
$$
==EM算法是通过不断求解下界的极大化逼近求解对数似然函数极大化的算法==

==EM算法不能保证找到全局最优解==

#### EM算法在非监督学习中的应用

EM算法可以用于生成模型的非监督学习，生成模型由联合概率分布$P(X,Y)$表示 ，可以认为非监督学习训练数据是联合概率分布产生的数据；X为观测数据，Y为未观测数据。

（不同于监督学习的训练数据$\{(x_1,y_1),(x_2,y_2), \cdots , (x_N,y_N) \}$，非监督学习的训练数据缺失输出Y）

### 收敛性

> 定理 9.1
>
> 设$P(Y|\theta)$为观测数据的似然函数，$\theta^{(i)}\ (i=1,2,\cdots)$为EM算法得到的参数估计序列，$P(Y|\theta^{(i)})$为对应的似然函数序列，则$P(Y|\theta^{(i)})$是单调递增的，即：
> $$
> P(Y|\theta^{(i+1)}) \geq P(Y|\theta^{(i)})
> $$
> 定理 9.2
>
> 设$L(\theta)=logP(Y|\theta)$为观测数据的对数似然函数，$\theta^{(i)}\ (i=1,2,\cdots)$为EM算法得到的参数估计序列，$L(\theta^{(i)})$为对应的对数似然函数序列。
>
> - 如果$P(Y|\theta)$有上界，则$L(\theta^{(i)})=logP(Y|\theta^{(i)})$收敛到某一值$L^*$.
> - 在函数$Q(\theta,\theta')$与$L(\theta)$满足一定条件下，由EM算法得到的参数估计序列$\theta^{(i)}$的收敛值$\theta^*$是$L(\theta)$的稳定点.

### EM算法在高斯混合模型学习中的应用

#### 高斯混合模型（Gaussian Mixture Model）

具有如下形式的概率分布模型：
$$
P(y|\theta)=\sum_{k=1}^K\alpha_k \phi(y|\theta_k)
$$
其中，$\alpha_k$是系数，$\alpha_k \geq 0$，$\sum_{k=1}^K \alpha_k =1$；$\phi(y|\theta_k)$是高斯分布密度，$\theta_k=(\mu_k,\sigma_k^2)$，
$$
\phi(y|\theta_k)=\frac{1}{\sqrt{2\pi}\sigma_k}exp(- \frac{(y-\mu_k)^2}{2\sigma_k^2})
$$
称为第k个分模型

#### 高斯混合模型参数估计的EM算法

假设观测数据$y_1,y_2,\cdots, y_N$由高斯混合模型生成，
$$
P(y|\theta)=\sum_{k=1}^K\alpha_k \phi(y|\theta_k)
$$
其中，$\theta = (\alpha_1,\alpha_2,\cdots,\alpha_K;\theta_1,\theta_2,\cdots,\theta_K)$，下面用EM算法估计高斯混合模型的参数$\theta$

- 明确隐变量，写出完全数据的对数似然函数

  

- E-Step：确定Q函数

- M-Step：求Q函数对$\theta$的极大值



> 输入：观测数据，高斯混合模型
>
> 输出：高斯混合模型参数
>
> （1）取参数的初始值开始迭代
>
> （2）E-Step：依据当前模型参数，计算分模型k对观测数据$y_j$的响应度
>
> （3）M-Step：计算新一轮迭代的模型参数
>
> （4）重复（2）（3）步，直至收敛





### 推广

#### F函数的极大-极大算法

F函数：

假设隐变量数据Z的概率分布为$\tilde{P}(Z)$，定义分布$\tilde{P}$与参数$\theta$的函数$F(\tilde{P},\theta)$如下：
$$
F(\tilde{P},\theta)=E_{\tilde{P}}[logP(Y,Z|\theta)]+H(\tilde{P})
$$
$H(\tilde{P})=-E_{\tilde{P}}log\tilde{P}(Z)$是分布$\tilde{P}(Z)$的熵.



#### GEM算法（1）（Generalized expectation maximization）

输入：观测数据，F函数

输出：模型参数

（1）初始化参数$\theta^{(0)}$，开始迭代

（2）第$i+1$次迭代，第1步：记$\theta^{(i)}$为参数$\theta$的估计值，$\tilde{P}^{(i)}$为函数$\tilde{P}$的估计，求$\tilde{P}^{(i+1)}$使$\tilde{P}$极大化$F(\tilde{P},\theta^{(i)})$

（3）第2步：求$\theta^{(i+1)}$使$F(\tilde{P}^{(i+1)},\theta)$极大化

（4）重复（2）（3），直至收敛

问题：有时求$Q(\theta,\theta^{(i)})$的极大化是很困难的

#### GEM算法（2）

输入：观测数据，Q函数

输出：模型参数

（1）初始化参数$\theta^{(0)}$，开始迭代

（2）第$i+1$次迭代，第1步：记$\theta^{(i)}$为参数$\theta$的估计值，计算
$$
\begin{align}
Q(\theta,\theta^{(i)})&=E_Z[logP(Y,Z|\theta)|Y,\theta^{(i)}]\\
&=\sum_ZP(Z|Y,\theta^{(i)})logP(Y,Z|\theta)
\end{align}
$$
（3）第2步：求$\theta^{(i+1)}$使
$$
Q(\theta^{(i+1)},\theta^{(i)}) > Q(\theta^{(i)},\theta^{(i)})
$$
（4）重复（2）（3），直至收敛

#### GEM算法（3）

输入：观测数据，Q函数

输出：模型参数

（1）初始化参数$\theta^{(0)}=(\theta^{(0)}_1,\theta^{(0)}_2,\cdots, \theta^{(0)}_d)$，开始迭代

（2）第$i+1$次迭代，第1步：记$\theta^{(i)}=(\theta^{(i)}_1,\theta^{(i)}_2,\cdots, \theta^{(i)}_d)$为参数$\theta=(\theta_1,\theta_2, \cdots, \theta_d)$的估计值，$\tilde{P}^{(i)}$为函数$\tilde{P}$的估计，计算
$$
\begin{align}
Q(\theta,\theta^{(i)})&=E_Z[logP(Y,Z|\theta)|Y,\theta^{(i)}]\\
&=\sum_ZP(Z|y,\theta^{(i)})logP(Y,Z|\theta)
\end{align}
$$
（3）第2步：进行$d$次条件极大化

==细节待补充==

（4）重复（2）（3），直至收敛































