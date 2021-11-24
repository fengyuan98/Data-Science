- A method of estimating the value of an unknown quantity using the principles of inferential statistics.
- inferential Statistics
  - Population: a set of example
  - Sample: a proper subset of a population
  - Key fact: a random sample tends to exhibit the same properties as the population from which it is drawn.

>**在独立同分布条件下的随机变量平均值的表现**：
>
>- 大数定律
>
>  - 在相同条件下大量重复进行一种随机实验时，一件事情发生的次数与实验次数的比值，即该事件发生的频率值会趋近于**某一数值**。
>  - **样本均值收敛到总体均值**，即**期望**。
>
> $$
>  \frac{1}{n}S_n - E(X)\stackrel{p}{\rightarrow}0\ \ \ (S_n = \sum_{i=1}^nX_i)
> $$
>
>- 中心极限定理
>
>  - 在很一般的条件下，n个随即变量的和当n趋近于正无穷时的极限分布是**正态分布**。
>  - 当样本足够大时，**样本均值的分布会慢慢变成正态分布**。
>
> $$
>  \sqrt{n}(\frac{S_n}{n}-E(X))\stackrel{D}{\rightarrow}N(0, \Sigma)
> $$
>
>这两个定律都是在说样本均值性质。随着n增大，大数定律说样本均值几乎必然等于均值。中心极限定律说，他越来越趋近于正态分布。并且这个正态分布的方差越来越小。
>
>直观上来讲，想到大数定律的时候，你脑海里浮现的应该是一个样本，而想到中心极限定理的时候脑海里应该浮现出很多个样本。 



[<https://www.yanxishe.com/TextTranslation/2679>]



## 马尔可夫链蒙特卡洛算法（MCMC）

MCMC方法是用来在概率空间，通过随机采样估算兴趣参数的后验分布

#### 动机

- 假如你需要对一维随机变量$X$进行采样， ![[公式]](https://www.zhihu.com/equation?tex=X) 的样本空间是 ![[公式]](https://www.zhihu.com/equation?tex=%5C%7B1%2C2%2C3%5C%7D) ，且概率分别是 ![[公式]](https://www.zhihu.com/equation?tex=%5C%7B1%2F2%2C1%2F4%2C1%2F4%5C%7D) ，这很简单，只需写这样简单的程序：首先根据各离散取值的概率大小对 ![[公式]](https://www.zhihu.com/equation?tex=%5B0%2C1%5D)区间进行等比例划分，如划分为 ![[公式]](https://www.zhihu.com/equation?tex=%5B0%2C0.5%5D%2C%5B0%2C5%2C0.75%5D%2C%5B0.75%2C1%5D) 这三个区间，再通过计算机产生 ![[公式]](https://www.zhihu.com/equation?tex=%5B0%2C1%5D) 之间的伪随机数，根据伪随机数的落点即可完成一次采样。接下来，假如 ![[公式]](https://www.zhihu.com/equation?tex=X) 是连续分布的呢，概率密度是 ![[公式]](https://www.zhihu.com/equation?tex=f%28X%29) ，那该如何进行采样呢？聪明的你肯定会想到累积分布函数， ![[公式]](https://www.zhihu.com/equation?tex=P%28X%3Ct%29%3D%5Cint+_%7B-%5Cinfty%7D%5E%7Bt%7Df%28x%29dx) ，即在 ![[公式]](https://www.zhihu.com/equation?tex=%5B0%2C1%5D) 间随机生成一个数 ![[公式]](https://www.zhihu.com/equation?tex=a) ，然后求使得使 ![[公式]](https://www.zhihu.com/equation?tex=P%28x%3Ct%29%3Da) 成立的 ![[公式]](https://www.zhihu.com/equation?tex=t) ， ![[公式]](https://www.zhihu.com/equation?tex=t) 即可以视作从该分部中得到的一个采样结果。这里有两个前提：一是概率密度函数可积；第二个是累积分布函数有反函数。假如条件不成立怎么办呢？MCMC就登场了。
- 假如对于高维随机变量，比如 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbb%7BR%7D%5E%7B50%7D) ，若每一维取100个点，则总共要取 ![[公式]](https://www.zhihu.com/equation?tex=10%5E%7B100%7D) ，而已知宇宙的基本粒子大约有 ![[公式]](https://www.zhihu.com/equation?tex=10%5E%7B87%7D) 个，对连续的也同样如此。因此MCMC可以解决“维数灾难”问题。

### 蒙特卡洛

#### 引入







































