==统计学习方法里的AdaBoost还没看==

西瓜书：171到190页

集成学习通过构建并结合多个学习器来完成学习任务，有时也称为多分类器系统（multi-classifier system）或基于委员会的学习（committee- based learning）

- 一般结构
  - 产生一组“个体学习器”（individual learner）
  - 用某种策略将它们结合起来

- 同质（homogeneous）集成：只包含同种类型的个体学习器（亦称基学习器：base learner）
- 异质（heterogeneous）集成：由不同的学习算法生成，个体学习器称为“组件学习器（component learner）”

> ​	弱学习器（weak learner）：泛化性能略优于随机猜测的学习器；例如在二分类问题上精度略高于50%的分类器

- 个体学习器应“好而不同”：个体有一定的准确性，个体间有一定的多样性

> 随着集成中个体分类器数目T的增大，集成的错误率将指数级下降，最终趋于0

- 类别
  - Boosting为代表：个体学习器间存在强依赖关系，必须串行生成的序列化方法
  - Bagging和随机森林为代表：个体学习器间不存在强依赖关系，可同时生成的并行化方法

### Boosting

- 一族可将弱学习器提升为强学习器的算法

#### 工作机制

先从初始训练集训练出一个基学习器，再根据基学习器的表现对训练样分布进行调整，使得先前基学习器做错的训练样本在后续受到更多的关注，然后基于调整后的样本分布来训练下一个基学习器，如此重复进行，直至基学习器数目达到事先指定的值T，最终将这T个基学习器进行加权结合

#### AdaBoost

##### 算法流程

> ##$y_i\in \{-1,+1\};f$ 是真实函数
>
> 输入：训练集$D=\{(x_1,y_1), \cdots ,(x_m,y_m)\}$
>
> ​			基学习算法A
>
> ​			训练轮数$T$
>
> 过程：
>
> - $\mathcal{D}_1(x)=1/m$    （$\mathcal{D_1}$表示样本权值分布）
>
> - For $t=1,2,\cdots, T$ do
>
>   - $h_t=A(D,\mathcal{D}_t)$    （$h_t$表示一个基分类器）
>
>   - $\epsilon_t=P_{x \sim \mathcal{D_t}}(h_t(x) \neq f(x))$   （$h_t$的错误率）
>
>   - if $\epsilon > 0.5$ then break（若分类器错误率大于50%，就舍弃该分类器）
>
>   - $\alpha_t= \frac{1}{2}ln(\frac{1-\epsilon_t}{\epsilon_t})$    （确定分类器权重）
>
>   - $$
>     \mathcal{D_{t+1}(x)=\frac{\mathcal{D_t(x)}}{Z_t}} \times \begin{cases} exp(-\alpha_t) & if &h_t(x)=f(x) \\ exp(\alpha_t) & if &h_t(x)\neq f(x) \end{cases}
>      =\frac{\mathcal{D_t(x)}exp(-\alpha_tf(x)h_t(x))}{Z_t}
>     $$
>
>     （更新样本分布，其中$Z_t$是规范化因子，以确保$\mathcal{D_{t+1}}$是一个分布）
>
>   - End for    （若有超过半数的基分类器正确，则集成分类就正确）
>
> 输出：$H(x)=sign(\sum_{t=1}^T\alpha_t h_t(x))$ 

##### 推导方式

基于“加性模型（additive model）”，即基学习器的线性组合
$$
H(x)=\sum_{t=1}^T\alpha_t h_t(x)
$$
来最小化指数损失函数（exponential loss function）
$$
\mathcal{l_{exp}}(H|D)=\mathbb{E}_{x \sim \mathcal{D}}[e^{-f(x)H(x)}]
$$
若$H(x)$能令指数损失函数最小化，则：
$$
\begin{align}
& \frac{\partial{l_{exp}(H|D)}}{\partial{H(x)}}=-e^{-H(x)}P(f(x)=1|x)+e^{H(x)}P(f(x)=-1|x)=0
\\
& H(x)=\frac{1}{2}ln \frac{P(f(x)=1|x)}{P(f(x)=-1|x)}
\\
& 因此有：\\
& sign(H(x))=sign(\frac{1}{2}ln\frac{P(f(x)=1|x)}{P(f(x)=-1|x)})= \begin{cases} 1 & P(f(x)=1|x)>P(f(x)=-1|x)  \\ -1 & P(f(x)=1|x)<P(f(x)=-1|x) \end{cases}
\\ &=argmax_{y \in \{-1,1\}}P(f(x)=y|x)
\end{align}
$$
==这意味着 $sign(H(x))$ 达到了贝叶斯最优错误率==

##### 

- $h_1$是通过直接将基学习算法用于初始数据分布而得到的，此后迭代地生成$h_t$和$\alpha_t$ 
- 基分类器$h_t$ 基于分布$\mathcal{D_t}$产生
- 该基分类器权重$\alpha_t$应使得$\alpha_th_t$最小化指数损失函数

$$
\begin{align}
l_{exp}(\alpha_t h/t|\mathcal{D_t})&=\mathbb{E_{x \sim \mathcal{D_t}}}[e^{-f(x)\alpha_th_t(x)}] \\
&=\mathbb{E_{x \sim \mathcal{D_t}}}[e^{\alpha_t}\mathbb{I}(f(x)=h_t(x))+e^{\alpha_t}\mathbb{I}(f(x) \neq h_t(x))] \\
&= e^{-\alpha_t}P_{x\sim \mathcal{D_t}}(f(x)=h_t(x))+
 e^{\alpha_t}P_{x\sim \mathcal{D_t}}(f(x) \neq h_t(x)) \\
&= e^{-\alpha_t}(1-\epsilon_t)+e^{\alpha_t}\epsilon_t
 \end{align}
$$

 其中错误率$\epsilon_t=P_{x \sim \mathcal{D_t}}(h_t(x)\neq f(x))$ 
$$
\begin{align}
& \frac{\partial{l_{exp}(\alpha_th_t|\mathcal{D_t})}}{\partial{\alpha_t}}=-e^{-\alpha_t}(1-\epsilon_t)+e^{\alpha_t}\epsilon_t=0 \\
& \alpha_t = \frac{1}{2}ln(\frac{1-\epsilon_t}{\epsilon_t})
\end{align}
$$
$\mathcal{D_{t+1}}$的更新式在西瓜书175页，式（8.12）开始

##### 问题

若被抛弃的基学习器$h_t$太多，初始设置的学习轮数T还远未达到，导致最终集成中只包含很少的基学习器而性能不佳

- 解决：重采样法（re-sampling）在每一轮学习中，根据样本分布对训练集重新进行采样，再用重采样得到的样本集对基学习器进行训练

- 重赋权法（re-weighting）：在训练过程的每一轮中，根据样本分布为每个训练样本重新赋予一个权重

从偏差-方差分解的角度看，Boosting主要关注降低偏度，因此Boosting能基于泛化性能相当弱的学习器构建出很强的集成。

### Bagging

- 基于自助采样法（bootstrapping sampling）：随机取出一个样本放入采样集中，再把该样本放回初始数据集
- 基于每个采样集训练出一个基学习器，再将这些基学习器进行结合

#### 算法流程

>输入：训练集$D=\{(x_1,y_1),(x_2,y_2),\cdots, (x_m,y_m)\}$
>
>​			基学习算法$A$
>
>​			训练轮数$T$
>
>过程：
>
>- for $t=1,2,\cdots, T$ do
>  - $h_t=A(D,\mathcal{D_{bs}})$
>- End for
>
>输出：$H(x)=argmax_{y \in \mathbb{y}}\sum_{t=1}^T \mathbb{I}(h_t(x)=y)$ 

- 由于每个基学习器只使用了初始训练集中约63.2%的样本，剩下约36.8%的样本可用作验证集来对泛化性能进行“外包估计（out-of-bag estimate）”

  为此，需记录每个基学习器所使用的训练样本

  - $D_t$：$h_t$实际使用的训练样本集
  - $H^{oob}(x)$：对样本$x$的外包预测，即仅考虑那些未使用$x$训练的基学习器在$x$上的预测

$$
H^{oob}(x)=argmax_{y \in \mathbb{y}}\sum_{t=1}^T \mathbb{I}(h_t(x)=y)\cdot \mathbb{I}(x \notin D_t)
$$

​				则Bagging泛化误差的外包估计为
$$
\epsilon^{oob}=\frac{1}{|D|}\sum_{(x,y)\in D}\mathbb{I}(H^{oob}(x)≠y)
$$

- 从偏差-方差分解的角度看，Bagging主要关注降低方差，因此它在不剪枝决策树，神经网络等易受样本扰动的学习器上效用更为明显



### 结合策略

假定集成包含$T$个基学习器$\{h_1,h_2,\cdots, h_T\}$，其中$h_i$在示例$x$上的输出为$h_i(x)$。

#### 好处

##### 统计方面

##### 计算方面

##### 表示方面

#### 常见策略

##### 平均法（averaging）

对数值型输出最常见的结合策略

- 简单平均法（simple averaging）：

$$
H(x)=\frac{1}{T}\sum_{i=1}^Th_i(x)
$$

- 加权平均法（weighted averaging）：

$$
H(x)=\sum_{i=1}^Tw_ih_i(x)  \ \ \ \ \ w_i\geq 0; \ \ \ \sum_{i=1}^T w_i =1
$$

​		权重一般是从训练数据中学习而得，现实任务中的训练样本通常不充分或者存在噪声，这将使得学出的权重不完全可靠，尤其是对规模比较大的集成来说，要学习的权重比较多，较容易导致过拟合

​		一般，在个体学习器性能相差较大时宜使用加权平均法，而在个体学习器性能相近时宜使用简单平均法

##### 投票法（voting）

- 绝对多数投票法（majority voting）

$$
H(x)=\begin{cases}
c_j & if \ \ \sum_{i=1}^Th_i^j(x)>0.5\sum_{k=1}^N\sum_{i=1}^Th_i^k(x) \\
reject  & otherwise
\end{cases}
$$

即若某标记得票过半数，则预测为该标记；否则拒绝预测。$h_i^j(x)$是$h_i$在类标记$c_j$上的输出

- 相对多数投票法（plurality voting）

$$
H(x)=c_{argmax_j \sum_{=1}^Th_i^j(x)}
$$

 即预测为得票最多的标记，若同时有多个标记获得最高票，则从中随机选取一个

- 加权投票法（weighted voting）

$$
H(x)=c_{argmax_j\sum_{i=1}^Tw_ih_i^j(x)}
$$

与加权平均法类似，$w_i$是$h_i$的权重，通常$w_i \geq 0,\ \sum_{i=1}^Tw_i=1$

** 注：若基学习器的类型不同，则其类概率值不能直接进行比较；在此情况下，通常可将类概率输出转化为类标记输出（例如将类概率输出最大的$h_i^j(x)$设为1，其余设为0）然后再投票

##### 学习法

通过另一个学习器来进行结合，如Stacking；这里个体学习器被称为初级学习器，用于结合的学习器称为次级学习器或元学习器（meta-learner）

Stacking

先从初始数据集训练出初级学习器，然后“生成”一个新数据集用于训练次级学习器

在新数据集中，初级学习器的输出被当作样例输入特征，而初始样本的标记仍被当作样例标记

初级学习器可以使用同质或异质的学习算法

>输入：训练集$D=\{()\}$





###  多样性



































