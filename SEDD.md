### 一、离散扩散过程

对于文本生成任务，每次从词汇表中选取一个字符作为输出，这个分布是离散的，记为 $X=\{1,...,N\}$ ，为了描述选取每个词的概率，可以用概率向量 $p\in \mathbb{R}^N$ 进行表示，其第 i 个分量表示选取第 i 个词的概率，因此向量元素为正且总和为1。概率向量随时间演化，记为 $p_t\in \mathbb{R}^N$，另外约定一下符号：

$p_t(x)$ 表示时刻 $t$ 时状态在 $x$ 的概率，$p_x(t)$ 也是一样的，也可以写作 $p(x_t=x)$ ，即：
$$
p_t(x)=p(x_t=x)=p_x(t)
$$
后面可能有混用。

#### 1.1 连续马尔可夫链

与离散马尔可夫链不同，连续时间马尔可夫链的时间是连续的，状态变化可以发生在任何时刻，核心是**概率流**的流动。而离散马尔可夫链的状态转移是每隔固定时间发生一次转移。针对这一特点，采用“转移速率”对其进行描述，确切来说是**概率对时间的密度**（随时间的变化率），即转移速率矩阵 $Q$ 。

以这个矩阵 $Q$ 为例，其有3个状态：
$$
Q=
\begin{bmatrix}
-2.0 & 0.5 & 0.3 \\
1.0 & -1.5 & 0.4 \\
1.0 & 1.0 & -0.7 \\
\end{bmatrix}
$$
其中非对角元素 $Q(i,j),i\neq j$ 表示从状态 $j$ 转移到状态 $i$ 的瞬时速率，而对角线元素 $Q(i,i)$ 总是负的，表示的是离开这一状态的瞬时速率。

需要注意的是其仅仅描述瞬时速率。比如仅在极短的时间 $\Delta t$ 内，$Q_{ij}\Delta t$ 才表示在 $\Delta t$ 内状态从 $j$ 转移到 $i$ 的概率。因为其仅仅是线性近似。

现在考虑概率向量 $p_t$ ，设$p_i(t)$ 表示时刻 $t$ 处于状态 $i$ 的概率，考察在 $[t,t+\Delta t]$ 内，$p_i(t)$ 如何变化。

时刻 $t+\Delta t$ 处于状态 $i$ 的概率 $p_i(t+\Delta t)$ 来源于两部分：

+ 流入量：从其他状态 $j$ 在 $\Delta t$ 内跳转到 $i$ 的概率。
  + 状态 $j$ 出发的概率是 $p_j(t)$ ，在 $\Delta t$ 内从 $j$ 跳转到 $i$ 的概率是 $Q_{ij}(t)\Delta t$ 。求和则有 $\sum_{j\neq i}p_j(t)\cdot Q_{ij}\Delta t$
+ 未流出量：原本在状态 $i$ ，但在 $\Delta t$ 内未流出。
  + 在状态 $i$ 的概率是 $p_i(t)$ ，离开 $i$ 的总速率是 $-Q_{ii}$ ，在 $\Delta t$ 内跳出的概率大约是 $p_i(t)\cdot (-Q_{ii}\Delta t)$  。
  + 反过来没有流出的概率是 $p_i(t)\cdot (1+Q_{ii}\Delta t)$

因此有平衡方程：
$$
p_i(t + \Delta t) = \underbrace{p_i(t)(1 +Q_{ii}\Delta t)}_{\text{未离开}} + \underbrace{\sum_{j \neq i} p_j(t)Q_{ij}\Delta t}_{\text{从别处跳入}}
$$
进一步有：
$$
\begin{aligned}
&p_i(t+\Delta t)-p_i(t)=\sum_{j}p_j(t)Q_{ij}\Delta t\\
\Longrightarrow\ \ &\frac{p_i(t+\Delta t)-p_i(t)}{\Delta t}
=\sum_{j}p_j(t)Q_{ij}\\
\overset{\Delta t\rightarrow 0}\Longrightarrow\ \ &
\frac{dp_i(t)}{dt}=\sum_{j=1}^{N}p_j(t)Q_{ij}
\end{aligned}
$$
这就是 Kolmogorov 前向方程（主方程）。导数只考虑线性主部，因此导数在这里也只用看一步转移速率。另外其可以用矩阵表示，比如以 $p_1(t)$ 为例：
$$
\frac{d}{dt} \begin{bmatrix}
p_1(t) \\
\ldots \\
\ldots
\end{bmatrix}
=
\begin{bmatrix}
Q_{11} & Q_{12} & Q_{13} \\
\ldots & \ldots & \ldots \\
\ldots & \ldots & \ldots
\end{bmatrix}
\begin{bmatrix}
p_1(t) \\
p_2(t) \\
p_3(t)
\end{bmatrix}
$$
因此有：
$$
\frac{dp_t}{dt}=Q_tp_t
$$

#### 1.2 前向加噪过程

回到主线，离散扩散过程就是给定真实数据分布 $p_{\text{data}}$ ，首先将编码初始化为概率向量 $p_0$，然后启动连续时间马尔可夫链：
$$
\frac{dp_t}{dt}=Q_tp_t \tag{1},\ \ \ p_0\approx p_{\text{data}}
$$
我们让 $Q_t$ 足够简单，比如 $Q_t=\sigma (t) Q$ ，因此当 $t\rightarrow \infty$ 时，$p_t$ 会收敛于一个 $p_{\text{base}}$ （比如均匀分布或单点分布），这样就相当于不断往数据中添加噪声，直到完全模糊。

上述的过程是连续的，在实际操作中会取一个较短的步长进行离散化模拟，从状态 $x$ 出发，转移概率为：
$$
p(x_{t+\Delta t}=y|x_t=x) = \delta_{xy}+Q_t(y,x)\Delta t+O(\Delta t^2) \tag{2}
$$
其中第一项 $\delta_{xy}=\begin{cases} 1,\ x=y\\0,\ x\neq y\end{cases}$   ；第二项表示发生一次跳转 $x\rightarrow y$ 的概率，为一阶；后面的表示发生多次跳转再到 $y$ 的概率，为高阶无穷小量。

#### 1.3 反向去噪过程

反向过程从 $p_T$ 开始，通过另一个扩散矩阵 $\bar{Q}_t$ 逐步去除噪声，恢复原始数据分布。仍然考虑：
$$
p(x_{t+\Delta t}=y|x_t=x) = \delta_{xy}+Q_t(y,x)\Delta t+O(\Delta t^2)
$$
对于逆过程，可以用贝叶斯公式：
$$
\begin{aligned}
p(x_t=x|x_{t+\Delta t}=y)
&=
\frac{p(x_t=x,x_{t+\Delta t}=y)}{p_{y}(t+\Delta t)} \\\\
&=
\frac{p(x_{t+\Delta t}=y|x_t=x)\cdot p_x(t)}{p_{y}(t+\Delta t)}\\\\
&=
\frac{p_x(t)(\delta_{xy}+Q_t(y,x)\Delta t)+O(\Delta t^2)}{p_{y}(t+\Delta t)}\\\\
\end{aligned}
$$
分母展开到一阶：
$$
p_y(t+\Delta t)=p_y(t)+\frac{dp_y(t)}{dt}\Delta t+O(\Delta t^2)
\\=p_y(t)+\left(\sum_z p_z(t)Q_t(y,z) \right)\Delta t+O(\Delta t^2)
$$
带回去有：
$$
p(x_t=x|x_{t+\Delta t}=y)=\frac{p_x(t)(\delta_{xy}+Q_t(y,x)\Delta t)+O(\Delta t^2)}{p_y(t)+\left(\sum_z p_z(t)Q_t(y,z) \right)\Delta t+O(\Delta t^2)}
$$
现在考虑两种情况，$x=y$ 或 $x\neq y$：

##### 1.3.1 情况1：$x=y$

此时 $\delta_{xy}=1$，有：
$$
\begin{aligned}
p(x_t=x|x_{t+\Delta t}=x)
&=\frac{p_x(t)(1+Q_t(x,x)\Delta t)+O(\Delta t^2)}{p_x(t)+\left(\sum_z p_z(t)Q_t(x,z) \right)\Delta t+O(\Delta t^2)}\\\\
&=\frac{(1+Q_t(y,x)\Delta t)+O(\Delta t^2)}
{1+\frac{1}{p_x(t)}\sum_z p_z(t)Q_t(x,z)\Delta t +O(\Delta t^2)} \quad \rightarrow \text{把 }p_x(t) \text{ 提出来之后上下消掉} \\\\
\end{aligned}
$$
接下来利用展开式 $\frac{1}{1+u}=1-u+O(u^2)$ ，其中 $u=\frac{1}{p_x(t)}\sum_z p_z(t)Q_t(x,z)\Delta t$ 。由于仅考虑线性项，因此分母的二次项对一阶项无贡献，最后也会保留 $O(\Delta t^2)$ ：
$$
\begin{aligned}
p(x_t=x|x_{t+\Delta t}=x)
&=\left(1+Q_t(x,x)\Delta t+O(\Delta t^2)\right)\left(1-\frac{1}{p_x(t)}\sum_z p_z(t)Q_t(x,z)\Delta t +O(\Delta t^2)\right)\\
&=1+\left[Q_t(x,x)-\frac{1}{p_x(t)}\sum_z p_z(t)Q_t(x,z)\right]\Delta t+O(\Delta t^2)
\end{aligned}
$$

##### 1.3.2 情况2：$x\neq y$

此时 $\delta_{xy}=0$，同样对分母进行展开并提出来 $p_y(t)$：
$$
\begin{aligned}
p(x_t=x|x_{t+\Delta t}=y)
&=\frac{p_x(t)Q_t(y,x)\Delta t+O(\Delta t^2)}
{p_y(t)(1+\frac{1}{p_y(t)}\sum_z p_z(t)Q_t(y,z) \Delta t+O(\Delta t^2))}\\\\
&=\frac{p_x(t)Q_t(y,x)\Delta t+O(\Delta t^2)}{p_y(t)}\left(1-\frac{1}{p_y(t)}\sum_z p_z(t)Q_t(y,z)\Delta t +O(\Delta t^2)\right)\\\\
&=\frac{p_x(t)}{p_y(t)}Q_t(y,x)\Delta t + O(\Delta t^2) \quad \rightarrow\text{保留一阶项}
\end{aligned}
$$
综上，我们得到了**逆向推导的核心公式**：
$$
p(x_t=x|x_{t+\Delta t}=y)=
\begin{cases}
\displaystyle
1+\left[Q_t(x,x)-\frac{1}{p_x(t)}\sum_z p_z(t)Q_t(x,z)\right]\Delta t+O(\Delta t^2) 
&\quad ,x=y\\\\
\displaystyle
\frac{p_x(t)}{p_y(t)}Q_t(y,x)\Delta t + O(\Delta t^2)
&\quad ,x\neq y
\end{cases}
\tag{3}
$$
这个形式不太好看，我们回忆一下公式(2) ，希望把逆向过程也推导成这种形式：
$$
p(x_t=x|x_{t+\Delta t}=y)=\delta_{xy}+\bar{Q}_t(x,y)\Delta t +O(\Delta t^2) \tag{4}
$$
对比公式 (3) ，可以得到：
$$
\bar{Q}_t(x,y)=\begin{cases}
\displaystyle
Q_t(x,x)-\frac{1}{p_x(t)}\sum_z p_z(t)Q_t(x,z) &\quad ,x=y\\\\
\displaystyle
\frac{p_x(t)}{p_y(t)}Q_t(y,x) &\quad ,x\neq y
\end{cases}
$$
其实当我们反过来思考的时，$x=y$ 的情况就是 $x\neq y$ 的情况之和的相反数，这和正向过程是一模一样的。
$$
\bar{Q}_t(x,x) = -\sum_{y\neq x}\bar{Q}_t(y,x)
$$

> 事实上我们可以证明上面的两个关于 $x=y$ 的描述是等价的，对于 $x\neq y$ 有：
> $$
> \bar{Q}_t(x,y)=\frac{p_x(t)}{p_y(t)}Q_t(y,x)
> $$
> 则有：
> $$
> \sum_{y\neq x}\bar{Q}_t(y,x)=\sum_{y\neq x}\frac{p_y(t)}{p_x(t)}Q_t(x,y)=\frac{1}{p_x(t)}\sum_{y\neq x}p_y(t)Q_t(x,y)
> $$
> 反过来看另一个式子 ：
> $$
> \begin{aligned}
> Q_t(x,x)-\frac{1}{p_x(t)}\sum_z p_z(t)Q_t(x,z)
> &=Q_t(x,x)-\frac{1}{p_x(t)}\left[p_x(t)Q_t(x,x)+\sum_{y\neq x}p_y(t)Q_t(x,y) \right]\\\\
> &=-\frac{1}{p_x(t)}\sum_{y\neq x}p_y(t)Q_t(x,y)\\\\
> &=-\sum_{y\neq x}\bar{Q}_t(y,x)
> \end{aligned}
> $$
> 可见二者完全等价。

最后将这个 $\bar{Q}$ 的符号稍微和正向的对齐一下，就有了：
$$
\bar{Q}_t(y,x)=\begin{cases}
\displaystyle
-\sum_{z\neq x}\bar{Q}_t(z,x)&\quad ,x=y\\\\
\displaystyle
\frac{p_y(t)}{p_x(t)}Q_t(x,y) &\quad ,x\neq y
\end{cases}
\tag{5}
$$
进一步，对于每个状态都有和所有状态的类似交互关系，同样令 $\Delta t\rightarrow 0$ ，有导数关系：
$$
\frac{dp_{T-t}}{dt}=\bar{Q}_{T-t}p_{T-t}
$$

#### 1.4 前向和反向过程总结

前向过程是不断加噪的过程，时刻 $t$ 处于状态 $x$ ，$\Delta t$ 之后处于状态 $y$ 的概率是：
$$
p(x_{t+\Delta t}=y|x_t=x) = \delta_{xy}+Q_t(y,x)\Delta t+O(\Delta t^2)
$$
每个状态都与 $N$ 个状态产生这种联系，令 $\Delta t\rightarrow 0$ ，可以用导数描述这一关系：
$$
\frac{dp_t}{dt}=Q_tp_t 
$$
后向过程是不断去噪的过程，时刻 $t+\Delta t$ 处于状态 $y$ ，时刻 $t$ 处于状态 $x$ 的概率是：
$$
p(x_t=x|x_{t+\Delta t}=y)=\delta_{xy}+\bar{Q}_t(x,y)\Delta t +O(\Delta t^2)
$$
前向就是不断去噪去噪，直到还原原数据：
$$
\frac{dp_{T-t}}{dt}=\bar{Q}_{T-t}p_{T-t}
$$
其中正向扩散矩阵 $Q$ 和逆向扩散矩阵 $\bar{Q}$ 存在这个关系：
$$
\bar{Q}_t(y,x)=\frac{p_t(y)}{p_t(x)}Q_t(x,y)
$$
这里为了和文献兼容，换了种写法，用 $p_t(y)$ 替代了 $p_y(t)$

#### 1.5 我们需要什么

整理一下上述离散扩散过程中我们已知和未知的量。

已知的量有：

+ 正向扩散矩阵 $Q_t$，这是人为设计的，通常取 $Q_t=\sigma(t)Q$ 
+ 前向转移概率，对于任意小步长 $\Delta t$ ，可以计算 $p(x_{t+\Delta t}|x_t)$ 的值
+ 初始噪声分布 $p_T$ ，我们一般选择简单分布（比如均匀分布），可以直接采样

未知的量有：

+ 逆向扩散矩阵 $\bar{Q}_t$ ，这是构建逆向过程的关键，本质在于分布比率未知

+ 分布比率 (concrete score) $\frac{p_t(y)}{p_t(x)}$ ，这是逆向过程未知的根本

反过来思考离散扩散模型的目的：其与连续扩散模型其实是一致的，都是能够在概率空间中采样简单噪声，逐步去噪形成最终输出。这个过程依赖于逆向扩散矩阵 $\bar{Q}$ ，更本质的依赖于 分布比率 $\frac{p_t(y)}{p_t(x)}$ ，只要拿到这个比率，就能够构建上述的扩散过程。这个比率很难直接得到，因此往往通过神经网络学习取得，下面就着重阐述一下不同学习路线。

---

### 二、离散扩散模型——分布比率的不同学习路线

#### 2.1 均值预测（Mean Prediction）

这种方法比较暴力，它直接学习一个神经网络 $f_{\theta}(x_t,t)$ 来预测给定噪声状态 $x_t$ 时，原始数据 $x_0$ 的条件分布 $p(x_0|x_t)$ ，可以通过贝叶斯定理拿到 $p_t(x)$：
$$
p(x_0=y|x_t=x)=\frac{p(x_t=x|x_0=y)p_0(y)}{p_t(x)}
$$
把 $x$ 和 $y$ 一交换同样有 $p_t(y)$ 。两个东西一除就有：
$$
\frac{p_t(y)}{p_t(x)}=
\frac{p(x_0=y|x_t=x)}{p(x_0=x|x_t=y)} \cdot 
\frac{p(x_t=y|x_0=x)p_0(x)}
{p(x_t=x|x_0=y)p_0(y)}
$$
其中 $p(x_t=y|x_0=x)$ 可以用前向过程得到，$p_0(x)$ 则需要用训练集中的频率估计概率。

可以看出来这种方法有几个问题：

+ 其一，直接学习 $p_{0|t}$ 很难拟合，本来词表（状态）就很多，还要有 $t$ 步跳跃，其复杂度可想而知，拟合难度极大。
+ 其二，其依赖于前向过程和训练集频率，前向过程需要记录造成空间复杂度提升，频率估计概率的先验也会带来误差。
+ 其三，有研究表明该目标在连续时间内会失效，此时必须进行近似。

综上，这种方法并不佳。

#### 2.2 其他方法

> 这块引用论文没细看，翻译了一下原文内容

比率匹配（Ratio Matching）：最初由 Hyvärinen（2007）提出，并经 Sun 等人（2023）改进，通过最大似然训练来学习每个维度的边际概率。但由此产生的设置偏离了标准分数匹配，且需要专门且昂贵的网络架构（Chen & Duvenaud，2019）。因此，其性能往往不如均值预测。

Meng 等人（2022）推广了分数匹配中的标准费希尔散度，通过具体分数匹配学习 $s_{\theta}(x, t) \approx [\frac{p_{t}(y)}{p_{t}(x)}]_{y \neq x}$：
$$
\mathcal{L}_{\text{CSM}}=\frac{1}{2}\mathbb{E}_{x\sim p_t}\left[\sum_{y\neq x}\left(s_\theta(x_t,t)_y-\frac{p_t(y)}{p_t(x)}\right)^2\right]
$$
然而 $\ell^2$ 损失是对称的，对于正负的惩罚一样，与 $\frac{p_{t}(y)}{p_{t}(x)}$ 的非负性不兼容，导致无法充分惩罚负值或零值，甚至导致发散。

---

#### 三、分数熵离散扩散模型（Score Entropy Discrete Diffusion Models）

这是另外一种计算分布比率的方式。与具体分数匹配类似，这里学习的目标也是：
$$
s_{\theta}(x, t) \approx \left[\frac{p_{t}(y)}{p_{t}(x)}\right]_{y \neq x}
$$
其中 $s_{\theta}: X \times \mathbb{R} \to \mathbb{R}^{|X|}$。只是损失函数改为了分数熵损失函数，融入了比率为正且且在离散扩散下演化的特性。

#### 3.1 费希尔散度（Fisher Divergence）和布雷格曼散度（Bregman Divergence）

费希尔散度用于衡量两个分布 $p$ 和 $q$ 的之间的差异，核心思想是比较于两个分布的**梯度**有多像：
$$
D_F(p||q)=\mathbb{E}_{x\sim p}\left[||\nabla_x \log p(x)-\nabla_x\log q(x)||^2\right]
$$
布雷格曼散度则是一个更广义的距离概念，基于凸函数 $F$ 定义。设 $F:\Omega \subset \mathbb{R}^d\rightarrow \mathbb{R}$ 是严格凸且可微的，则布雷格曼散度为：
$$
D_F(p,q)=F(p)-\left[F(q)+\nabla F(q)^T(p-q)\right]
$$
其实就是 $p$ 在 $q$ 处基于函数 $F$ 进行一阶泰勒展开（切线）的线性近似误差。

#### 3.2 分数熵及其基于布雷格曼散度的推导

对于分布 $p$、权重 $w_{xy} \geq 0$ 和分数网络 $s_{\theta}(x)_{y}$，分数熵 $L_{SE}$ 定义为：
$$
L_{SE}=\mathbb{E}_{x\sim p} \left[\sum_{y\neq x}w_{xy}\left(s_{\theta}(x)_y-\frac{p(y)}{p(x)}\log s_{\theta}(x)_y+K\left(\frac{p(y)}{p(x)}\right)\right)\right]
\tag{6}
$$
其中 $K(a) = a(\log a - 1)$ 是归一化常数函数，确保 $L_{SE} \geq 0$。

分数熵本质是布雷格曼散度中令 $F=-\log$ 推出来的，不妨考虑标量情况（对于每个 $y$， $s_{\theta}(x)_{y}$ 和 $\frac{p(y)}{p(x)}$ 都是正标量），向量情况可以自然推广。

令凸函数 $F(u)=-\log u$ ，定义域 $u>0$，则有：
$$
\begin{aligned}
D_F(a,b)&=F(a)-F(b)-F'(b)(a-b)\\
&=\frac{a}{b}-\log a+\log b - 1
\end{aligned}
$$
两边同乘正数 $b$，则有：
$$
\begin{aligned}
b\cdot D_F(a,b)&=b\cdot \left(\frac{a}{b}-\log a+\log b - 1\right)\\\\
&=a-b\log a+b\log b -b\\\\
&=a-b\log a+b(\log b - 1)
\end{aligned}
$$
令 $K(b)=b(\log b - 1)$ ，有：
$$
b\cdot D_F(a,b)=a-b\log a+K(b)
$$
对比分数熵的公式 (5) ，可以对照出来：
$$
a=s_{\theta}(x)_{y},b=\frac{p(y)}{p(x)}
$$
即 $a$ 为**预测比率**，$b$ 为**真实比率**。把加权也融入进去，我们就有了：
$$
L_{SE}=\mathbb{E}_{x\sim p}\left[\sum_{y\neq x}w_{xy}\cdot \left(\frac{p(y)}{p(x)}\cdot D_F\left(s_{\theta}(x)_y,\frac{p(y)}{p(x)}\right)\right)\right] \tag{7}
$$
直观地理解一下，这个损失的核心就是散度 $D_F$ ，它表征了我们目标比例和实际比例的差异。

内层加权 $p(y)/p(x)$ 则是一个**重要性加权**，它放大了较大比率的预测误差的惩罚。回去看这玩意的用途我们也知道，它越大显然对 $\bar{Q}$ 的预测影响越大，这个重要性加权很合理。

外层加权 $w_{xy}$ 是基于扩散过程的，一般就取 $w_{xy}=Q_t(x,y)$ ，它引导模型更关注在前向过程中更可能发生的转移，也是一个**重要性加权**。

**一个核心+两个重要性加权**，共同构成了分数熵损失。

#### 3.3 分数熵的优良性质

由于分数熵损失基于布雷格曼散度，因此它也继承了一些优异性质。

##### 3.3.1 非负、对称且凸

非负直接由布雷格曼散度带来的；对称指的是任意状态 $x$ 和 $y$ 对称；而凸性指的是给定真实比率 $b$ ，对于预测值 $a$ 来说，布雷格曼散度是凸的：
$$
\mathcal{l}(a;b)=a-b\log a+K(b)
$$
这种分量凸性保证了在**神经网络最后一层**是凸的，为前面所有复杂且非凸的网络层提供了一个干净可靠的梯度来源，也保证了最后一层的判断是可靠的。

##### 3.3.2 一致性

其实就是在说最理想的条件下，最小化分数熵损失函数所得到的最优模型，其输出就是我们想要学习的真实目标。具体可以描述为：

分布 $p$ 具有完全支撑集，且对于所有成对的 $x\neq y$ ，有 $w_{xy}>0$ （也就是对所有的可能的样本都考虑、都关注），则当样本容量和模型容量趋于无穷时，通过最小化分数熵损失 $L_{SE} $ 所得到的最优参数 $\theta ^*$ 所对应的分数网络，对于所有的 $x,y(y\neq x)$ 都满足：
$$
s_{\theta^*}(x)_y=\frac{p(y)}{p(x)}
$$
且在最优处，分数熵损失为零：$L_{SE}(\theta^* )=0$

**证明过程**：

其实就是直接用布雷格曼散度的性质，回忆一下分数熵损失的定义（公式 (7)）：
$$
L_{SE}=\mathbb{E}_{x\sim p}\left[\sum_{y\neq x}w_{xy}\cdot \left(\frac{p(y)}{p(x)}\cdot D_F\left(s_{\theta}(x)_y,\frac{p(y)}{p(x)}\right)\right)\right]
$$
这里可以利用布雷格曼散度的定义和性质（由凸函数 $F$ 保证）：

+ $D_F(a,b)\geq 0$ 对任意 $a,b$ 都成立
+ $D_F(a,b)=0$ **当且仅当** $a=b$

由于 $w_{xy}>0$ 且 $\frac{p(y)}{p(x)}>0$ ，那么求和中的某一项为 $0$ ，当且仅当 $D_F\left(s_{\theta}(x)_y,\frac{p(y)}{p(x)}\right)=0$，这就等价于 $s_{\theta}(x)_y=\frac{p(y)}{p(x)}$ 

而对于全局最优，那么应该**每一项**求和项都等于 $0$，因此刚刚提到的条件应该对任意 $x,y$ 都成立。

因此，最优解 $\theta^* $ 必须对于任意 $x,y$ ，都有$s_{\theta^*}(x)_y=\frac{p(y)}{p(x)}$， 测试 $L_{SE}(\theta^*)=0$ ，证明完成。

##### 3.3.3 自适应末梯度调节

这里主要讨论的是在神经网络最后一层的差异。仍然让 $a$ 为预测梯度，$b$ 是真实梯度。前面提到的具体分数匹配（CSM）的分量损失为：
$$
l_{\text{CSM}}(a;b)=\frac{1}{2}(a-b)^2
$$
对于预测值 $a$ 的梯度为：
$$
\nabla_al_{\text{CSM}}=a-b
$$
而分数熵的分量损失为：
$$
&l_{\text{SE}}(a;b)=a-b\log a+K(b)\\
$$
进一步求梯度：
$$
\begin{aligned}
\nabla_al_{SE}&=1-\frac{b}{a}\\
&=\frac{1}{a}(a-b)\\
&=\frac{1}{a}\cdot \nabla_al_{\text{CSM}}
\end{aligned}
$$
总损失是这些分量损失的加权和，因此该梯度关系在整体损失层面仍然成立，使得对于当预测 $a$ 很大时，往前传的梯度也不至于太大；而当预测 $a$ 很小时，反而会放大这个效果。这种自适应调节就是分数熵相比具体分数匹配的优势之一。

#### 3.4 隐式分数熵 （Implicit Score Entropy， ISE）

然而上面的推导还存在一个问题，我们很难知道真实比率是多少。真实比率定义是：
$$
\frac{p_t(y)}{p_t(x)}=\frac{\text{在时刻t，所有数据中处于状态y的概率}}{\text{在时刻t，所有数据中处于状态x的概率}}
$$
两两组合+时刻要求，产生了维数灾难。而且训练集中到具体的情况很可能极其稀疏，导致用频率估计概率的误差极大，因此我们需要进一步改造损失函数，把这一项用其他东西来表示。隐式分数熵就是一种方式，其改造结果为：
$$
\mathcal{L}_{\text{ISE}}=\mathbb{E}_{x\sim p}
\left[
\sum_{y\neq x}w_{xy}s_{\theta}(x)_y-w_{yx}\log s_{\theta}(y)_x \right]
$$

> 注意，网络 $s_{\theta}(\cdot)$ 是输入一个状态 $x$，输出一个 $|X|$ 维向量，$s_{\theta}(x)_y$ 表示第 $y$ 个分量，即 $p(y)/p(x)$ 的估计。

这个损失的第一项直接过一遍正向传播就能拿到，但是第二项就很难计算，我们需要对所有的 $y\neq x$ ，都跑一遍正向传播，才能拿到完整的结果（或者跑蒙特卡洛估计）。这意味着算一次Loss，就要跑少说上千次正向传播，这纯扯淡，得考虑其他方法。

#### 3.5 去噪分数熵（Denoising Score Entropy，DSE）

可以证明，公式 (6) 和 (7) 表示的 $L_{SE}$ 与下述形式等效：
$$
\mathcal{L}_{SE}=\mathbb{E}_{x\sim p} \sum_y w_{xy}
\left[K(s_{\theta}(x)_y)-K\left(\frac{p(y)}{p(x)}\right)\right]
\tag{8}
$$
将分布 $p$ 用基分布 $p_0$ 通过转移核 $p(\cdot|x_0)$ 扰动表出：
$$
p(x)=\sum_{x_0}p(x|x_0)p_0(x_0)
$$
具体落到扩散过程中有：
$$
p_t(x)=\sum_{x_0}p_t(x|x_0)p_0(x_0)
$$
带回到上面的式子中：
$$
\begin{aligned}
\mathcal{L}_{SE}&=\mathbb{E}_{x\sim p} \sum_y w_{xy}
\left[K(s_{\theta}(x)_y)-K\left(\frac{p(y)}{p(x)}\right)\right]\\\\
&=\sum_x p(x)\sum_y w_{xy}
\left[K(s_{\theta}(x)_y)-K\left(\frac{p(y)}{p(x)}\right)\right]\\\\
&=\sum_x\sum_{x_0}p(x|x_0)p_0(x_0)\sum_y w_{xy}
\left[K(s_{\theta}(x)_y)-K\left(\frac{p(y)}{p(x)}\right)\right]\\\\
&=\mathbb{E}_{x_0\sim p_0,x\sim p(\cdot|x_0)}\sum_y w_{xy}
\left[K(s_{\theta}(x)_y)-K\left(\frac{p(y)}{p(x)}\right)\right]
\quad \rightarrow  \text{求和转双层期望}
\end{aligned}
$$

> 其中 $\mathbb{E}_{x_0\sim p_0,x\sim p(\cdot|x_0)}$ 表示先从分布 $p_0$ 中采样得到 $x_0$ ，在给定 $x_0$ 的条件下，从条件分布 $p(\cdot|x_0)$ 中采样得到 $x$，也就是**两阶段采样**后经过期望算子 $\mathbb{E}$ 得到结果。

这里采用一个trick：注意到函数 $K$ 内部实际上是常数，可以换成更容易计算的 $p(y|x_0)/p(x|x_0)$ ：
$$
\begin{aligned}
\mathcal{L}_{SE}
&=\mathbb{E}_{x_0\sim p_0,x\sim p(\cdot|x_0)}\sum_y w_{xy}
\left[K(s_{\theta}(x)_y)-K\left(\frac{p(y|x_0)}{p(x|x_0)}\right)\right]\\\\
&+\mathbb{E}_{x_0\sim p_0,x\sim p(\cdot|x_0)}\sum_y w_{xy}
\left[K\left(\frac{p(y|x_0)}{p(x|x_0)}\right)-K\left(\frac{p(y)}{p(x)}\right)\right]
\end{aligned}
$$
第一项记作 $L_{\text{DSE}}$ ，第二项与 $\theta$ 无关，可以记为常数 $C$，最终得到：
$$
L_{SE}=L_{DSE}+C
$$

$$
L_{DSE}=\mathbb{E}_{x_0\sim p_0,x\sim p(\cdot|x_0)}\sum_y w_{xy}
\left[K(s_{\theta}(x)_y)-
K\left(\frac{p(y|x_0)}{p(x|x_0)}\right)
\right] 
\\
\tag{9}
$$

如果直接用公式 (6) 进行推导，结果应该为：
$$
L_{DSE}=\mathbb{E}_{x\sim p} 
\left[\sum_{y\neq x}w_{xy}\left(s_{\theta}(x)_y-
\frac{p(y|x_0)}{p(x|x_0)}\log s_{\theta}(x)_y
+K\left(\frac{p(y|x_0)}{p(x|x_0)}\right)\right)
\right]
\tag{10}
$$
其实这里的 trick 我想更多是连续扩散过程启发的，用 $p(y|x_0)$ 可以说是有异曲同工之妙。

#### 3.6 离散扩散模型的似然上界

##### 3.6.1 参数化的反向扩散矩阵和密度演化过程

回顾一下公式(5) ：真实的反向扩散和密度演化过程
$$
\bar{Q}_t(y,x)=\begin{cases}
\displaystyle
-\sum_{z\neq x}\bar{Q}_t(z,x)&\quad ,x=y\\\\
\displaystyle
\frac{p_y(t)}{p_x(t)}Q_t(x,y) &\quad ,x\neq y
\end{cases}
\\
\frac{dp_{T-t}}{dt}=\bar{Q}_{T-t}p_{T-t}，\quad p_T\approx p_{\text{base}}
$$
里面的分布密度实际上用分数网络 $s_{\theta}(x,t)_y$ 来近似它，由此定义参数化的逆向扩散矩阵和密度演化：
$$
\bar{Q}^{\theta}_t(y,x)=\begin{cases}
\displaystyle
-\sum_{z\neq x}\bar{Q}_t(z,x)&\quad ,x=y\\\\
\displaystyle
s_{\theta}(x,t)_y\cdot Q_t(x,y) &\quad ,x\neq y
\end{cases}
\\
\frac{dp_{T-t}^{\theta}}{dt}=\bar{Q}^{\theta}_{T-t}p^{\theta}_{T-t}，\quad p^{\theta}_T\approx p_{\text{base}}
\tag{11}
$$
其中 $p_t^{\theta}$ 表示的是由我们的模型参数 $\theta$ 所定义的概率分布。我们的最终目的是让反向去噪得到的 $p_0^{\theta}$ 接近 $p_0$

##### 3.6.2  证据下界（ELBO）和扩散加权去噪分数熵

用极大似然估计，要让反向去噪得到的 $p_0^{\theta}$ 接近 $p_0$，等价于最小化负对数似然期望：
$$
\min _{\theta}\mathbb{E}_{x\sim p}[-\log p_0^{\theta}(x)]
$$
然而负对数似然不好算，对于任意一个样本点 $x_0$ ，论文中直接给出了一个便于计算的证据下界（ELBO）：
$$
-\log p_{0}^{\theta}(x_0)\leq \mathcal{L}_{\text{DWDSE}}(x_0)+D_{KL}(p_{T|0}(\cdot | x_0)||p_{\text{base}}) \tag{12}
$$
其中 $\mathcal{L}_{\text{DWDSE}}$ 是数据点 $x_0$ 的扩散加权去噪分数熵：
$$
\mathcal{L}_{\text{DWDSE}}=\int_{0}^{T}\mathbb{E}_{x_t\sim p_{t|0}(\cdot | x_0)}\sum_{y\neq x_t}Q_t(x_t,y)\left(s_{\theta}(x_t,t)_y-\frac{p_{t|0}(y|x_0)}{p_{t|0}(x_t|x_0)}\log s_{\theta}(x_t,t)_y+K\left(\frac{p_{t|0}(y|x_0)}{p_{t|0}(x_t|x_0)}\right)\right)dt
\tag{13}
$$
对比一下原版的 $\mathcal{L}_{DSE}$（公式 (10) ）：
$$
L_{DSE}=\mathbb{E}_{x\sim p} 
\left[\sum_{y\neq x}w_{xy}\left(s_{\theta}(x)_y-
\frac{p(y|x_0)}{p(x|x_0)}\log s_{\theta}(x)_y
+K\left(\frac{p(y|x_0)}{p(x|x_0)}\right)\right)
\right]
$$
相比之下，$\mathcal{L}_{\text{DWDSE}}$ 是真正面向 $[0,T]$ 这个扩散过程的，主要是以下几个不同：

+ 显式时间积分：对时间 $t$ 从 $0$ 到 $T$ 积分；
+ 权重具体化：直接面向扩散过程，就可以直接取权重 $w_{xy}=Q_t(x,y)$ ，好处在 3.2 中有提到；
+ 分数网络时间输入：分数网络显式依赖于时间 $t$ ，记为 $s_{\theta}(x_t,t)_y$；
+ 条件采样明确化：对于每个时间 $t$，从条件分布 $p_{t|0}(\cdot |x_0)$ 采样 $x_t$。

本质就是**面向扩散**。

---

#### 四、 从一个词到一句话：高维序列的建模

#### 4.1 基础的问题建模

之前都是针对一个文字（或者说一个 token），现在扩展到整个文本序列。对于长度为 $d$ 的序列整体的状态空间可以表示为：
$$
X=\{1,...,n\}^d
$$
状态空间大小 $|X|=n^d$，每个位置都可以取 $n$ 个可能的值，状态间的转移就是 $\mathbb{R}^{d\times n}\rightarrow \mathbb{R}^{d\times n}$。

考虑整个序列的速率矩阵 $Q_t$ ，其维数达到了 $n^d\times n^d$ ，直接训练根本不可行，因此必然要尝试稀疏化。

#### 4.2 针对单个 token 的扰动

##### 4.2.1 扰动单个 token 

假设任意时刻 $t$ 的速率矩阵为 $Q_t\in \mathbb{R}^{n^d \times n^d}$ 。同时对于每一个位置 $i$ ，定义 token 级速率矩阵 $Q^{\text{tok}}_t\in \mathbb{R}^{n\times n}$.

现在我们只允许在每个位置上独立地扰动 token。换句话说对于任意的两个序列：
$$
\mathbf{x} = x^1 \dots x^d \\
\mathbf{y} = y^1 \dots y^d
$$
当且仅当它们在恰好某**一个位置** $i$ 上不同时，才可能有非零的转移速率，此时状态 $\mathbf{x}$ 到状态 $\mathbf{y}$ 的转移速率，等价于 $x^i$ 到 $y^i$ 的转移速率：
$$
Q_t(\mathbf{x}, \mathbf{y}) = 
\begin{cases}
Q_t^{\text{tok}}(x^i, y^i) & \mathbf{x} \text{ 和 } \mathbf{y} 
\text{ 只有一个 token 不一样} \\
0 & \text{如果两个以上位置不同}
\end{cases}
$$
这样 $Q_t$ 只需要看汉明距离为 $1$ 的序列了。

##### 4.2.2 简化网络参数

考虑序列 $\mathbf{x}=x^1...x^i...x^d$ ，对 $x^i$ 进行扰动，得到 $\mathbf{\hat{x}}=x^1...\hat{x}^i...x^d$  

由于我们只关注汉明距离为1的序列，因此网络 $s_{\theta}(\cdot,t):\{1,...,n\}^d\rightarrow \mathbb{R}^{n^d}$ 可以简化为 $s_{\theta}(\cdot,t):\{1,...,n\}^d\rightarrow \mathbb{R}^{d\times n}$：
$$
s_{\theta}(\mathbf{x},t)=
\begin{bmatrix}
s_{1,1} & s_{1,2} & \dots & s_{1,n}\ \\
s_{2,1} & s_{2,2} & \dots & s_{2,n}\ \\
\vdots & \vdots & \ddots & \vdots \\
s_{d,1} & s_{d,2} & \dots & s_{d,n}
\end{bmatrix}
$$

$$
s_{\theta}(\mathbf{x},t)_{\mathbf{\hat{x}}}=s_\theta(x^1 \dots x^i \dots x^d, t)_{i, \hat{x}^i} \approx \frac{p_t(x^1 \dots \hat{x}^i \dots x^d)}{p_t(x^1 \dots x^i \dots x^d)}=
\frac{p_t(\mathbf{\hat{x}},t)}{p_{t}(\mathbf{x},t)}
$$
其中下标 $(i,\hat{x}^i)$ 的 $i$ 表示序列中的位置索引，$\hat{x}^i$ 表示该位置上的新值，整个表达式表示矩阵 $s_\theta(\mathbf{x}, t)$ 的**第 $i$ 行**、**第 $\hat{x}^i$ 列**的元素。

#### 4.3 前向转移概率的计算

回顾一下公式(13)：
$$
\mathcal{L}_{\text{DWDSE}}(\mathbf{x}_0) = \int_{0}^{T} \mathbb{E}_{\mathbf{x}_t\sim p_{t|0}(\cdot | \mathbf{x}_0)} 
\sum_{\mathbf{y} \neq \mathbf{x}_t} Q_t(\mathbf{x}_t, \mathbf{y})
\left(
s_{\theta}(\mathbf{x}_t, t)_{\mathbf{y}} - \frac{p_{t|0}(\mathbf{y}|\mathbf{x}_0)}{p_{t|0}(\mathbf{x}_t|\mathbf{x}_0)} \log s_{\theta}(\mathbf{x}_t, t)_{\mathbf{y}} + K\left( \frac{p_{t|0}(\mathbf{y}|\mathbf{x}_0)}{p_{t|0}(\mathbf{x}_t|\mathbf{x}_0)} \right)
\right) dt
$$
整个序列的状态预测为了计算损失函数 $\mathcal{L}_{\text{DWDSE}}$，我们需要前向转移概率 $p_{t|0}(\mathbf{y}|\mathbf{x_0})$。$\mathbf{y}$ 是由 $\mathbf{x_0}$ 一步步狸猫换太子换出来的，因此这个前向转移概率可以分解为：
$$
p_{t|0}({\mathbf{y}}|\mathbf{x}) = \prod_{i=1}^d p_{t|0}^{\text{tok}}(\hat{x}^i|x^i_0)
$$
接下来我们需要计算每个 token 级的转移概率 $p_{t|0}^{\text{tok}}(\hat{x}^i|x^i_0)$

**步骤1：从主方程开始**

专注于第 $i$ 个 token，考虑这个 token 所有转移可能，因此定义向量：
$$
\mathbf{p}_t = [\ p^{\text{tok}}_{t|0}(1|x^i_0),\quad p^{\text{tok}}_{t|0}(2|x^i_0),\quad ...,\quad p^{\text{tok}}_{t|0}(n|x^i_0)\ ]^\top
$$
这就是连续马尔可夫下的众生百态，引用公式 (1) 则有：
$$
\frac{d\mathbf{p}_t}{dt} = Q^{\text{tok}}_t\  \mathbf{p}_t
$$
初始条件自然就是位于 $x_0^i$ 这个状态，对应的状态向量：$\mathbf{p}_0 = e_{x^i_0}$（one-hot）

**步骤2：分离时间项**

现在分离时间项，设 $Q_t^{\text{tok}}= \sigma(t) Q^{\text{tok}}$，其中 $\sigma(t)$ 是噪声调度，$Q^{\text{tok}}$ 是固定的转移矩阵。定义累积噪声强度：
$$
\tau = \bar{\sigma}(t) = \int_0^t \sigma(s) ds
$$
把主方程的时间变量换一下：
$$
\frac{d\mathbf{p}_t}{dt} = \frac{d\mathbf{p}_t}{d\tau} \cdot \frac{d\tau}{dt} = \frac{d\mathbf{p}_t}{d\tau} \cdot \sigma(t)
$$
原方程变为：
$$
\begin{aligned}
&\frac{d\mathbf{p}_t}{d\tau} \cdot \sigma(t) = \sigma(t) Q^{\text{tok}}\  \mathbf{p}_t\\\\
\Longrightarrow \quad&\frac{d\mathbf{p}_t}{d\tau}  =  Q^{\text{tok}} \mathbf{p}_t\\\\
\Longrightarrow \quad&\mathbf{p}_t=\exp(\tau Q^{\text{tok}})\mathbf{p}_0 \quad\rightarrow\text{该微分方程可直接求通解}\\\\
\Longrightarrow \quad& \mathbf{p}_t = \exp(\bar{\sigma} Q^{\text{tok}}) e_{x^i} \quad\rightarrow\text{带回}\\\\
\end{aligned}
$$
矩阵 $\exp(\bar{\sigma} Q^{\text{tok}})$ 乘以单位列向量 $e_{x^i_0}$ 的结果，自然就是 $\exp(\bar{\sigma} Q^{\text{tok}})$ 的第 $x^i_0$ 列。

如果要找一个确定的变化后状态 $\hat{x}^{i}$，那么就有：
$$
p_{t|0}^{\text{tok}}(\hat{x}^i | x_0^i) = \left[\exp\left(\bar{\sigma}(t) Q\right)\right]_{\hat{x}^i, x_0^i} \tag{14}
$$
$[\cdot]_{\hat{x}^i,x_{0}^i}$ 表示第 ${x_0^i}$ 列，第 $\hat{x}^i$ 行。换一种表达，就是变化后的 token 状态 $\hat{x}^i$ 服从于该分布：
$$
\hat{x}^i\sim p_{t|0}^{\text{tok}}(\cdot | x_0^i)=[\exp(\bar{\sigma }(t)Q^{\text{tok}})]_{x_0^i} \tag{15}
$$

#### 4.4 速率矩阵的设计

$Q^{\text{tok}}$ 维度仍然达到了 $n\times n$，对于词表较大的语言生成任务，存储和计算完整的 $Q^{\text{tok}}$ 矩阵显存可能得几十个 GB 往上，访问速度还很慢。因此这里分别选用了两种特殊结构矩阵：

##### 4.4.1 均匀速率矩阵（Uniform Transtion）

这种方式对应的是均匀扩散，拓扑上是一种全连接图结构，每个状态都以相同的速率跳转到其他状态：
$$
Q^{\text{uniform}} = 
\begin{bmatrix}
1-n & 1 & \cdots & 1 \\
1 & 1-n & \cdots & 1 \\
\vdots & \vdots & \ddots & \vdots \\
1 & 1 & \cdots & 1-n
\end{bmatrix}
=\mathbf{1}_{n\times n} - n\mathbf{I}_n
\in \mathbb{R}^{n \times n}
$$
其中 $\mathbf{1}$ 是全 1 矩阵，$\mathbf{I}$ 是单位矩阵。

带回前面的公式 (15) ，得到：（具体推导过程 TODO）
$$
x^i_k\sim \frac{e^{\bar{\sigma}(t)}-1}{ne^{\bar{\sigma(t)}}}\mathbf{1}+e^{-\bar{\sigma}(t)}e_{x_0^i}
$$

##### 4.4.2 吸收转移矩阵（Absorbing Transition）

这种比较特殊，词汇表扩展到 $N+1$ ，包含 `[MASK]`，所有状态都在逐步变成噪声噪声：
$$
Q^{\text{absorb}} = 
\begin{bmatrix}
-1 & 0 & \cdots & 0 & 0 \\
0 & -1 & \cdots & 0 & 0 \\
\vdots & \vdots & \ddots & \vdots & \vdots \\
0 & 0 & \cdots & -1 & 0 \\
1 & 1 & \cdots & 1 & 0
\end{bmatrix}
\in \mathbb{R}^{(n+1) \times (n+1)}
$$
其中前 $n$ 行对应原始词汇，第 $n+1$ 行对应 MASK 状态。

同样带回公式 (15)，得到：（具体推导过程 TODO）
$$
x_t^i\sim  e^{-\bar{\sigma}(t)}e_{x_0^i}+(1-e^{-\bar{\sigma}(t)})e_{\text{MASK}}
$$
其中 $e_{\text{MASK}}$ 也是 one-hot 向量。

#### 4.5 $\mathcal{L}_{\text{DWDSE}}$ 的进一步改进

原来的 $\mathcal{L}_{\text{DWDSE}}$ 是：
$$
\mathcal{L}_{\text{DWDSE}}= \int_{0}^{T} \mathbb{E}_{\mathbf{x}_t\sim p_{t|0}(\cdot | \mathbf{x}_0)} 
\sum_{\mathbf{y} \neq \mathbf{x}_t} Q_t(\mathbf{x}_t, \mathbf{y})
\left(
s_{\theta}(\mathbf{x}_t, t)_{\mathbf{y}} - \frac{p_{t|0}(\mathbf{y}|\mathbf{x}_0)}{p_{t|0}(\mathbf{x}_t|\mathbf{x}_0)} \log s_{\theta}(\mathbf{x}_t, t)_{\mathbf{y}} + K\left( \frac{p_{t|0}(\mathbf{y}|\mathbf{x}_0)}{p_{t|0}(\mathbf{x}_t|\mathbf{x}_0)} \right)
\right) dt
$$
在实际的算法中，采用的是：
$$
\hat{\mathcal{L}}_{DWDSE}=\sigma(t)\sum_{i=1}^d\sum_{y\neq x_t^i}\left(s_{\theta}(\mathbf{x}_t,t)_{i,y}-\frac{p_{t|0}(y|x_0^i)}{p_{t|0}(x_t^i|x^i_0)}\log s_{\theta}(\mathbf{x}_t,t)_{i,y}\right) \tag{16}
$$
主要几个改进是：

+ 省略与参数 $\theta$ 无关的常数项 $K(\cdot)$ ；

+ 基于汉明距离为1的假设，将原有的对所有可能序列 $\mathbf{y}\neq \mathbf{x_t}$ 的求和（复杂度 $O(n^d)$ ），分解为了对每个位置 $i$ 和每个可能的 token 状态 $y$ 的求和：
  $$
  \sum_{\mathbf{y}\neq \mathbf{x}_t}Q_t(\mathbf{x_t},\mathbf{y})[\dots]
  =\sum_{i=1}^d \sum_{y\neq x_t^i}\sigma(t)Q^{\text{tok}}(x_t^i,y)[\dots]
  $$
  对于均匀矩阵来说，非对角线元素全都是 1 ，因此矩阵 $Q^{\text{tok}}$ 的值直接就是 1；对于吸收矩阵，则直接只剩下 MASK 是 1。这里感觉原论文图方便直接就按均匀矩阵写了。

+ 原版的网络换成了 4.2.2 简化后的网络，大大降低网络复杂度；

+ 前向传播过程的简化。回顾 4.3 提到的这个分解：
  $$
  p_{t|0}({\mathbf{x}_t}|\mathbf{x}_0) = \prod_{i=1}^d p_{t|0}({x}_t^i|x^i_0)
  $$
  而对于与 $\mathbf{x}_t$ 仅在位置 $i$ 不同的序列 $\mathbf{y}$ 有：
  $$
  p_{t|0}(\mathbf{y}|\mathbf{x}_0)=p_{t|0}(y|x_0^i)\prod_{j\neq i} p_{t|0}(x_t^j|x_0^j)
  $$
  因此序列层面的概率比简化为 token 级的概率比：
  $$
  \frac{p_{t|0}(\mathbf{y}|\mathbf{x}_0)}{p_{t|0}(\mathbf{x}_t|\mathbf{x}_0)}
  =
  \frac{p_{t|0}({y}|{x}_0^i)}{p_{t|0}({x}_t^i|{x}_0^i)}
  $$

+ 针对时间积分的蒙特卡洛。原始损失包含了对时间 $t$ 的积分，在实际训练中难以直接计算。这里用**蒙特卡洛近似**：每次迭代时，从均匀分布 $t\sim \mathcal{U}([0,T]) $ 中取样时间点作为整个积分的估计。

#### 4.5 SEDD 训练算法

这就是原论文的算法1。

输入包括：

+ 神经网络 $s_{\theta}$
+ 噪声调度 $\sigma$ 
+ 累积噪声 $\bar{\sigma}$ 
+ token 级的转移矩阵 $ Q $
+ 训练时间区间 $[0,T]$
+ 真实数据分布（训练集） $p_{\text{data}}$

具体过程：

+ 采样：$\mathbf{x}_0\sim p_{\text{data}}$，$t\sim \mathcal{U}([0,T])$

+ 从 $\mathbf{x}_0$ 中构建 $\mathbf{x}_t$ ，对每个分量来说： $x_t^i\sim p_{t|0}(\cdot | x_0^i)=\exp(\bar{\sigma}(t)Q)_{x_0^i}$

+ 判断矩阵 $Q$：

  + 如果 $Q$ 是 Absorb矩阵：
    $$
    x^i_t\sim e^{-\bar{\sigma}(t)}e_{x_0^i}+(1-e^{-\bar{\sigma}(t)})e_{\text{MASK}}
    $$

  + 如果 $Q$ 是 Uniform矩阵：
    $$
    x^i_t\sim \frac{e^{\bar{\sigma}(t)-1}}{ne^{\bar{\sigma(t)}}}\mathbb{1}+e^{-\bar{\sigma}(t)}e_{x_0^i}
    $$

+ 计算：
  $$
  \hat{\mathcal{L}}_{DWDSE}=\sigma(t)\sum_{i=1}^d\sum_{y=1}^n(1-\delta_{x_t^i}(y))\left(s_{\theta}(\mathbf{x}_t,t)_{i,y}-\frac{p_{t|0}(y|x_0^i)}{p_{t|0}(x_t^i|x^i_0)}\log s_{\theta}(\mathbf{x}_t,t)_{i,y}\right)
  $$

+ 反向传播 $\nabla_\theta\hat{\mathcal{L}}_{DWDSE}$ ，更新参数。

---

### 五、反向去噪

现在假设我们已经知道了 $s_{\theta} $ ，如何进行反向去噪呢？

#### 5.1 欧拉法

第一种方法是欧拉法，就是公式 (4)：
$$
p(x_t=x|x_{t+\Delta t}=y)=\delta_{xy}+\bar{Q}_t(x,y)\Delta t +O(\Delta t^2)
$$
把他推广到序列上，给定序列 $\mathbf{x}_t$ ，通过独立采样每个位置上的新 token $x_{t-\Delta t}^i$ 来构建 $\mathbf{x}_t-{\Delta t}$ 。最终算法里用的式子是：
$$
p^{i}(y|x_t^i)=\delta_{x_t^i}(y)+\Delta tQ_{t}^{\text{tok}}(x_t^i,y)s_{\theta}(\mathbf{x}_t,t)_{i,y} \tag{17}
$$
在实际使用的时候需要把负值截断成0，然后重新归一化。

这里存在几个问题：

+ 这里忽略了二次项，因此只适用于小步长 $\Delta t$ 
+ 这种方法实际上是串行的，因为 $Q$ 的结构导致每一步只能修改一个位置，效率很低。$s_{\theta} $ 也没有被用到最佳。

#### 5.2  Tweedie $\tau$-跳跃

如果 $p_t$ 遵循 $dp_t/dt=Qp_t$ ，则真实去噪器可以直接给出：
$$
p_{0|t}(\mathbf{x}_0|\mathbf{x}_t)=\left(\exp{(-tQ)\left[\frac{p_t(\mathbf{y})}{p_t(\mathbf{x}_t)}\right]_{\mathbf{y}=1}^N}\right)_{\mathbf{x}_0}\exp(tQ)(\mathbf{x}_t,\mathbf{x}_0)
$$
存在一个问题，这里需要拿到所有的概率比 ${p_t(\mathbf{y})}/{p_t(\mathbf{x}_t)}$，但我们只知道汉明距离为1的状态间的概率比，因此论文提出了一种 Tweedie $\tau$-条约的方法。

具体来说，将 token 转移概率替换为：
$$
p^i(y|x_t^i)=
(\exp(-\sigma_{t}^{\Delta t}Q)s_{\theta}(\mathbf{x}_t,t)_{i})_{y}
\exp (\sigma_{t}^{\Delta t}Q)(x_t^i,y) \tag{18}
$$
其中：

+ $ p^i(y|x_t^i) $ 表示第 $ i $ 个位置，$x_t^i$ 变成状态 $y$ 的概率

+ $\sigma_t^{\Delta t}=\bar{\sigma}(t)-\bar{\sigma}(t-\Delta t),\ \bar{\sigma}(t)$ 为累积噪声。其表示的是反向时间步 $\Delta t$ 内积累的噪声的变化量。
+ $\exp (\sigma_{t}^{\Delta t}Q)(x_t^i,y)$ 表示取矩阵中的第 $x_t^i$ 行 $y$ 列元素，即从当前 token 到目标 token 的前向扩散概率。

#### 5.3 分数熵采样：无条件注入

这里是原论文的算法2：

输入：

- 网络 \(s_\theta\)
- 噪声规划 \(\sigma\)（累积噪声 \( \overline{\sigma} \)）
- 令牌转移矩阵 \(Q\)
- 时间区间 \([0, T]\)
- 步长 \(\Delta t\)

步骤：

1. 通过从 \(Q\) 的平稳分布中采样每个 \(x_T^i\) 来采样 \( x_T \sim p_{\text{base}} \)。

2. 设 \( t \leftarrow T \)。

3. 当 \( t > 0 \) 时，执行以下操作：

   - 如果使用欧拉方法，则：

     - 构建转移密度：
       $$
       p^i(y|x_t^i) = \delta_{x_t^i}(y) + \Delta t Q_i^{\text{tok}}(x_t^i, y) s_\theta(x_t, t)_{i,y}
       $$

   - 否则如果使用Tweedie去噪，则：

     - 构建转移密度：
       $$
       p^i(y|x_t^i) =  \left( \exp(\overline{\sigma}(t - \Delta t) - \overline{\sigma}(t)) Q \right) s_\theta(x_t, t)_i )_y
       
       \exp((\overline{\sigma}(t) - \overline{\sigma}(t - \Delta t)) Q)(x_t^i, y)
       $$

   - 对 \( p^i(\cdot|x_t^i) \) 进行归一化（将值限制为最小0，如果需要，重新归一化总和为1）。

   - 对所有 \(i\)，从 \( p^i(y|x_t^i) \) 中采样 \(x_{t-\Delta t}^i\)，构建 \(x_{t-\Delta t}\) 从 \( x_t^i - \Delta t \)。

   - 设 \( t \leftarrow t - \Delta t \)。

4. 结束循环。

5. 返回：\(x_0\)。

#### 5.4 填充问题

考虑部分序列已知，生成剩余未知部分的任务。假设整个序列长度为 $d$ ，索引集合 $\{1,2,...,d\}$ 划分成两部分：

+ $ \Omega  $ 是未填充的序列索引，需要生成
+ $\bar{\Omega}$ 是已经填充的索引

已知 $\mathbf{y}=[y^1,y^2,...,y^{|\bar{\Omega}|}]$ 是 $\bar{\Omega }$ 位置上的值，需要生成 $ \Omega  $ 位置上的值 ：
$$
p_t(\mathbf{x}^{\Omega}|\mathbf{x}^{\bar{\Omega}}=y)
$$
对于任意两个不同的填充结果 $\mathbf{z}$ 和 $\mathbf{z'}$ ，贝叶斯定理告诉我们他俩的条件概率之比等于无条件概率之比：
$$
\frac
{p_t(\mathbf{x}^{\Omega}=\mathbf{z'}|\mathbf{x}^{\bar{\Omega}}=\mathbf{y})}
{p_t(\mathbf{x}^{\Omega}=\mathbf{z}|\mathbf{x}^{\bar{\Omega}}=\mathbf{y})}
=
\frac
{p_t(\mathbf{x}=\mathbf{z'}⊕_{\Omega}\mathbf{y})}
{p_t(\mathbf{x}=\mathbf{z}⊕_{\Omega}\mathbf{y})}
\tag{19}
$$
其中 $\oplus_{\Omega}$ 是 $ \Omega  $ 和 $\bar{\Omega}$ 的拼接，就是全序列。

因此在考虑更新位置 $i\in \Omega$ 时，需要估计：
$$
\frac{p_t({\mathbf{x}_t^{i\rightarrow y}}|\mathbf{x}^{\bar{\Omega}}=y)}{p_t({\mathbf{x}}_t|\mathbf{x}^{\bar{\Omega}}=y)}
=
\frac{p_t({\mathbf{x}_t^{i\rightarrow y}})}{p_t({\mathbf{x}}_t)}
$$
这正是分数网络要估计的。

#### 5.5 分数熵采样：有条件注入

这是原论文的算法3

输入：

- 网络 \(s_\theta\)
- 噪声规划 \(\sigma\)（累积噪声 \( \overline{\sigma} \)）
- 令牌转移矩阵 \(Q\)
- 时间区间 \([0, T]\)
- 步长 \(\Delta t\)
- Prompt 空间 \( \Omega  \) 和 token \(T\)。

步骤：

1. \( x_T \sim p_{\text{base}} \) 如上所述。将 \( \Omega  \) 中的所有索引设置为 \(T\) 中对应的token

2. 设 \( t \leftarrow T \)。

3. 当 \( t > 0 \) 时，执行以下操作：

   + 使用算法2的方式，为所有 \( i \) 构建转移密度 \( p^i(y|x_t^i) \)。

   - 仅对 \( i \notin \Omega \) 的所有 \( i \)，从 \( p^i(y|x_t^i) \) 中采样 \(x_{t-\Delta t}^i\)。否则，对于 \( i \in \Omega \)，设置 \( x_{t-\Delta t}^i \leftarrow x_t^i \)。从 \(x_{t-\Delta t}^i\) 构建 \(x_{t-\Delta t}\)。

   - 设 \( t \leftarrow t - \Delta t \)。

4. 返回 $x_0$

其实没有什么东西，只是改成只填充空的地方。