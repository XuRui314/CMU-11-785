&ensp; In this HW, i'm gonna to complete the basic MLP (aka the simplest NN) by designing my own `mytorch` library which can be reused in the subsequent HW. 

&ensp; Here is the layout of this note (blog):

1. Brief Introduction
2. Python Implementation
3. Torch Pipeline

## Introduction

### Representation

&ensp; MLPs are universal function approximators , they can model any Boolean function, classification function, or regression. Now, I will explain this powerful representation ability in an intuitive way.

&ensp; First, MLPs can be viewed as universal Boolean functions, because perceptrons can model`AND `, `OR`, `NOT` gate. Any truth table can be expressed in one-hidden-layer MLP, if we just get every item from the table and compose an expression, it costs $2^{N-1}$ neurons.

&ensp; But what if using more layers instead of just one-hidden-layer, by increasing the depth of the network, we are able to use less perceptrons, because it captures the relations between neurons.

&ensp; The decision boundary captured by the boolean network can be any convex region, and can be composed together by adding one more layer to get a bigger maybe non-convex region. In this way, we can get any region we want in theory. 

<img src=".\image\HW1P1_image1.png" alt="image-20221215153529805" style="zoom:80%;" />

&ensp;But can one-hidden-layer MLP model all the decision boundary, in another word, is one-hidden-layer MLP a universal classifier?

&ensp; The answer is YES, the intuition behind this is as follows. Consider now we want to model $\sum_{i=1}^Ny_i\geq N$, when we push $N$ to infinite, we will get a circle region in 2d, and a cylinder in 3d.

<img src=".\image\HW1P1_image2.png" alt="image-20221215153609086" style="zoom:80%;" />

&ensp; So in theory, we can achieve this:

<img src=".\image\HW1P1_image3.png" alt="image-20221215153753410" style="zoom:80%;" />

&ensp; But this is true only when N is infinite, this actually won't work in practice. However, deeper networks can require far fewer neurons– 12 vs. ~infinite hidden neurons in this example. And in many cases, increase the depth can help a lot, a special case for this is the sum-product problem, also can be viewed as kind of DP idea.

&ensp; How can MLP model any function? The intuition is as behind:

<img src=".\image\HW1P1_image4.png" alt="image-20221215154409190" style="zoom:80%;" />



<img src=".\image\HW1P1_image5.png" alt="image-20221215154523328" style="zoom:80%;" />

&ensp; The network is actually a universal map from the entire domain of input values to the entire range of the output activation

&ensp; Next i will cover the trade-off between width, depth and sufficiency of architecture. Not all architectures can represent any function. If your layers width is narrow, and without proper activation function, the network may be not sufficient to represent some functions. However, narrow layers can still pass information to subsequent layers if the activation function is sufficiently graded. But will require greater depth, to permit later layers to capture patterns.

&ensp; In a short summary, deeper MLPs can achieve the same precision with far fewer neurons, but must still have sufficient capacity:

- The activations must pass information through 

- Each layer must still be sufficiently wide to convey all relevant information to subsequent layers.

&ensp; There are some supplement material analyzing the "capacity" of a network using VC dimension. I don't want to cover them all here. :)

&ensp; Then here let me introduce a variant network called RBF network. If you know Kernel SVM before, you must be familiar with the word RBF. So a RBF network is like this:

<img src=".\image\HW1P1_image6.png" alt="image-20221215183217223" style="zoom:80%;" />



&ensp; The difference between RBF network and BP network lies on the way they combine weights and inputs, see more here [link](https://stats.stackexchange.com/a/228596/363445).

&ensp; RBF network is the best approximation to continuous funcitons:

<img src=".\image\HW1P1_image9.png" style="zoom:80%;" />

&ensp; See more material on [RBFNN and SVM](https://www.cnblogs.com/pinking/p/9349695.html), [RBFNN and GMM](https://qr.ae/prpqSu)

### Learning theory

&ensp; We know that MLP can be constructed to represent any function, but there is a huge gap between "can" and "how to". One naive approach is to handcraft a network to satisfy it, but only for the simplest case. More generally, given the function to model, we want to derive the parameters of the network to model it, through computation. We learn networks (The network must have sufficient capacity to model the function) by “fitting” them to training instances drawn from a target function. Estimate parameters to minimize the error between the target function $g(X)$ and the network function $f(X,W)$.

<img src=".\image\HW1P1_image10.png" alt="image-20221217160346417" style="zoom:80%;" />

&ensp; But $g(X)$ is unknow, we only get the sampled input-output pairs for a number of samples of input $X_i$ , $(X_i, d_i)$ , where $d_i = g(X_i) + noise$. We must learn the entire function from these few “training” samples.

<img src=".\image\HW1P1_image11.png" alt="image-20221217160601273" style="zoom:80%;" />

&ensp; For classification problem, there's an old method called perceptorn algorithm. So i'll only cover the main idea here, which is the update rule. Everytime we mis-classified a data point, we adjust our $W$ vector to fit this point. The detailed proof can be found in mit 6.036.

<img src=".\image\HW1P1_image12.png" alt="image-20221217161120280" style="zoom:80%;" />

<img src=".\image\HW1P1_image13.png" alt="image-20221217161323356" style="zoom:80%;" />

<img src=".\image\HW1P1_image14.png" alt="image-20221217161338657" style="zoom:80%;" />

&ensp; So can we apply the perceptron algorithm idea to the training process of MLP using the threshold function? The answer is NO. Even using the perfect architecture, it will still cost exponential time because we are using threshold function, so nobody tells us how far is it to the right answer, we have to try out every possible combinations.

&ensp; Suppose we get every perceptron right except for the yellow cricle one, we need to train it to get the line as follows:

<img src=".\image\HW1P1_image15.png" alt="image-20221217163051488" style="zoom:80%;" />

&ensp; The individual classifier actually requires the kind of labelling shown below which is not given. So we need to try out every possible way of relabeling the blue dots such that we can learn a line that keeps all the red dots on one side

<img src=".\image\HW1P1_image16.png" alt="image-20221217164916738" style="zoom:80%;" />

<img src=".\image\HW1P1_image17.png" alt="image-20221217165158501" style="zoom:80%;" />

<img src=".\image\HW1P1_image18.png" alt="image-20221217165331580" style="zoom:80%;" />

&ensp; So how to get rid of this limitation? In fact, it costs people more than a decade to give the solution XD. The problem is binary error metric is not useful, there is no indication of which direction to change the weights to reduce error, so we have to try out every possibility.

&ensp; The solution is to change our way of computing the mismatch such that modifying the classifier slightly lets us know if we are going the right way or no.

- This requires changing both, our activation functions, and the manner in which we evaluate the mismatch between the classifier output and the target output 
- Our mismatch function will now not actually count errors, but a proxy for it

&ensp; So we need our mismatch function to be differentiable. Small changes in weight can result in non-negligible changes in output. This enables us to estimate the parameters using gradient descent techniques.

&ensp; **Come back** to the problem of learning, we use divergence ( actually a Functional, take function as input, output a number ) to measure the mismatch, which is an abstract compared with the error defined before. Because when we are talking about "Error", this is often referred to the data-point-wise terminology, and we use "Total Error" to measure the overall dismatch. But when we step into the probability distribution world, this is not sufficient enough, so we proposed the concept of divergence, just an abstract of the "Error" we talked before. 

&ensp; Divergence is a functional, because we want to measure the difference between functions given the input. We don't really care about the input $X$ here, because we just use all the input values to calculate the divergence. We are more concerned about the relation between output of the divergence and the weights( which determines how our $f$ changes). 

<img src=".\image\HW1P1_image19.png" alt="image-20221217171018978" style="zoom:80%;" />

&ensp;More generally, assuming $X$ is a random variable, we don't need to consider the range we never cover, so we introduce the expectation to get the best $W$:
$$
\begin{aligned}
\widehat{\boldsymbol{W}}= & \underset{W}{\operatorname{argmin}} \int_X \operatorname{div}(f(X ; W), g(X)) P(X) d X \\
& =\underset{W}{\operatorname{argmin}} E[\operatorname{div}(f(X ; W), g(X))]
\end{aligned}
$$
&ensp; We used the concept "Risk" associated with hypothesis $h(x)$, which is defined as follows:
$$
R(h)={\mathbf  {E}}[L(h(x),y)]=\int L(h(x),y)\,dP(x,y).
$$
&ensp; In practice, we can only get few sampled data, so we need to define "Empirical risk":
$$
{\displaystyle \!R_{\text{emp}}(h)={\frac {1}{n}}\sum _{i=1}^{n}L(h(x_{i}),y_{i}).}
$$
<img src=".\image\HW1P1_image21.png" alt="image-20221217180102605" style="zoom:80%;" />

<img src=".\image\HW1P1_image20.png" alt="image-20221217194520685" style="zoom:80%;" />

&ensp;Its really a measure of error, but using standard terminology, we will call it a “Loss” . The empirical risk is only an empirical approximation to the true risk which is our actual minimization objective. For a given training set the loss is only a function of W.

&ensp;  Breif summary: We learn networks by “fitting” them to training instances drawn from a target function. Learning networks of threshold-activation perceptrons requires solving a hard combinatorial-optimization problem.Because we cannot compute the influence of small changes to the parameters on the overall error. Instead we use continuous activation functions with non-zero derivatives to enables us to estimate network parameters. This makes the output of the network differentiable w.r.t every parameter in the network– The logistic activation perceptron actually computes the a posteriori probability of the output given the input. We define differentiable divergence between the output of the network and the desired output for the training instances. And a total error, which is the average divergence over all training instances. We optimize network parameters to minimize this error--Empirical risk minimization. This is an instance of function minimization 

### Backpropagation

&ensp; In the previous part, we have talked about Empirical risk minimization, which is a kind of function minimization. So it's natural to use gradient decent to optimize the objective function. The big picture of training a network(no loops, no residual connections) is described as the following. 

<img src=".\image\HW1P1_image22.png" alt="image-20221224210413766" style="zoom:80%;" />

&ensp;As you see,  the first step of gradient decent is to calculate the gradient of the loss value with respect to the parameters. In this section, I will introduce backpropagatoin, which is a really efficient way to calculate gradient on the network.

&ensp; Our goal is to calculate $\frac{d \boldsymbol{D i v}(Y, \boldsymbol{d})}{d w_{i, j}^{(k)}}$ on the network, the naive approach is to just calculate one by one without using the properties of a network, which costs unacceptable computation. A more reasonable way is to take the idea of chain rule and dynamic programming into consideration, which is backpropagation.

&ensp;Here are the **assumptions** (All of these conditions are frequently not applicable): 

1. The computation of the output of one neuron does not directly affect computation of other neurons in the same (or previous) layers 
2. Inputs to neurons only combine through weighted addition 
3. Activations are actually differentiable  

&ensp; So now let's work on the math part. Actually, we just need to figure out one layer's  computation, since the other layers' calculations are actually similar.  I will choose to start from the end of the network.

<img src=".\image\HW1P1_image23.png" alt="image-20221224214406340" style="zoom:80%;" />

&ensp; Start from the grey box (loss function calculation):
$$
\frac{\partial \operatorname{Div}(Y, d)}{\partial y_i^{(N)}}=\frac{\partial \operatorname{Div}(Y, d)}{\partial y_i}
$$
&ensp;Then we walk through the activation function:
$$
\frac{\partial D i v}{\partial z_1^{(N)}}=\frac{\partial y_1^{(N)}}{\partial z_1^{(N)}} \frac{\partial D i v}{\partial y_1^{(N)}}=f_N^{\prime}\left(z_1^{(N)}\right) \frac{\partial D i v}{\partial y_1^{(N)}}
$$
<img src=".\image\HW1P1_image24.png" alt="image-20221224220206074" style="zoom:80%;" />

&ensp;We get the desired gradient on layer N, Yeah~. But it's not the time for cheering up, we have to move forward because we just computed one layer gradients, the backpropagation is still continuing. So we also have to calculate the derivative with respect to $y_i^{N-1}$.

<img src=".\image\HW1P1_image25.png" alt="image-20221224221058953" style="zoom:80%;" />

&ensp; Afer iteratively calculating gradients like this till the last layer, backpropagation is finished. So let me take a brief summary bellow:

<img src=".\image\HW1P1_image26.png" alt="image-20221224221204602" style="zoom:80%;" />

<img src=".\image\HW1P1_image27.png" alt="image-20221224213345087" style="zoom:80%;" />

&ensp;In our assumptions, the activation function is scale-wise and all the fucntions are differentiable, but we don't get the two conditions in many cases. So for vector activation function, we need to do one more summation, and instead of directly using the gradient, we can choose subgradients sometimes.

<img src=".\image\HW1P1_image28.png" alt="image-20221224230903495" style="zoom:80%;" />

&ensp; The matrix form is as the following, and i think the only equation needed to be clarify is the term $\grad_{W_N}Div$, note that the dim of the derivative of a scaler with respect to a vector or matrix is the same as its transpose dim. And the dim of this expression looks resonable right? <img src=".\image\HW1P1_image31.png" alt="image-20221225220955510" style="zoom:80%;" />

&ensp;Now i'm gonna show you the math in detail. Our convention is to multiply gradient on the left, so to calculate the derivative  $\grad_{W_N}z_N$, we need to transpose the expression $z_N = W_Ny_{N-1} + b_N$ to get $z_N^T = y_{N-1}^TW_N^T + b_N^T$. By applying the chain rule, we can get:
$$
\grad_{W_N^T}Div = \grad_{z_N^T}Div\grad_{W_N^T}z_N^T=\grad_{z_N^T}Div \ y_{N-1}^T
$$
 &ensp; Adding transpose, we get:
$$
\grad_{W_N}Div = y_{N-1}\grad_{z_N}Div
$$
<img src=".\image\HW1P1_image30.png" alt="image-20221225214643166" style="zoom:80%;" />

<img src=".\image\HW1P1_image29.png" alt="image-20221225214508846" style="zoom:80%;" />

> 做了CMU的实验以后发现指导书写的和课堂上讲的不一样XD，课上讲的是分子布局，实验是分母布局，所以下面会把矩阵求导讲的全一点，争取切割这一节。

向量（或者标量）对向量的导数很简单，Jacobian就够了，这个我在HW0讲的比较细了，主要还是讨论一下矩阵对矩阵求导。接下来我讲的内容只是针对深度学习里的矩阵求导，因为可能不同领域对这块定义不一样，我也不太懂hh。

引用b乎大佬的回答，算是对矩阵求导做了一个定义：

来举个栗子吧，$A B=C$，$\left[\begin{array}{ll}a_1 & a_2 \\ a_3 & a_4\end{array}\right]\left[\begin{array}{ll}b_1 & b_2 \\ b_3 & b_4\end{array}\right]=\left[\begin{array}{ll}c_1 & c_2 \\ c_3 & c_4\end{array}\right]$
其中
$$
\left\{\begin{array}{l}
c_1=a_1 b_1+a_2 b_3 \\
c_2=a_1 b_2+a_2 b_4 \\
c_3=a_3 b_1+a_4 b_3 \\
c_4=a_3 b_2+a_4 b_4
\end{array}\right.
$$
那么
$$
\frac{\partial C}{\partial A} \Rightarrow\left\{\begin{array}{c}
\partial c_1 / \partial a_1=b_1 \\
\partial c_1 / \partial a_2=b_3 \\
\partial c_1 / \partial a_3=0 \\
\partial c_1 / \partial a_4=0 \\
\partial c_2 / \partial a_1=b_2 \\
\vdots \\
\partial c_4 / \partial a_4=b_4
\end{array}\right.
$$
其实相当于扩展了Jacobian的定义，即$C$中每一个元素，对于$A$中每一个元素进行求导。转化成标量的形式就好理解了吧~至于把以上16个标量求导写成$4 \times 4$的矩阵也好还是16维的向量也好，大多是为了形式（理论）上的美观，或是方便对求导结果的后续使用，亦或是方便编程实现，**按需自取**，其本质不变。

在神经网络里，所谓的矩阵形式都是通过标量进行形式化包装的，实际上的求导规则还是要和标量导数进行对应，因为前向传播中不同样本之间是互不影响的（暂时不考虑batch norm等），也就是只考虑affine function和element-wise的激活函数的情况（vector激活函数就单独算一下就行），所以对于上面的例子，数学上的答案应该是$B^\top \otimes I$，其中 $\otimes$ 是kronecker积，kronecker product的定义可以看这里 [here]( https://zhuanlan.zhihu.com/p/457055092) ，具体为啥是这结果可以看矩阵求导术。但是在神经网络里，我们的答案只是$B^\top$，因为本质上我们还是在处理一个多元函数的优化，矩阵的形式只是组织结果的一种方式，我们把多个训练样本组成一批形成矩阵，实际上和单个样本向量的处理差别只在于需要对不同batch样本得到的梯度结果求和，下面举个例子（沿用b乎大佬的[例子](https://zhuanlan.zhihu.com/p/37916911)）：

假设batch size为3，可以得到
$$
Y_{2 \times 3}=W_{2 \times 3} \cdot X_{3 \times 3}+B_{2 \times 1}
$$
对于其中的某个p sample而言
$$
\begin{aligned}
& y_{1p}=w_{11} x_{1p}+w_{12} x_{2p}+w_{13} x_{3p}+b_1 \\
& y_{2p}=w_{21} x_{1p}+w_{22} x_{2p}+w_{23} x_{3p}+b_2
\end{aligned}
$$
从而
$$
\frac{\partial C}{\partial w_{i j}}=\sum_{p} \frac{\partial C}{\partial y_{ip}} \frac{\partial y_{ip}}{\partial w_{i j}}=\sum_{p} x_{jp} \frac{\partial C}{\partial y_{ip}}
$$
这里的求和其实就是关键所在，我们是利用batch计算出的梯度进行更新，所以可以得到
$$
\begin{aligned}
\frac{\partial C}{\partial W} & =\left(\begin{array}{lll}
\sum_p x_{1p} \frac{\partial C}{\partial y_{1p}} & \sum_p x_{2p} \frac{\partial C}{\partial y_{1p}} & \sum_p x_{3p} \frac{\partial C}{\partial y_{1p}} \\
\sum_p x_{1p} \frac{\partial C}{\partial y_{2p}} & \sum_p x_{2p} \frac{\partial C}{\partial y_{2p}} & \sum_p x_{3p} \frac{\partial C}{\partial y_{2p}}
\end{array}\right) \\
& =\left(\begin{array}{c}
\frac{\partial C}{\partial y_{11}} & \frac{\partial C}{\partial y_{12}} & \frac{\partial C}{\partial y_{13}}\\
\frac{\partial C}{\partial y_{21}} & \frac{\partial C}{\partial y_{22}} & \frac{\partial C}{\partial y_{23}}
\end{array}\right) \cdot\left(\begin{array}{lll}
x_{11} & x_{21}  & x_{31} \\
x_{12} & x_{22} & x_{32} \\
x_{13} & x_{23} & x_{33} 
\end{array}\right) \\
& =\frac{\partial C}{\partial Y} \cdot X^T
\end{aligned}
$$
下面求 $\frac{\partial C}{\partial X}$ :
$$
\begin{aligned}
& \because \frac{\partial C}{\partial x_{jp} }=\frac{\partial C}{\partial y_{1p}} \frac{\partial y_{1p}}{\partial x_{jp}}+\frac{\partial C}{\partial y_{2p}} \frac{\partial y_{2p}}{\partial x_{jp}}=\frac{\partial C}{\partial y_{1p}} w_{1 j}+\frac{\partial C}{\partial y_{2p}} w_{2 j} \\
& \therefore \frac{\partial C}{\partial X}=\left(\begin{array}{l}
\frac{\partial C}{\partial x_{11}} & \frac{\partial C}{\partial x_{12}} & \frac{\partial C}{\partial x_{31}}\\
\frac{\partial C}{\partial x_{21}} & \frac{\partial C}{\partial x_{22}} & \frac{\partial C}{\partial x_{32}}\\
\frac{\partial C}{\partial x_{31}} & \frac{\partial C}{\partial x_{32}} & \frac{\partial C}{\partial x_{33}}
\end{array}\right) \\

& =\left(\begin{array}{ll}
w_{11} & w_{21} \\
w_{12} & w_{22} \\
w_{13} & w_{23}
\end{array}\right) \cdot\left(\begin{array}{c}
\frac{\partial C}{\partial y_{11}} & \frac{\partial C}{\partial y_{12}} & \frac{\partial C}{\partial y_{13}}\\
\frac{\partial C}{\partial y_{21}} & \frac{\partial C}{\partial y_{22}} & \frac{\partial C}{\partial y_{23}}
\end{array}\right) \\
& =W^T \cdot \frac{\partial C}{\partial Y} \\
&
\end{aligned}
$$
lab中的例子是：
$$
Z=A \cdot W^T+\iota \cdot b^T \quad \in \mathbb{R}^{N \times C_1}
$$
对于作业上的求导结果是这样的，可能唯一还需要想想的就是第一个式子，按道理不应该第一项的$\frac{\partial L}{\partial Z}$应该加上转置么（因为是分母布局），实际上因为链式法则也是有方向性，分母布局都是乘在左侧的（详细的解释看上面的例子和下面的总结），我们需要先把$Z$转置，得到$WA^T$的形式，再去对$A^T$求导，然后再把答案转置过来就能得到结果了。
$$
\begin{aligned}
& \frac{\partial L}{\partial A}=\left(\frac{\partial L}{\partial Z}\right) \cdot\left(\frac{\partial Z}{\partial A}\right)^T \quad \in \mathbb{R}^{N \times C_0} \\
& \frac{\partial L}{\partial W}=\left(\frac{\partial L}{\partial Z}\right)^T \cdot\left(\frac{\partial Z}{\partial W}\right) \quad \in \mathbb{R}^{C_1 \times C_0} \\
& \frac{\partial L}{\partial b}=\left(\frac{\partial L}{\partial Z}\right)^T \cdot\left(\frac{\partial Z}{\partial b}\right) \quad \in \mathbb{R}^{C_1 \times 1} \\
&
\end{aligned}
$$


For any linear equation of the kind $Z = AX + c$, the derivative of $Z$ with respect to $A$ is $X$. The derivative of $Z$ with respect to $X$ is $A^T$ . Also the derivative with respect to a transpose is the transpose of the derivative, so the derivative of $Z$ with respect to $X$ is $A^T$ but the derivative of $Z$ with respect to $X^T$ is $A$.

总结下就是:
$$
z=f(Y), Y=A X+B \rightarrow \frac{\partial z}{\partial X}=A^T \frac{\partial z}{\partial Y}
$$
这结论在 $\mathbf{x}$ 是一个向量的时候也成立, 即:
$$
z=f(\mathbf{y}), \mathbf{y}=A \mathbf{x}+\mathbf{b} \rightarrow \frac{\partial z}{\partial \mathbf{x}}=A^T \frac{\partial z}{\partial \mathbf{y}}
$$
如果要求导的自变量在左边, 线性变换在右边, 也有类似稍有不同的结论如下, 证明方法是类似的, 这里直接给出结论:
$$
z=f(Y), Y=X A+B \rightarrow \frac{\partial z}{\partial X}=\frac{\partial z}{\partial Y} A^T
$$
$$
z=f(\mathbf{y}), \mathbf{y}=X \mathbf{a}+\mathbf{b} \rightarrow \frac{\partial z}{\partial \mathbf{X}}=\frac{\partial z}{\partial \mathbf{y}} a^T
$$

最后还是提一嘴，别犯迷糊了，算出来的梯度是用来更新原参数的，也就是相当于$\Delta x$的感觉，因为是多元函数，所以最后对目标函数的增益应该是$\grad grad \cdot \Delta x$，之前太久没看这部分晕了一次。

> 参考资料：https://github.com/FelixFu520/README/blob/main/train/optim/matrix_bp.md

### Optimization

> 用英文写好累哇，开摆了，以后中英混着写，可能只有自己看得懂吧🤣，hh还是要保证通俗易懂

#### Material

参考资料：

http://deeplearning.cs.cmu.edu/F22/document/slides/lec6.optimization.pdf

http://deeplearning.cs.cmu.edu/F22/document/slides/lec7.stochastic_gradient.pdf

文章框架按照下面两个blog叙述，具体的细节和组织方式会丰富很多：

https://blog.paperspace.com/intro-to-optimization-in-deep-learning-gradient-descent/

https://blog.paperspace.com/intro-to-optimization-momentum-rmsprop-adam/

#### Basic conceptions

首先还是来介绍一些基础的概念和idea，会从最简单的梯度、海塞矩阵、泰勒展开谈起，我会分享我的一些思考方式。然后会介绍二次优化的方法，再过渡到神经网络的优化方法。

梯度能反映增长最快的方向，这个可以从hyperplane的角度去理解，可以参考b乎的这个回答 [here](https://www.zhihu.com/question/36301367/answer/198887937)。二阶导除了可以理解为导数的增长率，还可以从原函数局部均值的角度理解，也就是**Laplace** 算子的角度，可以参考这个视频 [here](https://www.youtube.com/watch?v=JQSC0lCPG24&list=PLSQl0a2vh4HC5feHa6Rc5c0wbRTx56nF7&index=68)。多元函数的泰勒展开可以写作矩阵乘法的形式其实也是从scale推的，二阶的展开推导可以看这里 [here](https://zhuanlan.zhihu.com/p/33316479)。

接下来就是和凸优化有关的理论了，对最简单的二次优化，其实牛顿法就可以给出每次更新的最优方向和步长，其实也就是泰勒二阶展开后，计算得到的最优值，一维scale的情况时，这个最优的学习率其实就是二阶导的倒数，而对于多元变量，就变成了海塞矩阵的逆了。

有关Lipschitz平滑、强对偶的理论直观上的感受可以看这里[here](https://zhuanlan.zhihu.com/p/27554191)，和梯度下降收敛性证明的有关理论可以看这个 [here](https://cs.mcgill.ca/~wlh/comp451/files/comp451_chap8.pdf)，还有这里 [here](https://www.cs.ubc.ca/~schmidtm/SVAN16/L4.pdf) (加拿大的学校的教学资料是真好啊)，详细的我就不介绍了。大概就是说强突的收敛的会更快，如果只是Lipschitz平滑也能收敛。

<img src=".\image\HW1P1_image32.png" alt="image-20230107230430281" style="zoom:80%;" />

<img src=".\image\HW1P1_image33.png" alt="image-20230107230509280" style="zoom:80%;" />

现在来讲讲海塞矩阵特征值和收敛的关系，我们知道通过二阶展开可以得到局部的近似，这个近似在局部最值附近的等值线体现为一个椭圆，而这个椭圆的长短轴就是和海塞矩阵的特征值成比例，原因这样的，通过泰勒展开我们可以把函数近似写为二次型的表达式，对应的等值线就是二次曲线这种，长短轴和特征值的关系就很明显了，通过SVD的几何意义也很容易得到。如果海塞矩阵的条件数很大，也就是椭圆的长短轴相差很大，说明这个损失函数平面是病态的，就比较难收敛。



<img src=".\image\HW1P1_image34.png" alt="image-20230107222504865" style="zoom:80%;" />

<img src=".\image\HW1P1_image35.png" alt="image-20230107222557585" style="zoom:80%;" />



但是海塞矩阵的逆计算代价大概是$O(N^3)$的，放到神经网络里面肯定是嫩算算不出来的，也就是说对参数的每个分量确定最优的学习率是很难的，我们考虑两种不同的做法，第一种还是采取每个分量独立的学习率，但是使用启发式的思路而不是嫩算（Rprop算法），第二种就是所有分量一个学习率，就是常见的学习率直接乘梯度向量，但是这样的话收敛性就不一定能保证了，以二次优化为例，如果学习率大于任一分量最优学习率的二倍，就会发散，也由此提出了学习率衰减和收敛的理论，Robbins-Munroe conditions算是最经典的收敛条件，也就是SGD对于满足凸性和平滑的模型，只要学习率满足下面的条件就会收敛
$$
\sum_{k=0}^{\infty} \alpha^{(k)}=\infty \quad \sum_{k=0}^{\infty}\left(\alpha^{(k)}\right)^2<\infty
$$
此外在梯度下降的过程中还存在主要的两个问题需要解决，针对这两个问题提出的各种解决方法就是这节的重点学习内容了。第一个问题是达不到全局最值，会收敛到局部最优点，第二个问题是对于病态的损失函数平面很可能会出现震荡的现象，导致收敛慢效果很差。



#### SGD and Batch

直接用全样本去计算loss function会有至少两个问题，第一个是cycle的表现，第二个是不能避免局部最值。针对这个问题，可以利用随机采样计算梯度，最简单的策略就是随机取一个样本，这是SGD的思路，但是这样会使收敛过程的方差变得很大，而且损失函数也不会变得足够小。

<img src=".\image\HW1P1_image36.png" alt="image-20230108115028332" style="zoom:80%;" />



<img src=".\image\HW1P1_image37.png" alt="image-20230108115100729" style="zoom:80%;" />

针对这个问题也是提出了mini-batch去解决，同样作为无偏估计，但是其方差缩小为原本的${1\over batch}$倍，收敛的效果也是得到了提升：

<img src=".\image\HW1P1_image38.png" alt="image-20230108115150825" style="zoom:80%;" />

<img src=".\image\HW1P1_image39.png" alt="image-20230108115259488" style="zoom:80%;" />

<img src=".\image\HW1P1_image40.png" alt="image-20230108115336885" style="zoom:80%;" />

#### Learning Rate and Grad direction

针对病态的震荡情况，因为我们只能利用一阶导的信息，如果可以利用二阶导其实可以获得曲率等信息进行避免，但是由于计算量我们还是想一些启发式的方法去解决，未来的改进方向肯定就是何如兼顾二阶信息又减少运算量。

这部分说实话资料都很全了，值得提一嘴的是基于学习率改进的算法，[AdaGrad](https://zhuanlan.zhihu.com/p/29920135) 和 RMS Prop的关系，RMS Prop通过移动平均解决了Adagrad中平方和累加过大缺乏正则化的问题。

### Normalization

<img src=".\image\HW1P1_image41.png" alt="image-20230108145934082" style="zoom:80%;" />

<img src=".\image\HW1P1_image42.png" alt="image-20230108150349572" style="zoom:80%;" />

值得注意的是，使用batch-norm的话 ，线性层就不需要额外的bias项了，会被归一化掉，算是个Arbitrary的选择，可加可不加。

<img src=".\image\HW1P1_image43.png" alt="image-20230108150531429" style="zoom:80%;" />





## Python Implementation

这部分直接看我仓库吧，就不放到这里写了：[code](https://github.com/XuRui314/CMU-11-785)



## Torch Pipeline

&ensp;Colloquially,training a model can be described like this:

1. We get data-pairs of questions and answers.

2. For a pair `(x, y)`, we run `x` through the model to get the model's answer `y`. 
3. Then, a "teacher" gives the model a grade depending on “how wrong” `y`  is compared to the true answer `y`.
4. Then based on the grade,we figure out who's fault the error is.
5. Then, we fix the faults so the model can do better next time.

&ensp;To train a model using Pytorch, in general, there are 5 main parts:

1. Data
2. Model
3. Loss Function
4. Backpropagation
5. Optimizer

### Data

&ensp;When training a model, data is generally a long list of `(x, y)` pairs, where you want the model to see `x` and predict `y`.

&ensp;Pytorch has two classes you will need to use to deal with data:

- `torch.utils.data.Dataset` 
- `torch.utils.data.Dataloader`

&ensp;Dataset class is used to preprocess data and load single pairs `(x, y)`

&ensp;Dataloader class uses your Dataset class to get single pairs and group them into batches

<img src=".\image\HW1P1_image7.png" alt="image-20221215214237507" style="zoom:80%;" />



&ensp;When defining a Dataset, there are three class methods that you need to implement 3 methods: `__init__`, `__len__`, `__getitem__`.

&ensp;Use `__init__` to load the data to the class so it can be accessed later, Pytorch will use `__len__` to know how many `(x, y)` pairs (training samples) are in your dataset. After using `__len__` to figure out how many samples there are, Pytorch will use `__getitem__` to ask for() a certain sample. So `__getitem__(i)` should return the "i-th" sample, with order chosen by you. You should use `getitem` to do some final processing on the data before it’s sent out. Since `__getitem__` will be called maybe millions of times, so make sure you do as little work in here as possible for fast code. Try to keep heavy preprocessing in `__init__`, which is only called once

&ensp;Here is a simple Dataset example:

```python
class MyDataset(data.Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
      
    def __len__(self.Y):
        return len(self.Y)
    
    def __getitem(self, index):
        X = self.X[index].float().reshape(-1) #flatten the input
        Y = self.Y[index].long()
        return X, Y
```

&ensp;Here is a simple Dataloader example:

```python
# Training
train_dataset = MyDataset(train.train_data, train.train_labels)

train_loader_args = dict(shuffle = True, batch_size = 256, num_workers = num_workers, pin_memory = True)\
if cuda else dict(shuffle = True, batch_size = 64)

train_loader = data.DataLoader(train_dataset, **train_loader_args)
```

### Model

&ensp;This section will be in two parts:

• How to generate the model you’ll use

• How to run the data sample through the model.

<img src=".\image\HW1P1_image8.png" alt="image-20221215224941804" style="zoom:80%;" />

&ensp;One key point in neural network is modularity, this means when coding a network, we can break down the structure into small parts and take it step by step.

&ensp;Now, let’s get into coding a model in Pytorch. Networks in Pytorch are (generally) classes that are based off of the `nn.Module class`. Similar to the Dataset class, Pytorch wants you to implement the `__init__ ` and `forward` methods.

• `__init__`: this is where you define the actual model itself (along with other 

stuff you might need)

• `Forward`: given an input `x`, you run it through the model defined in `__init__`

```python
class Our_Model(nn.Module):
	def __init__(self):
        super().__init__()
        
        self.layer1 = nn.Linear(3, 4)
        self.layer2 = nn.Linear(4, 4)
        self.layer3 = nn,Linear(4, 1)
      
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
      
    	return out
```

&ensp;However, it can get annoying to type each of the layers twice – once in `__init__ `and once in forward. Since on the right, we take the output of each layer and directly put it into the next, we can use the **nn.Sequential** module.

```python
class Our_Model(nn.Module):
    def __init__(self):
        layers = {
            nn.Linear(3, 4),
            nn.Linear(4, 4),
            nn.Linear(4, 1)
        }
        self.layers = nn.Sequential(*layers) # * operator opens up the lsit and directly puts them in as arguments of nn.Sequential
        
	def forward(self, x):
        return self.layers(out)
```

&ensp;As a beginner to Pytorch, you should definitely have [link](https://pytorch.org/docs/stable/nn.html) open. The documentation is very thorough. Also, for optimizers: [link](https://pytorch.org/docs/stable/optim.html).

&ensp;Now that we have our model generated, how do we use it? First, we want to put the model on GPU. Note that for `nn.Module` classes, `.to(device)` is in-place. However, for tensors, you must do `x = x.to(device)`

```python
device = torch.device("cuda" if cuda else "cpu")
model.to(device)
```

&ensp;Also, models have `.train()` and `.eval()` methods. Before training, you should run `model.train()` to tell the model to save gradients. When validating or testing, run `model.eval()` to tell the model it doesn’t need to save gradients (save memory and time). A common mistake is to forget to toggle back to .train(), then your model doesn’t learn anything.

&ensp;So far, we can build:

```python
# Dataset Stuff
train_dataset = MyDataset(train.train_data, train.train_labels)

train_loader_args = dict(shuffle = True, batch_size = 256, num_workers = num_workers, pin_memory = True)\
if cuda else dict(shuffle = True, batch_size = 64)

train_loader = data.DataLoader(train_dataset, **train_loader_args)

# Model Stuff
model = nn.Sequential(nn.Linear(3,4 ), nn.Linear(4, 4), nn.Linear(4, 1))
device = torch.device["cuda" if cuda else "cpu"]
model.to(device)

# Optimization Stuff
NUM_EPOCHS = 100
# ----------------- #
# |	Not	cover yet | # initialize the criterion
# ----------------- #

# Training
for epoch in range(NUM_EPOCHS):
    model.train()
    
    for(x, y) in train_loaders:
        # ----------------- #
		# |	Not	cover yet | # optimizer initialize
		# ----------------- #
        x.to(device)
        y.to(device)
        
        output = model(x)
        # ----------------- #
		# |	Not	cover yet | # calculate loss, minimize this loss and backpropagate
		# ----------------- #
```

### Loss Function

To recap, we have run x through our model and gotten “output,” or `y`. Recall we need something to tell us how wrong it is compared to the true answer `y`. We rely on a “loss function,” also called a “criterion” to tell us this. The choice of a criterion will depend on the model/application/task,  but for classification, a criterion called “CrossEntropyLoss” is commonly used.

```python
# Optimization Stuff
NUM_EPOCHS = 100
criterion = nn.CrossEntropyLoss()
# ----------------- #
# |	Not	cover yet | #
# ----------------- #

#Training
for epoch in range(NUM_EPOCHS):
    model.train()
    
    for(x, y) in train_loaders:
        # ----------------- #
		# |	Not	cover yet | #
		# ----------------- #
        x.to(device)
        y.to(device)
        
        output = model(x)
        loss = criterion(output, y)
        
        # ----------------- #
		# |	Not	cover yet | #
		# ----------------- #
```





### Backpropagation

Backpropagation is the process of working backwards from the loss and calculating the gradients of every single (trainable) parameter w.r.t the loss. The gradients tell us the direction in which to move to minimize the loss.

```python
#Training
for epoch in range(NUM_EPOCHS):
    model.train()
    
    for(x, y) in train_loaders:
        # ----------------- #
		# |	Not	cover yet | #
		# ----------------- #
        x.to(device)
        y.to(device)
        
        output = model(x)
        loss = criterion(output, y)
        
        loss.backward() # add here
        # ----------------- #
		# |	Not	cover yet | #
		# ----------------- #
```

By doing `loss.backward()`, we get gradients w.r.t the loss. Remember model.train()? That allowed us to compute the gradients. If it had been in the eval state, we wouldn’t be able to even compute the gradients, much less train.



### Optimizer

Now, backprop only *computes* the $∇p$ values – it doesn’t do anything with them. We want to *update* the value of $p$ using $∇p$. This is the optimizer’s job.

A crucial component of any optimizer is the “learning rate.” This is a hyperparameter that controls how much we should believe in $∇p$.  Again, this will be covered in more detail in a future lecture. Ideally, $∇p$ is a perfect assignment of blame w.r.t the **entire** dataset. However, it’s likely that optimizing to perfectly match the *current* (x, y) sample $∇p$ was generated from won’t be great for matching the entire dataset.

Among other concerns, the optimizer *weights* the $∇p$ with the learning rate and use the weighted ∇p to update $p$. 



```python
# Optimization Stuff
NUM_EPOCHS = 100
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 1e-4) # add here
#Training
for epoch in range(NUM_EPOCHS):
    model.train()
    
    for(x, y) in train_loaders:
        optimizer.zero_grad()
        x.to(device)
        y.to(device)
        
        output = model(x)
        loss = criterion(output, y)
        
        loss.backward() # add here
        optimizer.step()
```

What is zero_grad? Every call to .backward() saves gradients for each parameter in the model. However, calling `optimizer.step()` **does not** delete these gradients after using them. So, you want to remove them so they don’t interfere with the gradients of the next sample.

By doing `optimizer.step()`, we update the weights of the model using the computed gradients.

After here, you would generally perform validation (after every epoch or a couple), to see how your model performs on data it is not trained on. Validation follows a similar format as training, but without `loss.backward()` or `optimizer.step()`. You should  check the notebooks for more guidance.

> The complete code: [link](https://colab.research.google.com/drive/1huAQcxM9jMqSNb4h6XJ78Xd8EM1-UF_x#scrollTo=Sg8IUZ1er0dl)



## End

这个Chapter就当成闲话区吧，看瑞克莫蒂的时候突然想到了薛定谔的猫（这个实验想说明的点其实是反驳哥本哈根派，我只是突然想到了hh，和它的本意不太一样），然后联想到之前学概率论想到的问题，现实中很多时候都是事情的结果已经确定下来，而我们不知道，只能用概率去建模，无论怎么建模怎么说都挺reasonable的，于其说是预知未知的事件，不如说是在存在信息差的情况下进行分析。所以才需要贝叶斯这种不断更新获取后验的方法，以及各种消除information gap的策略吧，或者有没有什么根据因果溯源一样的建模手段，能更加直接的对概率进行更新，想想可能在现实中也是intractable的。







