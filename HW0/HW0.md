In this note(blog) , i will cover some basic ideas and knowledge you are supposed to hold in your mind. This can be quiet useful for cmu's deep-learning course.

So the layout is as the following:

1. Pytorch tutorial 
2. Mathematics tutorial



## Pytorch







## Math

### Matrix derivatives



Jacobian matrix explanation (2 aspect: derivative and integral):

I will introduce he main idea of vector function derivative, watch the video to get the idea of integral  :)

> https://www.youtube.com/watch?v=wCZ1VEmVjVo



Suppose we have $f(x, y) = (x^2-y^2, 3xy)$, just a vector function, and we want to calculate its derivative. One reasonable idea is to calculate every element in $f(x, y)$ respectively and concatenate them to get a vector. This is feasible, but still remaining a sense of just a little bit to go because we just separately viewing $f(x,y)$ as 2 multivariate functions and get the gradient vector, composing them up to get derivative matrix (in a specified way for matrix multiplication, without other meaning),  which looses  wholeness and overall perspective. We just get the value, haven't captured the relations and structure.

So we need to think from the perspective of the whole picture, see the picture bellow to get a feeling of regarding $f$ as nonlinear transformation.

<img src=".\image\HW0_image1.png" alt="image-20221225104006269" style="zoom:80%;" />

Since $f$ is a continuous function, let's zoom in, and we can get the linear picture like this:

<img src=".\image\HW0_image2.png" alt="image-20221225112702996" style="zoom: 80%;" />

Now think of adding a perturbation on point $(a, b)$, so what will the output look like? Instead of imaging a hyperplane and getting 1 dim output like multivariate function, we now have 2 dim output. The intuitive idea is that a perturbation in the input space like $(a + \Delta x, b + \Delta y)$ will cause the output point move along the white and yellow line in the above picture. 

The white line corresponds to the first col of the matrix, determining how much changing input $x$ will effect the output along the white line (corresponding the $x$ axis of input space). The yellow line corresponds to the second col of the matrix, the explanation is similar.
$$
\left(\begin{array}{ll}
\left.\frac{\partial f_1}{\partial x}\right|_{(a, b)}  & \left.\frac{\partial f_1}{\partial y}\right|_{(a, b)} \\
\left.\frac{\partial f_2}{\partial x}\right|_{(a, b)} & \left.\frac{\partial f_2}{\partial y}\right|_{(a, b)}
\end{array}\right)
$$
In fact, this matrix is called as the Jacobian matrix. Jacobian matrix is the matrix representing best linear map approximation of $f$ near $(a, b)$.

Let's look at an example, suppose we have $y = Ax$ , what's the derivative of $y$ with respect to $x$? The ans is ${dAx \over dx} = A$, the reason is obvious, because now the Jacobian matrix degenerates into a constant matrix $A$.

> 这里的结果是$A$而不是$A^\top$的原因是采用了分子布局，在Jacobian有关的资料里是这样用的，但是机器学习里面多采用分母布局，也就是最后答案是$A^\top$，分子布局和分母布局的答案互为转置（如果有Chain rule的应用的话应该是每个项转置而不是连乘的转置，乘积顺序是不变的），原因是在分母布局中规定$d \boldsymbol{f}=\frac{\partial \boldsymbol{f}}{\partial \boldsymbol{x}}^T d \boldsymbol{x}$，也就是说梯度的转置乘$dx$才是$df$，而分子布局中则是$d \boldsymbol{f}=\frac{\partial \boldsymbol{f}}{\partial \boldsymbol{x}} d \boldsymbol{x}$。

分母布局的例子可以看下面：

For any linear equation of the kind $Z = AX + c$, the derivative of $Z$ with respect to $A$ is $X$. The derivative of $Z$ with respect to $X$ is $A^T$ . (We will explain the rationale behind this in class). Also the derivative with respect to a transpose is the transpose of the derivative, so the derivative of $Z$ with respect to $X$ is $A^T$ but the derivative of $Z$ with respect to $X^T$ is $A$.











> http://deeplearning.cs.cmu.edu/S22/document/homework/HW1/Derivatives.pdf

