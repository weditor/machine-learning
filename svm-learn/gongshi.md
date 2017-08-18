

对于一个分类面f(x)
$$f(x) = w^{T} \cdot x + b$$

假设所有样本距离此分类面的最小距离为$\delta$,并且正反的分类分别为1和-1，则对于正样本和负样本有如下不等式。
$$\begin{cases}
 w^{T} \cdot x_i + b \ge \delta  &, (y_i=1)\\
 w^{T} \cdot x_i + b \le -\delta &, (y_i=-1)
\end{cases}$$

两端乘以$y$
$$ F(x_i, y_i) = (w^{T} \cdot x_i+b)*y_i \ge \delta $$

$$ \Rightarrow min(F(x, y) ) = \delta$$

所以转换为优化$\delta$,使其最大化。($max(\delta)$)
但是函数距离并不能代表一个点到一个分类面的真实距离,举例如下：
对于一个二维分类面 
$$f(x) = (1, 2) \cdot \begin{pmatrix}x_1\\x_2\end{pmatrix} + 0 = 0$$
可以表示成以下两种等价形式:
$$x+2y = 0$$
$$3x+6y = 0$$
考虑一个点(0, 1), 分别求到分类面的函数距离:
$$(0, 1) \Rightarrow f_1(x, y) = 0 + 2*1 = 2$$
$$(0, 1) \Rightarrow f_2(x, y) = 3*0 + 6*1 = 6$$
所以函数距离与w是相关的，每一个不同的w法向量对应于不同的距离。因此采用其几何距离，即函数距离除以w的模。而几何距离是不变的。
$$ \frac {F(x, y)}{\begin{Vmatrix}w\end{Vmatrix}} $$

因此最终最终转换为优化为下列式子：
$$ \Rightarrow min\left(\frac {F(x, y)}{\begin{Vmatrix}w\end{Vmatrix}} \right) = \frac {\delta}{\begin{Vmatrix}w\end{Vmatrix}}$$

在所有分类面中找出$\frac {\delta}{\begin{Vmatrix}w\end{Vmatrix}}$为最大值的分类面,由于$\delta$可以任意变化，并且确定$\delta$后，w也确定了，为了求解方便，就强制将$\delta$归一化为1。此时简化成了优化问题:
$$max\left(\frac 1{\begin{Vmatrix}w\end{Vmatrix}}\right) ,F(x_i, y_i) \ge 1$$

上述优化问题与以下优化问题等价:
$$min\left({\begin{Vmatrix}w\end{Vmatrix}}^2\right) ,F(x_i, y_i) \ge 1$$

$F(x_i, y_i) \ge 1$展开如下:
$$(w^{T} \cdot x_i+b)*y_i \ge 1$$

使用拉格朗日乘子法列出方程:

$$\Phi(\alpha, w, b) = \begin{Vmatrix}w\end{Vmatrix}^2 - \sum_{i=0}^n{\left\{\alpha_i(w^{T} \cdot x_i+b)*y_i\right\}}$$

求解:
$$\begin{cases}
\frac {\partial \Phi}{\partial w} = 0 &\Rightarrow w=\sum_{i=0}^n\alpha_i y_i x_i\\
\\
\frac {\partial \Phi}{\partial b} = 0 &\Rightarrow \sum_{i=0}^na_iy_i=0
\end{cases}$$

然而这组方程并不像表面上那么好解决。。。
需要使用SMO算法迭代出最优解。
最终可以迭代出w和b，即求出了分类面$f(x)=w^T \cdot x + b$
而SVM的训练过程即SMO算法迭代求解$w$和$b$的过程。

| ->$\alpha$ | $text_{ text_\alpha}$ | gerg|
|--------|--------|
|fuck| argwae|


