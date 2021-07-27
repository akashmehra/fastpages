## Optimization, Parameter Updates and Hyper-parameters

### Gradient based optimization

Gradient based optimization is a technique to minimize/maximize the function by updating the paremeters/weights of a model using the gradients of the Loss wrt to the parameters. If the Loss function is denoted by $\mathcal{L}(w)$​​​, where $w$​​​ are the parameters then we'd like to calculate $\nabla_{w} \mathcal{L}(w)$​​​ and to get the parameters for the next iteration, we'd like to perform the  $w_{t+1} = w_{t} - \alpha * \nabla_{w} \mathcal{L}(w_{t})$, where  $\alpha$​​​​ is the learning rate. 

> In order to update the parameters of a Machine Learning model we need a way for us to measure the rate of change in the output when the inputs (the parameters) are changed.

Here, we've assumed that the function $\mathcal{L}(w)$​​ is continuous. The function $\mathcal{L}(w)$​ can have kinks in which case we'll call the *gradient* a *subgradient*. *Subgradient* generalizes the notion of a derivative to functions that are not necessirily differentiable. More on *subgradients* will follow in a separate post, but for now assume that there exists a concept using which one can calculate the *gradient* of a function that is not differentiable everywhere (has kinks, for example: `ReLU` non-linearity).

There are many gradient based optimization algorithms that exist and they differ mainly in how the gradients are calculated or how the learning rate $\alpha$​ is chosen. We'll look at some of those algorithms that are used in practice. 

### Optimization basics

#### Univariate Optimality Conditions: Quick recap

Consider a function $f(x)$​, where $x$​ is univariate, the necessary conditions for a point $x=x_0$​ to be a minimum  of $f(x)$​  with respect to its *infinitesimal locality* are: $f'(x_0) = 0$​ and $f''(x_0) > 0$​. The optimality conditions can be well undertsood if we look at the *Taylor Series* expansion of $f(x)$​ in the small vicinty of $x_0 + \Delta$​.
$$
f(x_0 + \Delta) \approx f(x_0) + \Delta f'(x) + \frac{\Delta^2}{2} f''(x)
$$
Here, the value of $\Delta$​ is assumed to be very small. One can see that if $f'(x) = 0$​ and $f''(x) > 0$​ then $f(x_0 + \Delta) \approx f(x_0) + \epsilon$​​, which means that $f(x_0) < f(x_0 + \Delta)$​, for small values of $\Delta$​ (whether it is positive or negative) or $x_0$​ is a minimum wrt to its immediate locality.

#### Multivariate Optimality Conditions

Consider a function $f(x)$ where $x$ is an n-dimensional vector given by $\begin {bmatrix}x_1, x_2, x_3, \cdots, x_n \end {bmatrix}^T$. The *gradient vector* of $f(x)$ is 

given by the partial derivatives wrt to each of the components of $x$, $\nabla {f{x}} \equiv g(x) \equiv \begin {bmatrix} 
\frac{\partial f}{\partial x_1}\\
\frac{\partial f}{\partial x_2}\\
\frac{\partial f}{\partial x_3}\\
\vdots \\
\frac{\partial f}{\partial x_n}\\
\end{bmatrix}$ 

Note that the *gradient* in case of a *multivariate* functions is vector of *n-dimensions*. Similarly, one can define the *second derivative* of a *multivariate* function using a matrix of size $n \cross n$
$$
\nabla^2f(x)\equiv H(x) \equiv \begin{bmatrix}
\frac{\partial^2 f}{\partial x_{1}^2} &  \cdots & \frac{\partial^2 f}{\partial x_{1}\partial x_{n}}\\
\vdots & & \vdots \\
\frac{\partial^2 f}{\partial x_{n}\partial x_{1}} & \cdots &  \frac{\partial^2 f}{\partial x_{n}^2}\\

\end{bmatrix}
$$
Here, $H(x)$​​​​ is called a *Hessian Matrix*, and if the partial derivatives ${\partial^2 f}/{\partial x_{i}\partial x_{j}}$​​​​​ and ${\partial^2 f}/{\partial x_{j} \partial x_{i}}$​​​​​  are both defined and continuous and then by *Clairaut's Theorem* $\partial^2 f/\partial x_{i}\partial x_{j}$​​​​​​​ = $\partial^2 f/\partial x_{j}\partial x_{i}$​​​​​,​​ this second order *partial derivative* matrix becomes symmetric. 

If $f$ is quadratic then the *Hessian* becomes a constant, the function can then be expressed as: $f(x) = \frac{1}{2}x^THx + g^Tx + \alpha$ , and as in case of *univariate* case the optimality conditions can be derived by looking at the *Taylor Series* expansion of $f$ about $x_0$: 
$$
f(x_0 + \epsilon\overline{v}) = f(x_0) + \epsilon \overline{v}^T \nabla f(x_0) + \frac {\epsilon^2}{2} \overline{v}^T H(x_0 + \epsilon \theta \overline{v}) \overline{v}
$$
where $0 \leq \theta \leq 1$​, $\epsilon$​ is a *scalar* and $\overline{v}$​ is an *n-dimensional* vector. Now, if $\nabla f(x_0) = 0$​, then it leaves us with $f(x_0) + \frac{\epsilon^2}{2} \overline{v}^T H \overline{v}$​, which implies that for the $x_0$​ to be a point of minima,  $\overline{v}^T H \overline{v}> 0$​​ or the *Hessian* has to be *positive definite*. 

> Quick note on definiteness of the symmetric Hessian Matrix

* $H$​ is *positive definite* if $\bold v^TH \bold v > 0$​, for all *non-zero vectors* $\bold v \in \mathbb{R}^n$​​ (all *eigenvalues* of $H$​ are *strictly positive*)
* $H$​ is *positive semi-definite* if $\bold v^TH \bold v \geq 0$​, for all *non-zero vectors* $\bold v \in \mathbb{R}^n$​ (*eigenvalues* of $H$​ are *positive* or *zero*)
* $H$​​ is *indefinite* if there exists a  $\bold v, \bold u \in \mathbb{R}^n$​, such that $\bold v^TH \bold v > 0$ and $\bold u^T H \bold u < 0$ (*eigenvalues* of $H$ have mixed sign)
* $H$​​ is *negative definite* if $\bold v^TH \bold v < 0$​​, for all *non-zero vectors* $\bold v \in \mathbb{R}^n$​​ (all *eigenvalues* of $H$​​ are *strictly negative*)

### Need for Gradient Descent

One of the necessary conditions for a point to be a critical point (minima, maxima or saddle) is that the first order derivate $f'(x) = 0$, it is often the case that we're not able to exactly solve this equation because the derivative can be a complex function of $x$. A *closed form solution* so to speak, doesn't exist and things get even more complicated in *multivariate* case due to compultational and numerical challenges<sup>[1]</sup>. We use *Gradient Descent* to iteratively solve the optimization problem irrespective of the functional form of $f(x$) by taking a step in the direction of the steepest descent (because in Machine Learning we're optimizing a *Loss* or a *Cost* function, we tend to always solve the optimization problem from the perspective of *minimization*).













