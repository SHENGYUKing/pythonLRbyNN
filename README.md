# pythonLRbyNN
An Implementation of Logistics Regression in Neural Network way by Python

## Principle of Logistics Regression
### activation function
Logical regression usually uses sigmoid function as the activation unit:    
<img src="https://latex.codecogs.com/png.image?\dpi{120}&space;\bg_white&space;\mathit{sigmoid}(z)=\frac{1}{1&plus;\boldsymbol{e}^{-z}}" title="\bg_white \mathit{sigmoid}(z)=\frac{1}{1+\boldsymbol{e}^{-z}}" />    
After input data processed by the linear unit, we can use the above formula to compute the predictions.    
Linear unit just like the following formula:    
<img src="https://latex.codecogs.com/png.image?\dpi{120}&space;\bg_white&space;Z=\omega^{T}X&plus;b" title="\bg_white Z=\omega^{T}X+b" />    

### forward propagate
First, compute the activation of input data using the following formula:    
<img src="https://latex.codecogs.com/png.image?\dpi{120}&space;\bg_white&space;A=\sigma&space;(Z)=\mathit{sigmoid}(\omega^{T}X&plus;b)=(a^{(1)},a^{(2)},\cdots,a^{(m-1)},a^{(m)})" title="\bg_white A=\sigma (Z)=\mathit{sigmoid}(\omega^{T}X+b)=(a^{(1)},a^{(2)},\cdots,a^{(m-1)},a^{(m)})" />    
Then, calculate the cost function using cross-entropy formula:    
<img src="https://latex.codecogs.com/png.image?\dpi{120}&space;\bg_white&space;J=-\frac{1}{m}\sum_{i=1}^{m}(y^{(i)}\mathit{log}(a^{(i)})&plus;(1-y^{(i)})\mathit{log}(1-a^{(i)}))" title="\bg_white J=-\frac{1}{m}\sum_{i=1}^{m}(y^{(i)}\mathit{log}(a^{(i)})+(1-y^{(i)})\mathit{log}(1-a^{(i)}))" />    

### backward propagate    
In order to reduce the loss, we need to keep updating the parameters until the loss is minimal. So, we should calculate the partial derivatives of the cost function with respect to w and b at first.    
partial derivative of the cost function with respect to w:    
<img src="https://latex.codecogs.com/png.image?\dpi{120}&space;\bg_white&space;\frac{\partial&space;J}{\partial&space;w}=\frac{1}{m}X(A-Y)^T" title="\bg_white \frac{\partial J}{\partial w}=\frac{1}{m}X(A-Y)^T" />    
partial derivative of the cost function with respect to b:    
<img src="https://latex.codecogs.com/png.image?\dpi{120}&space;\bg_white&space;\frac{\partial&space;J}{\partial&space;b}=\frac{1}{m}\sum_{i=1}^m(a^{(i)}-y^{(i)})" title="\bg_white \frac{\partial J}{\partial b}=\frac{1}{m}\sum_{i=1}^m(a^{(i)}-y^{(i)})" />    

### update parameters
Now, we can use the above partial derivatives to update the parameters.    
update weights (w):    
<img src="https://latex.codecogs.com/png.image?\dpi{120}&space;\bg_white&space;w=w-\alpha\frac{\partial&space;J}{\partial&space;w}" title="\bg_white w=w-\alpha\frac{\partial J}{\partial w}" />    
update bias (b):    
<img src="https://latex.codecogs.com/png.image?\dpi{120}&space;\bg_white&space;b=b-\alpha\frac{\partial&space;J}{\partial&space;b}" title="\bg_white b=b-\alpha\frac{\partial J}{\partial b}" />    
The alpha in the above formulas means learning rate, and it is usually a very small number (about 0.01).    

### predict
At last, take the updated parameters into forward propagation to get the predictions.    
<img src="https://latex.codecogs.com/png.image?\dpi{120}&space;\bg_white&space;\hat{Y}=\sigma(w^T&space;X&plus;b)" title="\bg_white \hat{Y}=\sigma(w^T X+b)" />    
In fact that the result y_hat means the probability, and we can design reasonable thresholds for getting the exact conclusion according to reality situation.    

## custom function summary
- generate_dataset(mode, n_samples, noise):    
  generate a virtual dataset to train and test our algorithm    
- sigmoid(z):    
  compute the sigmoid function of input z    
- init_wb(dim):    
  initialize parameters of logistics regression by np.random.randn    
- propagate(w, b, X, y):    
  implement the cost function and its gradient for the propagation, including FP and BP    
- optimize(w, b, X, y, num_iterations, learning_rate, show):    
  implement the optimization of the parameters w (weights) and b (bias) by gradient descent algorithm    
- predict(w, b, X):    
  implement the prediction of input examples using learned logistics regression model    
- model(X_train, y_train, X_test, y_test, num_iterations, learning_rate, show):    
  build the logistics regression model using the functions we defined above    
- plot_cost(costs, learning_rate):    
  plot the learning curve based on the costs in the whole training process    
- main():    
  main function to run    
