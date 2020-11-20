# End to end project

**Look at the big picture**

* Frame the problem

  * What is the business objective ? 

  * What does the current solution looks like ? 

* Select a performance measure

  * Distance measures
    
    The $l_k$ norm of a vector v containing n elements is defined as $\vert \vert v \vert \vert_k = (\vert v_0\vert^k + ... + \vert v_n\vert^k)^{\frac{1}{k}}$. $l_0$ just gives the number of non-zero elements in the vector, and $l_{\infty}$ gives the maximum absolute value in the vector. 

    The higher the norm index, the more it focuses on large values and neglects small ones. That is why the RMSE is more sensitive to outliers than MAE. When outliers are exponentially rare (like in a bell-shape curve), the RMSE performs very well and is generally preferred. 

  * Notes from [Comparison between MAE and RMSE](https://medium.com/human-in-a-machine-world/mae-and-rmse-which-metric-is-better-e60ac3bde13d)
    
    * Similarities

      Both MAE and RMSE express aveage model prediction error in units of the variable of interest. Both range from 0 to $\infty$ and are indifferent to the direction of errors. They are negatively-oriented scores, the lower the better.

    * Differences 
     
      Since errors are squred before they are averaged, the RMSE gives a relatively high weight to large errors. This means the RMSE should be more useful when large errors are particularly undesirable. RMSE does not necessarily increase with the variance of teh errors. RMSE increases with the variance of the frequency distribution of error magnitudes. 

      $$
      [MAE] \leq [RMSE]
      $$

      The RMSE result will always be larger or equal to the MAE. If all the errors have the same magnitude, then $RMSE = MAE$.

      $$
      [RMSE] \leq [MAE * sqrt(n)]
      $$
      where $n$ is the number of test samples. The difference between RMSE and AME is greatest when all of the prediction error comes form a single test sample. 

      RMSE has a tendency to be increasingly larger than AME as the test sample size increases. And this can be problematic when comparing RMSE results calcualted on different sized test samples. 

      RMSE has the benefit of penalizing large errors more. From an interpretation standpoint, MAE is the winner. On the other hand, RMSE is desirable in many mathematical calcualtion. 

* Check the assumptions

  

**Get the data**

* Create the workspace
 
  


**Discover and visualize the data to gain insights**

**Prepare the data for ml algo**

**Select a model and train it**

**Fine-tune the model**

**Present your solution**

**Launch, monitor, and maintain your system**