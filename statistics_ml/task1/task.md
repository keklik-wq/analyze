Gaussian linear regression

Problem
Load data set and perform the following analysis
1. Fit a linear model for the continuous variable.
2. Calculate R2 and interpret the coefficients.
3. Add interaction on categorical variable for all variables (if it is impossible, do it only for
some variables).
4. Compare results by R2 on train set and R2 on randomly chosen test set.

Recommendations
Least squares fit is given by lm function. A lot of information is given by summary function
for lm object. It is possible to add interaction of variables x1 and x2 by adding x1*x2 term to
formula argument of lm function.