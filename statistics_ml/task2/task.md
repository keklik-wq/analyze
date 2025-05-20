Generalized linear models
Problems
Other than your personal dataset, you will need to use flare dataset.
1. Fit binary logistic regression for each level of categorical variable in the dataset.
2. Fit binary probit regression for each level of categorical variable in the dataset.
3. Choose test sample and compare test errors for both models above. What is the likelihood
of the models?
4. Fit multinomial logistic regression for categorical variable in the dataset. If there are only
two levels of this category, then choose some partitioning of continuous variable into 5
parts.
5. Fit Gamma regression for continuous variable. Is it better than Gaussian?
6. Fit Poisson regressions for number of 3 types of flares (see description) in flare dataset.


Recommendations
The function glm from the package stats offers a set of regression models. The syntax of this
function is similar to lm, only it is necessary to specify the distribution family of the dependent
variable in the argument family. Gaussian regression (the same as lm) is indicated using family
= gaussian. Logistic regression is specified using the family = binomial("logit") argument,
and probit regression using family = binomial ("probit"). Poisson regression is available
through family = poisson, and gamma regression is given by family = gamma. The set of
available link functions can be viewed by calling the help for family. Multinomial regression
can be implemented by the function multinom of the package multinom, the function mlogit
of the package mlogit, the function gam of the package mgcv.