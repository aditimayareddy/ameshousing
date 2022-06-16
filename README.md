# Ames Iowa House Price Predictions

This project focuses on accurate predictions of house prices in Ames, Iowa.
The dataset used is a subset of the Ames Iowa Housing Dataset from the Ames City Assessor's Office, made available by Dean De Cock of Truman University in 2011. Professor De Cock's original paper discussing the data can be found [here.](http://jse.amstat.org/v19n3/decock.pdf)

Presentation Slides for this project can be found [here.](https://www.beautiful.ai/player/-N4J5UYshyuRtwl5G4I7)

This project aims to create business value for real estate agents, buyers, sellers, and home-flippers in Ames, Iowa by:
* predicting property sale values
    * to help sellers appropriately valuate and list their homes
    * to help buyers make informed bidding decisions, and recognize underpriced or overpriced homes
* providing insights into what kinds of renovation recommendations should be made to clients looking to sell their home, or clients looking to buy a home to flip
* providing insights into what kinds of static house features potential buyers should prioritize, to maximize the potential resale value of their home - eg. Neighborhood, Lot Shape etc

### Models 
* Regression
  * EDA/Univariate Regression
  * Multiple Linear Regression
  * Lasso Regression
  * Elastic-Net
* Tree-based Models
  * Decision Tree
  * Random Forest
  * Boosting
* Support Vector Regression

### Model Validation and Feature Selection
For linear regression, features were selected if they were:
* significant (p-values)
* not multicolinear (did not inflate VIF of another feature)

For Tree-based models, features were selected using Lasso Regression.


## Code
* The code for this project is split across the following notebooks as follows:
  * `I. EDA & Cleaning` contains linear regression, elastic-net, interest rate, and backpropagation files
  * `II. Feature Engineering & Preprocessing` contains tree-based and SVR files
  * `III. Modeling & Evaluation` contains data cleaning, feature engineering, school district generation, and linear regression files
  * `helper_module` contains time series files
  * All of the above contain EDA as well
* Data (provided and generated) is in `data`
* Figures for presentation are in `figures`

    