I have been working on predicting and optimizing the Radar Cross Section (RCS) of aircraft. Initially, I employed Artificial Neural Networks (ANNs) for this purpose; however, the results were suboptimal. Consequently, I transitioned to simpler statistical approaches, particularly regression analysis, which significantly improved the accuracy, reducing the error to 18%.

In my analysis, I explored several regression methods, including:
- Polynomial Regression
- Random Forest Regression
- Decision Tree Regression
- Gradient Boosting Regression

Among these, Gradient Boosting Regression yielded the most favorable outcome, achieving the lowest mean squared error (MSE) of 18%. I subsequently applied this model to predict the RCS of a C130 aircraft, comparing the results with simulation data. While there were discrepancies between the predicted and simulated values, this was primarily due to the use of a smaller-scale model (mm dimensions) during the simulation.

Despite these differences, the overall trends were consistent. After refining the code further, I successfully reduced the error to an MSE of 3.45%.

This concludes the first phase of the project—predicting and improving the RCS using regression analysis. The next phase involves error prediction and correction. To achieve this, I developed code to quantify the differences between two methods of RCS calculation (vectorial summation and simulation). I then optimized the code to minimize the error and enhance the accuracy of the predictions.

Having achieved the desired results in regression and optimization, the project is now progressing toward further analysis and documentation. The upcoming steps include:
- Generating additional vectorial summation comparison plots for different aircraft to better understand the relationships between variables.
- Optimizing these plots to highlight key insights.
- Writing the thesis, incorporating the results obtained throughout the project.
- Revisiting the MATLAB-based ANN model to address the MSE issue and improve its predictive performance.
