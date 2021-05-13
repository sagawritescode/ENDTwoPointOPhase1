# Assignment 2 submission: Contains steps and screenshots

## Instructions

 - Same inputs, target outputs and initial weights which were discussed in the lecture were taken ![Image](https://github.com/sagawritescode/ENDTwoPointOPhase1/blob/main/Assignment2/images/2%20layer%20drawing.png)
 - The formula of every node and output of activation unit was written in terms of input, weights and output of nodes. Attaching screenshot for reference 
   - ![Image](https://github.com/sagawritescode/ENDTwoPointOPhase1/blob/main/Assignment2/images/formula%20for%20outputs%20and%20activated%20output%20in%20terms%20of%20weights.png)
 - The partial derivatives of the total error w.r.t every weight using chain rule were calculated. The calculations have been covered in the excel file. Attaching screenshot for reference 
   - ![Image](https://github.com/sagawritescode/ENDTwoPointOPhase1/blob/main/Assignment2/images/partial%20derivatives.png)
 - The first row was filled with formulas for all weights, node values, activated output of node values and the partial derivative of the total error w.r.t each weight
 - Weights are were updated using updation formula for each weight ```Wnew = Wold - Learning_Rate*(dE/dWold)``` in the next row
 - The formulas of the first row for next 51 rows and a graph of etotal was plotted 
 - For the various learning rates, below are the attached screenshots 
  - Learning Rate 0.1 ![Image](https://github.com/sagawritescode/ENDTwoPointOPhase1/blob/main/Assignment2/images/Learning%20rate%200.1.png)
  - Learning Rate 0.2 ![Image](https://github.com/sagawritescode/ENDTwoPointOPhase1/blob/main/Assignment2/images/Learning%20Rate%200.2.png)
  - Learning Rate 0.5 ![Image](https://github.com/sagawritescode/ENDTwoPointOPhase1/blob/main/Assignment2/images/Learning%20Rate%200.5.png)
  - Learning Rate 0.8 ![Image](https://github.com/sagawritescode/ENDTwoPointOPhase1/blob/main/Assignment2/images/Learning%20Rate%200.8.png)
  - Learning Rate 1 ![Image](https://github.com/sagawritescode/ENDTwoPointOPhase1/blob/main/Assignment2/images/Learning%20Rate%201.png)
  - Learning Rate 2 ![Image](https://github.com/sagawritescode/ENDTwoPointOPhase1/blob/main/Assignment2/images/Learning%20Rate%202.png)
  - From above images, it might be inferred that the more the learning rate, more faster is the rate with which the etotal reduces. But that is not the case. If the learning rate is exceedingly high then the Etotal might not converge to the minima. Following example of learning rate = 10000000 proves this ![Image](https://github.com/sagawritescode/ENDTwoPointOPhase1/blob/main/Assignment2/images/Learning%20Rate%2010000000.png)
