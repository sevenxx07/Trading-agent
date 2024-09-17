# Warning
RL cannot be used to real trading agent because the environment is not stationary.
The data contain 1-hour windows with information about open, high, low, and close prices.
Those numbers mean the price at the start of the window, the highest price over the window
span, the lowest price, and the price at the end of the window. This data are not in this 
project because I have no rights to them 

# Problem description
I implemented those methods. Method reward function can be modified to define reward. It calculates the
logarithm of the percentual gain of your method.
Method make features constructs features. I used only static features. In this
method, I get a data frame with open, close, high, and low valuations, and your goal
will be to create numeric features that are good for learning. 
To calculate the feature at time t, I use only values from time ≤ t. 
 The positions available to the agent can be customized in get position list method. In
this method, I specify the set of actions available to the agent; the values between
0 and 1 mean split of the portfolio into USD and Bitcoin, and values outside this interval
represent borrowing either currency. There is a small cost for each exchange, as well as
for a loan. 
 Method get test position will be your policy in test. Use trained model here to provide
decisions.
 Method train is used for training. 
