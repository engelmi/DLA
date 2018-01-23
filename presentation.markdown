#Presentation
## Part 1
### Idea
* predict stock value of the next day with information from google
* found two datasets on kaggle
    * nyse -> keywords for google search
    * s&p500 (Standard and Poors 500 biggest US companies) -> used the stock data
* google terrible no api
    * get a score for a keyword in a time
    * how often was it requested
    * get blocked from google because of to many requests
* build own data set
    * label computed
    * stock price
    * 14 google values
    * split into groups of 30 days
    * only very small (training 209, testing 16)

## Part 2
* lstm
    * recurrent neural network
    * good against exploding or vanishing gradients
* our model
    * 30 timesteps -> 30 lstm cells
    * hidden size of 100
    * dropout possible
    * k-cross validation
* evaluation / conclusion