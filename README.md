# Titanic Survival Prediction Machine Learning Model

### Overview

There is a [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic/overview)
competition on Kaggle to create a model to see how well you can do at predicting survival based on available data.  

I am not a data scientist (as you might be able to deduce) but I have been working in an Azure ML environment
for around a year, so that has piqued my interest in AI.  I have been trying to learn as much as I
can on the subject, but had no previous knowledge of AI or Machine Learning.  The invent of ChatGPT opens up a whole
new world of learning opportunities, so this repo was an experimentation into what could be achieved without the
pre-requisite data science skills to create a model that could produce output to allow a submission to the competition.  

You could say this is cheating, you are just the copying and pasting code to get a result, which is mostly right.
But there is an overall contextual learning opportunity in using this method, in that from only understanding the requirement you are able to get working output that you can then drill into to understand the generated solution.  

For example:

`Me:`  
`"X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)`
"explain this code in words a 15 year old would understand"

`ChatGPT:`  
> This code is dividing a dataset into two parts: one part is going to be used to train a machine learning model, and the other part is going to be used to evaluate how well the model performs.
>
> The dataset is represented by two matrices: **X** (which contains the features or input variables) and **y** (which contains the target or output variable). In this case, the dataset represents information about passengers on the Titanic, and the target variable is whether or not each passenger survived the sinking.
>
> The **train_test_split** function takes in these two matrices (**X** and **y**) and splits them into four new matrices: **X_train**, **X_val**, **y_train**, and **y_val**. The function randomly assigns each row of the original dataset to either the training set or the validation set.
>
> The **test_size=0.2** argument tells the function to assign 20% of the original dataset to the validation set, meaning that the remaining 80% of the data will be used to train the model.
>
> The **random_state=42** argument ensures that the data is split in the same way each time the code is run, which allows us to reproduce the same results if we need to.
>
> Once the data is split into **X_train**, **X_val**, **y_train**, and **y_val**, we can use **X_train** and **y_train** to train a machine learning model, and **X_val** and **y_val** to evaluate how well the model performs on new data that it has not seen before.

__Note:__ The [Titanic Tutorial](https://www.kaggle.com/code/alexisbcook/titanic-tutorial) provides the classic way to develop the machine learning model for this challenge.


### GPT-4 prompts

In total I used the following prompts to GPT-4 to get to the point where I could submit a competition submission to Kaggle:

* __based on the titanic CSV data, can you create a machine learning model to determine if a passenger was likely to survive?__
* __can you show me how to do this in Python?__
* __how do I save the model to a file?__
* __how do i convert the survived column to an integer?__
* __how can I improve the success rate of the model?__
* __can you show me in Python how I can make success rate improvements?__
* __how do I know what inputs to pass to the model?__

_Note:_ Some of the prompts were just used for additional information
and the output not incorporated into the code.

### Result

The code in this repo is the result of the output from the above prompts.
The training and test data is not included, as you will need to get that from
[Kaggle](https://www.kaggle.com/c/titanic/data).

The first submission I did had a zero score, which didn't seem right because even
if the model was rubbish the law of averages should suggest some kind of result above zero.
This is the reason for the __how do i convert the survived column to an integer?__ prompt.

![submissions](images/submissions.png)

After fixing that issue and resubmitting I was very pleased to see a score of **0.77511**,
putting me in a position of **9907** out of **16324** contenders on the [leaderboard](https://www.kaggle.com/competitions/titanic/leaderboard?search=skidmore) (at time of writing).
I know this isn't a great result, but it is a start and produced what I set out to achieve.

![titanic-leaderboard](images/kaggle_leaderboard.png)

Maybe I am a being a bit of a charlatan to feel a sense of achievement by using AI to get the result, but
on the other hand I have been using Google for years to achieve the same ends.  So no,
I am counting it as a new skill 😜.

The initial inspiration for using the Titanic example came from watching another great [YouTube video](https://www.youtube.com/watch?v=Ebb4gUI2IpQ&t=90s) from [Nicholas Renotte](https://github.com/nicknochnack).
