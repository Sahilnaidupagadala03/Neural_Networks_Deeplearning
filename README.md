# Neural_Networks_Deeplearning Assignment 1
Neural_Network_ICP1_700748694 summary:
Q1a: The code takes a user-inputted string, removes at least the first two characters, reverses the remaining characters, and then prints the result.
Q1b: The code takes two numbers from the user, performs basic arithmetic operations, and then prints the results, taking care to handle the case where division by zero might occur.
Q2: The code allows the user to input a sentence, replaces occurrences of 'python' with 'pythons', and then displays the modified sentence.
Q3: The code allows the user to input a class score, determines the corresponding letter grade based on a predefined grading scale, and then displays the result. If the entered score is outside the valid range, it is labeled as "Invalid score."
# Recording of ICP1 (Video link): 
https://youtu.be/i_VSd-_4CiE

# Neural_Networks_Deeplearning Assignment 2
Neural_Network_ICP2_700748694 summary:
Q1a: This Python code defines a function called fullname that takes two parameters (first_name and last_name), concatenates them with a space in between, and returns the full name. The code then prompts the user to enter their first and last names, calls the fullname function with the provided inputs, and prints the resulting full name.
Q1b: The code defines a function string_alternative that returns every second character of a given string. It then uses this function to process a previously obtained full name (result), displaying both the original full name and the result obtained by selecting every second character.
Q2: 
The code defines a Python function wordcount_per_line that reads lines from an input file, counts the occurrences of each word, and writes both the original lines and the word counts to an output file. The main part of the code specifies input and output file names, calls the function with these names, and handles the case where the input file is not found.
Q3a: 
The code defines a function to convert heights from inches to centimeters. It prompts the user to enter the number of customers and their heights in inches, converting and storing the values. The code handles invalid input and then prints both the input heights in inches and the corresponding converted heights in centimeters.
Q3b: 
The code initializes a list of heights in inches, prints the original list, and then creates a new list by converting each height to centimeters (multiplied by 2.54). The code prints both the original list in inches and the new list representing the heights in centimeters.
# Recording of ICP2 (Video link): 
https://youtu.be/7FIybRw69Ds

# Neural_Networks_Deeplearning Assignment 3
Neural_Network_ICP3_700748694 summary:
Q1: 
The code illustrates inheritance by defining an Employee class that has a subclass called FullTimeEmployee. It includes a static method to determine the average wage of a list of employees called calculate_average_salary. Both classes contain an initialization method called init, and FullTimeEmployee uses super() to initialize the parent class. The total number of workers is tracked through the usage of a class variable called total_employees. Instances of both classes are generated, added to a list, and the average wage is computed and shown in the main function. A dedicated method, display_fulltime_benefits, is introduced by the FullTimeEmployee class to emphasize benefits for full-time employees.

Q2: 
The code generates a random 1D array, reshapes it into a 2D array, finds the indices of maximum values in each row, and replaces those maximum values with 0 in the respective rows.
# Recording of ICP3 (Video link): 
https://youtu.be/UUfwtnK8uqo

# Neural_Networks_Deeplearning Assignment 4
Neural_Network_ICP4_700748694 summary:
Q1: a:  Importing pandas as pan. x = pd.read_csv('data.csv') line reads csv file named data.csv and stores data in ‘x’.  
c: output a summary of the statistics for each numeric column in dataframe. 
d: x.isnull().any() expression is used to check whether there are any Null values in each column and .any() extension to it checks if there is at least one missing value in each column. 
i) fillna method to replace missing values in x with the mean of each column. Mean() calculates the mean separately and inpace = true modifies the original data frame x. 
e: This code computes the following aggregations for the 'Pulse' and 'Maxpulse' columns in the DataFrame x.
f: .loc locates the values of the calories between 500 and 1000.
g:  .loc [….] locates the data values for calories greater than 500 and pulse less than 100 and filters the table and store in x. Output is printed. 
h: df_modified stores the newer data frame with Duration, Pulse and Calories in it.
i: Del deletes the Maxpulse coloumn from the x data frame.
j: .datatypes prints the data type of the respective column and .astype() converts the datatype of the required column and calorie in this case as int.
k: .plot.scatter() will create a scatter plot graph in the output. Here with Duration on x axis, calories o the y axis and the plot colour will be Dark blue. 

All the imports namely seaborn, numpy, pandas, matplotlib.pyplot, train_test_split from sklearn.model_selection, sklearn.learn_model that has LinearRegression lib init, metrics and preprocessing from sklearn and mean_squared_error from sklearn.metrics.  
Q2: a: .read_csv() to read the salary_Data dataset csv file.
b: X: This line extracts the features from your Data Frame. It selects all rows and all columns except the last one (iloc[:,:-1]). The result is assigned to the variable X. 
Y: This line extracts the target variable from your Data Frame. It selects all rows and only the last column (iloc[:, 1]). The result is assigned to the variable Y. train_test_split() function is used to split your dataset into training and testing sets. It takes the features (X) and the target variable (Y) and test_size=1/3, random_state=0 parameters and returns four sets of data for train and splits fr X and Y. 
c: LinearRegression is a simple linear regression model used for predicting a target variable based on one or more independent variables. regressor.fit(X_Train, Y_Train): This line trains (fits) the linear regression model using the training data. It takes the training features X_Train and the corresponding target variable Y_Train to learn the coefficients of the linear regression equation. Y_Pred line predicts the target variable (Y_Pred) for the testing set X_Test using the trained linear regression model. 
d: mean_squared_error() function sklearn.metrics will give the mean swaure value which will be the error.
e: Xlabel and Y label are the titles for the x and y axis and title() is the method used for naming the entire plot. Plt.scatter() will prepare  a scatter plot for the values inserted and .show() will give the plot output. 
# Recording of ICP2 (Video link): 
https://youtu.be/Nb10efzPFhM
