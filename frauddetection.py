# Import pandas and read csv
import pandas as pd
df = pd.read_csv("creditcard_data.csv")

# Explore the features available in your dataframe
print(df.info())

# Count the occurrences of fraud and no fraud and print them
occ = df['Class'].value_counts()
print(occ)

# Print the ratio of fraud cases
print(occ / len(df))

# Define a function to create a scatter plot of our data and labels
def plot_data(X, y):
	plt.scatter(X[y == 0, 0], X[y == 0, 1], label="Class #0", alpha=0.5, linewidth=0.15)
	plt.scatter(X[y == 1, 0], X[y == 1, 1], label="Class #1", alpha=0.5, linewidth=0.15, c='r')
	plt.legend()
	return plt.show()

# Create X and y from the prep_data function
X, y = prep_data(df)

# Plot our data by running our plot data function on X and y
plot_data(X, y)


#Applying SMOTE
#In this exercise, you're going to re-balance our data using the Synthetic Minority Over-sampling Technique (SMOTE).
# Unlike ROS, SMOTE does not create exact copies of observations, but creates new, synthetic,
# samples that are quite similar to the existing observations in the minority class. SMOTE is t
# herefore slightly more sophisticated than just copying observations, so let's apply SMOTE to our credit card data

# Get the mean for each group
df.groupby(by='Class').mean()

# Implement a rule for stating which cases are flagged as fraud
df['flag_as_fraud'] = np.where(np.logical_and(df['V1']<-3,df['V3']<-5), 1, 0)

# Create a crosstab of flagged fraud cases versus the actual fraud cases
print(pd.crosstab(df.Class, df.flag_as_fraud, rownames=['Actual Fraud'], colnames=['Flagged Fraud']))

# Create the training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=0)


#MML - Logistic Regression
# Fit a logistic regression model to our data
model = LogisticRegression()
model.fit(X_train,y_train)

# Obtain model predictions
predicted = model.predict(X_test)

# Print the classifcation report and confusion matrix
print('Classification report:\n', classification_report(y_test,predicted))
conf_mat = confusion_matrix(y_true=y_test, y_pred=predicted)
print('Confusion matrix:\n', conf_mat)