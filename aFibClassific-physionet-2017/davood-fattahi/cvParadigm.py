# Cross validation:
# •	For j=1:J times
#     o	Split the data to K-fold
#     o	For k=1:k do
#         	testData=fold(k), trainData=fold(~k)
#         	standardize trainingData, and save the standardization setting
#         	by voting, initially select features for each feature group and trainData
#         	Concatenate and score the initially selected features using different scoring methods.
#         	bestFeatures(j,k) = select the best features by voting
#         	train the classifier based on the bestFeatures(j,k)
#         	compute the training and test error
# •	Final best features = selectByVote(bestFeatures)
# •	Final scores = average score(j,k)
