function [trainedClassifier, validationAccuracy] = trainNNClassifier(trainingData, responseData, layer_sizes, k_folds)
% [trainedClassifier, validationAccuracy] = trainClassifier(trainingData,
% responseData)
% Returns a trained classifier and its accuracy. This code recreates the
% classification model trained in Classification Learner app. Use the
% generated code to automate training the same model with new data, or to
% learn how to programmatically train models.
%
%  Input:
%      trainingData: A matrix with the same number of columns and data type
%       as the matrix imported into the app.
%
%      responseData: A vector with the same data type as the vector
%       imported into the app. The length of responseData and the number of
%       rows of trainingData must be equal.
%
%  Output:
%      trainedClassifier: A struct containing the trained classifier. The
%       struct contains various fields with information about the trained
%       classifier.
%
%      trainedClassifier.predictFcn: A function to make predictions on new
%       data.
%
%      validationAccuracy: A double containing the accuracy as a
%       percentage. In the app, the Models pane displays this overall
%       accuracy score for each model.
%
% Use the code to train the model with new data. To retrain your
% classifier, call the function from the command line with your original
% data or new data as the input arguments trainingData and responseData.
%
% For example, to retrain a classifier trained with the original data set T
% and response Y, enter:
%   [trainedClassifier, validationAccuracy] = trainClassifier(T, Y)
%
% To make predictions with the returned 'trainedClassifier' on new data T2,
% use
%   yfit = trainedClassifier.predictFcn(T2)
%
% T2 must be a matrix containing only the predictor columns used for
% training. For details, enter:
%   trainedClassifier.HowToPredict

% Auto-generated by MATLAB on 28-Apr-2022 10:26:11


% Extract predictors and response
% This code processes the data into the right shape for training the
% model.
% Convert input to table
%inputTable = array2table(trainingData, 'VariableNames', {'column_1', 'column_2', 'column_3', 'column_4', 'column_5', 'column_6', 'column_7', 'column_8', 'column_9', 'column_10', 'column_11', 'column_12', 'column_13', 'column_14', 'column_15', 'column_16', 'column_17', 'column_18', 'column_19', 'column_20', 'column_21', 'column_22', 'column_23', 'column_24', 'column_25', 'column_26', 'column_27', 'column_28'});

%predictorNames = {'column_1', 'column_2', 'column_3', 'column_4', 'column_5', 'column_6', 'column_7', 'column_8', 'column_9', 'column_10', 'column_11', 'column_12', 'column_13', 'column_14', 'column_15', 'column_16', 'column_17', 'column_18', 'column_19', 'column_20', 'column_21', 'column_22', 'column_23', 'column_24', 'column_25', 'column_26', 'column_27', 'column_28'};
%predictors = inputTable(:, predictorNames);
%response = responseData;

idx = randperm(numel(responseData));
idx_train = idx(1:round(0.8*numel(idx)));
idx_val = idx(round(0.8*numel(idx))+1:end);

validation_data = {};
validation_data{1} = trainingData(idx_val,:);
validation_data{2} = responseData(idx_val);

X = trainingData(idx_train,:);
Y = responseData(idx_train);

% Train a classifier
% This code specifies all the classifier options and trains the classifier.
classificationNeuralNetwork = fitcnet(...
    X, ...
    Y, ...
    'LayerSizes', layer_sizes, ... % default [10, 10, 10]
    'Activations', 'relu', ...
    'Lambda', 0, ... % default 0
    'IterationLimit', 1000, ...
    'Standardize', true, ...
    'Verbose', false, ...
    'ValidationData', validation_data, ...
    'ClassNames', {'anger'; 'happiness'; 'sadness'});

% Create the result struct with predict function
%predictorExtractionFcn = @(x) array2table(x, 'VariableNames', predictorNames);
%neuralNetworkPredictFcn = @(x) predict(classificationNeuralNetwork, x);
trainedClassifier.predictFcn = @(x) predict(classificationNeuralNetwork, x);
%trainedClassifier.predictFcn = @(x) neuralNetworkPredictFcn(predictorExtractionFcn(x));

% Add additional fields to the result struct
trainedClassifier.ClassificationNeuralNetwork = classificationNeuralNetwork;
trainedClassifier.About = 'This struct is a trained model exported from Classification Learner R2022a.';
trainedClassifier.HowToPredict = sprintf('To make predictions on a new predictor column matrix, X, use: \n  yfit = c.predictFcn(X) \nreplacing ''c'' with the name of the variable that is this struct, e.g. ''trainedModel''. \n \nX must contain exactly 47 columns because this model was trained using 47 predictors. \nX must contain only predictor columns in exactly the same order and format as your training \ndata. Do not include the response column or any columns you did not import into the app. \n \nFor more information, see <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appclassification_exportmodeltoworkspace'')">How to predict using an exported model</a>.');

% Perform cross-validation
%partitionedModel = crossval(trainedClassifier.ClassificationNeuralNetwork, 'KFold', k_folds);

% Compute validation predictions
%[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

% Compute validation accuracy
%validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');

validationAccuracy = 1 - classificationNeuralNetwork.TrainingHistory.ValidationLoss(end-6);
