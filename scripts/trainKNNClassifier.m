function [trainedClassifier, validationAccuracy] = trainKNNClassifier(trainingData, responseData, neighbors, k_folds)

X = trainingData;
Y = responseData;

optimOpts.ShowPlots = false;
optimOpts.Verbose = 0;

% Train a classifier
% This code specifies all the classifier options and trains the classifier.
classificationKNN = fitcknn(...
    X, ...
    Y, ...
    'Distance', 'Euclidean', ...
    'Exponent', [], ...
    'NumNeighbors', neighbors, ...
    'DistanceWeight', 'Equal', ...
    'Standardize', true, ...
    'OptimizeHyperparameters', {'NumNeighbors'}, ...
    'HyperparameterOptimizationOptions', optimOpts, ...
    'ClassNames', {'anger'; 'happiness'; 'sadness'});

% Create the result struct with predict function
trainedClassifier.predictFcn = @(x) predict(classificationKNN, x);

% Add additional fields to the result struct
trainedClassifier.ClassificationKNN = classificationKNN;
trainedClassifier.About = 'This struct is a trained model exported from Classification Learner R2022a.';
trainedClassifier.HowToPredict = sprintf('To make predictions on a new predictor column matrix, X, use: \n  yfit = c.predictFcn(X) \nreplacing ''c'' with the name of the variable that is this struct, e.g. ''trainedModel''. \n \nX must contain exactly 47 columns because this model was trained using 47 predictors. \nX must contain only predictor columns in exactly the same order and format as your training \ndata. Do not include the response column or any columns you did not import into the app. \n \nFor more information, see <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appclassification_exportmodeltoworkspace'')">How to predict using an exported model</a>.');

% Perform cross-validation
partitionedModel = crossval(trainedClassifier.ClassificationKNN, 'KFold', k_folds);

% Compute validation predictions
[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

% Compute validation accuracy
validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');

close all;
