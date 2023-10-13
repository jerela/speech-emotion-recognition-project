function [trainedClassifier, validationAccuracy] = trainSVMClassifier(trainingData, responseData, k_folds)

X = trainingData;
Y = responseData;

optimOpts.ShowPlots = false;
optimOpts.Verbose = 0;

% Train a classifier
% This code specifies all the classifier options and trains the classifier.
template = templateSVM(...
    'KernelFunction', 'gaussian', ... % default linear
    'PolynomialOrder', [], ...
    'KernelScale', 6, ... % default 1
    'BoxConstraint', 240, ... % default 1
    'Standardize', true);
classificationSVM = fitcecoc(...
    X, ...
    Y, ...
    'Learners', template, ...
    'Coding', 'onevsone', ...
    'OptimizeHyperParameters', 'auto', ...
    'HyperparameterOptimizationOptions', optimOpts, ...
    'ClassNames', {'anger'; 'happiness'; 'sadness'});

% Create the result struct with predict function
trainedClassifier.predictFcn = @(x) predict(classificationSVM, x);

% Add additional fields to the result struct
trainedClassifier.ClassificationSVM = classificationSVM;
trainedClassifier.About = 'This struct is a trained model exported from Classification Learner R2022a.';
trainedClassifier.HowToPredict = sprintf('To make predictions on a new predictor column matrix, X, use: \n  yfit = c.predictFcn(X) \nreplacing ''c'' with the name of the variable that is this struct, e.g. ''trainedModel''. \n \nX must contain exactly 47 columns because this model was trained using 47 predictors. \nX must contain only predictor columns in exactly the same order and format as your training \ndata. Do not include the response column or any columns you did not import into the app. \n \nFor more information, see <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appclassification_exportmodeltoworkspace'')">How to predict using an exported model</a>.');

% Perform cross-validation
partitionedModel = crossval(trainedClassifier.ClassificationSVM, 'KFold', k_folds);

% Compute validation predictions
[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

% Compute validation accuracy
validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');
