function [trainedClassifier, validationAccuracy] = trainNNClassifier(trainingData, responseData, layer_sizes, k_folds)

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
trainedClassifier.predictFcn = @(x) predict(classificationNeuralNetwork, x);

% Add additional fields to the result struct
trainedClassifier.ClassificationNeuralNetwork = classificationNeuralNetwork;
trainedClassifier.About = 'This struct is a trained model exported from Classification Learner R2022a.';
trainedClassifier.HowToPredict = sprintf('To make predictions on a new predictor column matrix, X, use: \n  yfit = c.predictFcn(X) \nreplacing ''c'' with the name of the variable that is this struct, e.g. ''trainedModel''. \n \nX must contain exactly 47 columns because this model was trained using 47 predictors. \nX must contain only predictor columns in exactly the same order and format as your training \ndata. Do not include the response column or any columns you did not import into the app. \n \nFor more information, see <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appclassification_exportmodeltoworkspace'')">How to predict using an exported model</a>.');

validationAccuracy = 1 - classificationNeuralNetwork.TrainingHistory.ValidationLoss(end-6);
