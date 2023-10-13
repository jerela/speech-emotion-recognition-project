clc;clear;close all;

path_root = 'C:\Users\lavik\OneDrive\Documents\Machine Learning for Speech\Project\Unpacked';

path_extracted = 'C:\Users\lavik\OneDrive\Documents\Machine Learning for Speech\Project\Extracted_data';

% target sampling rate for resampling
f_target = 16000;


%% extract data from AESDD
data_path = 'AESDD';
subfolders = {'anger', 'disgust', 'fear', 'happiness', 'sadness'};

labels = {};
data = {};

for subfolder = 1:length(subfolders)
    
    
    
    % get emotion
    emotion = subfolders{subfolder};
    
    % switch to target folder
    cd(fullfile(path_root,data_path,subfolders{subfolder}))
    
    % find .wav files
    wavs = dir('*.wav');

    for j = 1:length(wavs)
        % get file name
        file_name = wavs(j).name;
        % load audio file; if failed, skip it
        try
            [Y FS] = audioread(file_name);
        catch
            continue
        end
        % if there are multiple channels, pick just 1
        y = Y(:,1);
        % if sampling rate is not 16 kHz, filter and resample to 16 kHz
        if FS ~= 16000
            % design and apply a lowpass filter at 16 Hz
            [b,a] = butter(4,f_target/FS,'low');
            y = filtfilt(b,a,y);
            % resample the filtered waveform
            y = resample(y,f_target,FS);
            
        end
        
        % get subject
        subject = str2num(file_name(end-5));
        % get utterance
        utterance = str2num(file_name(2:3));
        
        % save to struct
        label = struct();
        label.emotion = emotion;
        label.subject = subject;
        label.utterance = utterance;
        label.dataset = 'AESDD';
        labels{numel(labels)+1} = label;
        % save waveform to struct
        data{numel(data)+1} = y;
        
    end
    
    
end


data_AESDD = data;
labels_AESDD = labels;
clearvars -except data_* labels_* f_target path_*

cd(path_extracted)
save('data_AESDD','data_AESDD')
save('labels_AESDD','labels_AESDD')

%% extract data from Berlin emotional speech dataset
data_path = 'Berlin\wav';

labels = {};
data = {};

% switch to target folder
cd(fullfile(path_root,data_path))

% find .wav files
wavs = dir('*.wav');

for j = 1:length(wavs)
    % get file name
    file_name = wavs(j).name;
    % load audio file; if failed, skip it
    try
        [Y FS] = audioread(file_name);
    catch
        continue
    end
    % if there are multiple channels, pick just 1
    y = Y(:,1);
    % if sampling rate is not 16 kHz, filter and resample to 16 kHz
    if FS ~= 16000
        % design and apply a lowpass filter at 16 Hz
        [b,a] = butter(4,f_target/FS,'low');
        y = filtfilt(b,a,y);
        % resample the filtered waveform
        y = resample(y,f_target,FS);
    end
    
    % get subject
    subject = str2num(file_name(1:2));
    % get utterance
    utterance = file_name(3:5);
    % get emotion
    emotion = file_name(6);

    % uniformize emotions
    switch emotion
        case 'W'
            emotion = 'anger';
        case 'L'
            emotion = 'boredom';
        case 'E'
            emotion = 'disgust';
        case 'A'
            emotion = 'fear';
        case 'F'
            emotion = 'happiness';
        case 'T'
            emotion = 'sadness';
    end

    % save to struct
    label = struct();
    label.emotion = emotion;
    label.subject = subject;
    label.utterance = utterance;
    label.dataset = 'Berlin';
    labels{numel(labels)+1} = label;
    % save waveform to struct
    data{numel(data)+1} = y;
    
end

    

data_Berlin = data;
labels_Berlin = labels;
clearvars -except data_* labels_* path_root f_target path_*

cd(path_extracted)
save('data_Berlin','data_Berlin')
save('labels_Berlin','labels_Berlin')


%% extract data from Oreau dataset
data_path = 'OréauFR_02\OréauFR_02';

labels = {};
data = {};

gender_folders = {'m','f'};




for gender_folder_i = 1:length(gender_folders)
    gender_folder = gender_folders{gender_folder_i};
    
    subfolders = dir(fullfile(path_root,data_path,gender_folder));
    subfolders = subfolders(3:end);
    subfolders = {subfolders.name};


    for subfolder = 1:length(subfolders)
    
        % switch to target folder
        cd(fullfile(path_root,data_path,gender_folder,subfolders{subfolder}))
    
        % find .wav files
        wavs = dir('*.wav');
        
        for j = 1:length(wavs)
            % get file name
            file_name = wavs(j).name;
            % load audio file; if failed, skip it
            try
                [Y FS] = audioread(file_name);
            catch
                continue
            end
            % if there are multiple channels, pick just 1
            y = Y(:,1);
            % if sampling rate is not 16 kHz, filter and resample to 16 kHz
            if FS ~= 16000
                % design and apply a lowpass filter at 16 Hz
                [b,a] = butter(4,f_target/FS,'low');
                y = filtfilt(b,a,y);
                % resample the filtered waveform
                y = resample(y,f_target,FS);
            end
            
            % remove ".wav" from file name
            file_name = file_name(1:end-4);
    
            % get emotion
            emotion = file_name(6);
            % get subject
            subject = file_name(1:2);
            % get utterance
            utterance = file_name(3:5);

            % uniformize emotions
            switch emotion
                case 'C'
                    emotion = 'anger';
                case 'T'
                    emotion = 'sadness';
                case 'J'
                    emotion = 'happiness';
                case 'P'
                    emotion = 'fear';
                case 'D'
                    emotion = 'disgust';
                case 'S'
                    emotion = 'surprise';
                case 'N'
                    emotion = 'neutral';
            end
        
            % save to struct
            label = struct();
            label.emotion = emotion;
            label.subject = subject;
            label.utterance = utterance;
            label.dataset = 'Oreau';
            labels{numel(labels)+1} = label;
            % save waveform to struct
            data{numel(data)+1} = y;
            
        end
    end

end
    

data_Oreau = data;
labels_Oreau = labels;
clearvars -except data_* labels_* f_target path_*



cd(path_extracted)
save('data_Oreau','data_Oreau')
save('labels_Oreau','labels_Oreau')



%% extract data from RAVDESS dataset
data_path = 'Ryerson';

labels = {};
data = {};

subfolders = dir(fullfile(path_root,data_path));
subfolders = subfolders(3:end);
subfolders = {subfolders.name};



for subfolder = 1:length(subfolders)

    % switch to target folder
    cd(fullfile(path_root,data_path,subfolders{subfolder}))

    % find .wav files
    wavs = dir('*.wav');
    
    for j = 1:length(wavs)
        % get file name
        file_name = wavs(j).name;
        % load audio file; if failed, skip it
        try
            [Y FS] = audioread(file_name);
        catch
            continue
        end
        % if there are multiple channels, pick just 1
        y = Y(:,1);
        % if sampling rate is not 16 kHz, filter and resample to 16 kHz
        if FS ~= 16000
            % design and apply a lowpass filter at 16 Hz
            [b,a] = butter(4,f_target/FS,'low');
            y = filtfilt(b,a,y);
            % resample the filtered waveform
            y = resample(y,f_target,FS);
        end
        
        % split file name into substrings based on underscore delimiter
        file_name_substrings = strsplit(file_name(1:end-4),'-');

        % get emotion
        emotion = file_name_substrings{3};
        % get subject
        subject = file_name_substrings{7};
        % get utterance
        utterance = file_name_substrings{5};
        
        % uniformize emotions
        switch emotion
            case '01'
                emotion = 'neutral';
            case '02'
                emotion = 'calm';
            case '03'
                emotion = 'happiness';
            case '04'
                emotion = 'sadness';
            case '05'
                emotion = 'anger';
            case '06'
                emotion = 'fear';
            case '07'
                emotion = 'disgust';
            case '08'
                emotion = 'surprise';
        end

        % save to struct
        label = struct();
        label.emotion = emotion;
        label.subject = subject;
        label.utterance = utterance;
        label.dataset = 'RAVDESS';
        labels{numel(labels)+1} = label;
        % save waveform to struct
        data{numel(data)+1} = y;
        
    end
end
    

data_RAVDESS = data;
labels_RAVDESS = labels;
clearvars -except data_* labels_* path_* f_target



cd(path_extracted)
save('data_RAVDESS','data_RAVDESS')
save('labels_RAVDESS','labels_RAVDESS')




%% extract data from Toronto dataset
data_path = 'Toronto';

labels = {};
data = {};

% switch to target folder
cd(fullfile(path_root,data_path))

% find .wav files
wavs = dir('*.wav');

for j = 1:length(wavs)
    % get file name
    file_name = wavs(j).name;
    % load audio file; if failed, skip it
    try
        [Y FS] = audioread(file_name);
    catch
        continue
    end
    % if there are multiple channels, pick just 1
    y = Y(:,1);
    % if sampling rate is not 16 kHz, filter and resample to 16 kHz
    if FS ~= 16000
        % design and apply a lowpass filter at 16 Hz
        [b,a] = butter(4,f_target/FS,'low');
        y = filtfilt(b,a,y);
        % resample the filtered waveform
        y = resample(y,f_target,FS);
    end
    
    % split file name into substrings based on underscore delimiter
    file_name_substrings = strsplit(file_name(1:end-4),'_');

    % get subject
    subject = file_name_substrings{1};
    % get utterance
    utterance = file_name_substrings{2};
    % get emotion
    emotion = file_name_substrings{3};

    % uniformize emotions
    switch emotion
        case 'angry'
            emotion = 'anger';
        case 'happy'
            emotion = 'happiness';
        case 'sad'
            emotion = 'sadness';
    end

    % save to struct
    label = struct();
    label.emotion = emotion;
    label.subject = subject;
    label.utterance = utterance;
    label.dataset = 'TESS';
    labels{numel(labels)+1} = label;
    % save waveform to struct
    data{numel(data)+1} = y;
    
end

    

data_TESS = data;
labels_TESS = labels;
clearvars -except data_* labels_* path_* f_target


cd(path_extracted)
save('data_TESS','data_TESS')
save('labels_TESS','labels_TESS')


%% extract data from Urdu dataset
data_path = 'URDU-Dataset-master';

labels = {};
data = {};

subfolders = {'Angry', 'Happy', 'Neutral', 'Sad'};




for subfolder = 1:length(subfolders)

    % switch to target folder
    cd(fullfile(path_root,data_path,subfolders{subfolder}))

    emotion = subfolders{subfolder};

    % uniformize emotions
    switch emotion
        case 'Angry'
            emotion = 'anger';
        case 'Happy'
            emotion = 'happiness';
        case 'Sad'
            emotion = 'sadness';
        case 'Neutral'
            emotion = 'neutral';
    end

    % find .wav files
    wavs = dir('*.wav');
    
    for j = 1:length(wavs)
        % get file name
        file_name = wavs(j).name;
        % load audio file; if failed, skip it
        try
            [Y FS] = audioread(file_name);
        catch
            continue
        end
        % if there are multiple channels, pick just 1
        y = Y(:,1);
        % if sampling rate is not 16 kHz, filter and resample to 16 kHz
        if FS ~= 16000
            % design and apply a lowpass filter at 16 Hz
            [b,a] = butter(4,f_target/FS,'low');
            y = filtfilt(b,a,y);
            % resample the filtered waveform
            y = resample(y,f_target,FS);
        end
        
        % split file name into substrings based on underscore delimiter
        file_name_substrings = strsplit(file_name(1:end-4),'_');
    
        % get subject
        subject = file_name_substrings{1};
        % get utterance
        utterance = file_name_substrings{2};
    
        % save to struct
        label = struct();
        label.emotion = emotion;
        label.subject = subject;
        label.utterance = utterance;
        label.dataset = 'Urdu';
        labels{numel(labels)+1} = label;
        % save waveform to struct
        data{numel(data)+1} = y;
        
    end
end
    

data_Urdu = data;
labels_Urdu = labels;
clearvars -except data_* labels_* path_* f_target


cd(path_extracted)
save('data_Urdu','data_Urdu')
save('labels_Urdu','labels_Urdu')










%% Combine all data

cd(path_extracted)

data = {};
labels = {};

dataset_names = {'AESDD', 'Berlin', 'Oreau', 'RAVDESS', 'TESS', 'Urdu'};

dataset_ids = [];

files_data = dir('data_*');
files_labels = dir('labels_*');

for i = 1:length(dataset_names)
    % load the cell arrays containing data from individual datasets
    data_current = load(['data_' dataset_names{i} '.mat']);
    data_current = data_current.(['data_' dataset_names{i}]);
    labels_current = load(['labels_' dataset_names{i} '.mat']);
    labels_current = labels_current.(['labels_' dataset_names{i}]);
    % concatenate cell arrays
    data = [data data_current];
    labels = [labels labels_current];
    dataset_ids = [dataset_ids; ones(length(data_current),1)*i];
end



%% for all vectors in data, perform feature extraction

% length of a frame in milliseconds
frame_length_ms = 25;
% length of a hop between frame starting points in milliseconds
frame_hop_ms = 10;
frame_length = frame_length_ms*f_target*1e-3;
frame_hop = frame_hop_ms*f_target*1e-3;
window = hamming(round(frame_length));

%cd(data_paths{1})
aFE = audioFeatureExtractor(SampleRate = f_target, ...
    Window = window, ...
    OverlapLength=frame_length-frame_hop, ...
    FFTLength = numel(window), ...
    SampleRate = f_target, ...
    SpectralDescriptorInput = 'melSpectrum');

% define features to extract
%aFE.linearSpectrum = true;
%aFE.melSpectrum = true;
%afe.barkSpectrum = true;
%afe.erbSpectrum = true;
aFE.mfcc = true;
%aFE.mfccDelta = true;
%aFE.mfccDeltaDelta = true;
%aFE.gtcc = true;
aFE.spectralCentroid = true;
aFE.spectralCrest = true;
aFE.spectralDecrease = true;
aFE.spectralEntropy = true;
aFE.spectralFlatness = true;
aFE.spectralFlux = true;
aFE.spectralKurtosis = true;
aFE.spectralRolloffPoint = true;
aFE.spectralSkewness = true;
aFE.spectralSlope = true;
aFE.spectralSpread = true;
aFE.pitch = true;
setExtractorParameters(aFE, 'pitch', Range=[50,400]);
aFE.harmonicRatio = true;
aFE.zerocrossrate = true;
aFE.shortTimeEnergy = true;

idx = info(aFE);

final_features = {};
final_features_matrix = [];

for data_i = 1:length(data)

    y = data{data_i};

    %% voice activity detection
    
    % calculate energy for the whole waveform
    energy = y.^2;
    
    % design a smoothing window for smoothing the energy
    % how many milliseconds should the smoothing window be
    smoothing_window_ms = 200;
    % create the window itself
    smoothing_window = ones(1,round(smoothing_window_ms*1e-3*f_target));
    
    % smooth the energy
    energy_smoothed = conv(energy,smoothing_window,'same');
    
    % get indices where voice activity is detected with median as threshold
    voice_indices = energy_smoothed > median(energy_smoothed);
    
    % trim y to only include the voice indices
    y_trimmed = y(voice_indices);
    
    %% feature extraction
    

    features = extract(aFE,y_trimmed);
    final_features{numel(final_features)+1} = mean(features);
    final_features_matrix = [final_features_matrix; mean(features)];


end

%% get final class labels

final_labels = {};

for label_i = 1:length(labels)
    final_labels{numel(final_labels)+1} = labels{label_i};
end




%% get data only from the emotions common to all datasets (anger, happiness, sadness)

final_emotions_common = {};
final_subjects_common = {};
final_datasets_common = {};
final_features_common = {};
final_features_matrix_common = [];
dataset_ids_common = [];

for label_i = 1:length(final_labels)
    current_emotion = final_labels{label_i}.emotion;
    current_subject = final_labels{label_i}.subject;
    current_dataset = final_labels{label_i}.dataset;
    if strcmp(current_emotion,'anger') || strcmp(current_emotion,'sadness') || strcmp(current_emotion,'happiness')
        final_emotions_common{numel(final_emotions_common)+1} = current_emotion;
        final_subjects_common{numel(final_subjects_common)+1} = current_subject;
        final_datasets_common{numel(final_datasets_common)+1} = current_dataset;
        final_features_common{numel(final_features_common)+1} = final_features{label_i};
        final_features_matrix_common = [final_features_matrix_common; final_features_matrix(label_i,:)];
        dataset_ids_common = [dataset_ids_common; dataset_ids(label_i)];
    end
end

cd(path_extracted)
save('final_data','dataset_ids_common', 'final_features_matrix_common', 'final_emotions_common', 'final_subjects_common', 'final_datasets_common', 'dataset_names', 'idx')


%% correlation analysis between predictors and target

% construct numerical labels
final_labels_common_numerical = zeros(size(final_emotions_common'));
emotions_common = unique(final_emotions_common');
for emotion_i = 1:length(emotions_common)
    final_labels_common_numerical(strcmp(final_emotions_common, emotions_common{emotion_i})) = emotion_i;
end

% go through all features and compare correlation
feature_names = fieldnames(idx)
correlation_coeffs = [];
for feature_i = 1:length(feature_names)
    current_idx = idx.(feature_names{feature_i});
    current_feature = final_features_matrix_common(:,current_idx);
    try
        R = corrcoef(current_feature,final_labels_common_numerical);
        correlation_coeffs = [correlation_coeffs; R(1,2)];
        plot(current_feature,final_labels_common_numerical,'.')
        w=waitforbuttonpress;
    catch
        correlation_coeffs = [correlation_coeffs; zeros(numel(current_idx),1)];
    end
end

%final_labels_categorical = categorical(final_labels)';
%final_labels_double = double(final_labels_categorical);


%% plot data distribution

clearvars -except path_*

cd(path_extracted)
load('final_data.mat')

% plot number of audio files per dataset

figure(1)
subplot(1,2,1)
piedata_datasets = [];
for i = 1:length(dataset_names)
    piedata_datasets = [piedata_datasets; sum(dataset_ids_common == i)];
end
pie(piedata_datasets,dataset_names)
title(sprintf('distribution of %i audio files',sum(piedata_datasets)))

% plot number of subjects per dataset

subplot(1,2,2)
piedata_subjects = {};
for subject_i = 1:length(final_subjects_common)
    piedata_subjects{subject_i} = strcat(final_datasets_common{subject_i},num2str(final_subjects_common{subject_i}));
end

piedata_unique_subjects = unique(piedata_subjects);

piedata_subjects_per_dataset = [];
for i = 1:length(piedata_unique_subjects)
    for j = 1:length(dataset_names)
        if contains(piedata_unique_subjects{i},dataset_names{j})
            piedata_subjects_per_dataset = [piedata_subjects_per_dataset; j];
        end
    end
end

piedata_subjects_n = [];
for i = 1:length(dataset_names)
    piedata_subjects_n = [piedata_subjects_n; sum(piedata_subjects_per_dataset == i)];
end

pie(piedata_subjects_n,dataset_names)
title(sprintf('distribution of %i subjects',sum(piedata_subjects_n)))

% order


%% classify data and calculate test accuracy on each individual dataset when they are excluded one in turn
clearvars -except path_*

cd(path_extracted)
load('final_data.mat')

% features to include
%features_to_include = {'mfcc', }
%temp = final_features_matrix_common;
%final_features_matrix_common = [];
%final_features_matrix_common = 

path_project = 'C:\Users\lavik\OneDrive\Documents\Machine Learning for Speech\Project';
cd(path_project)

% define parameters for different classifiers
neighbors = 1; % KNN
splits = 100; % DT/ensemble
splits_ensemble = 100;
layer_sizes = [5 5 5]; % NN
% define number of folds
k_folds = 5;

rng(1);

group_order = {'anger', 'happiness', 'sadness'};

% select dataset id for test set
for test_set_id = 1:6
    
    tic

    final_labels_train = final_emotions_common(dataset_ids_common ~= test_set_id)';
    final_features_train = final_features_matrix_common(dataset_ids_common ~= test_set_id,:);
    
    final_labels_test = final_emotions_common(dataset_ids_common == test_set_id)';
    final_features_test = final_features_matrix_common(dataset_ids_common == test_set_id,:);
    
    [classifier_KNN, validation_accuracy_KNN] = trainKNNClassifier(final_features_train,final_labels_train, neighbors, k_folds);
    [classifier_DT, validation_accuracy_DT] = trainDTClassifier(final_features_train,final_labels_train, splits, k_folds);
    [classifier_RF, validation_accuracy_RF] = trainEnsembleClassifier(final_features_train,final_labels_train, splits_ensemble, k_folds);
    [classifier_NN, validation_accuracy_NN] = trainNNClassifier(final_features_train,final_labels_train, layer_sizes, k_folds);
    [classifier_SVM, validation_accuracy_SVM] = trainSVMClassifier(final_features_train,final_labels_train, k_folds);
    
    predictions_KNN = classifier_KNN.predictFcn(final_features_test);
    predictions_DT = classifier_DT.predictFcn(final_features_test);
    predictions_RF = classifier_RF.predictFcn(final_features_test);
    predictions_NN = classifier_NN.predictFcn(final_features_test);
    predictions_SVM = classifier_SVM.predictFcn(final_features_test);
    
    test_accuracy_KNN = strcmp(predictions_KNN,final_labels_test);
    test_accuracy_DT = strcmp(predictions_DT,final_labels_test);
    test_accuracy_RF = strcmp(predictions_RF,final_labels_test);
    test_accuracy_NN = strcmp(predictions_NN,final_labels_test);
    test_accuracy_SVM = strcmp(predictions_SVM,final_labels_test);

    test_accuracy_KNN = sum(test_accuracy_KNN)/numel(test_accuracy_KNN);
    test_accuracy_DT = sum(test_accuracy_DT)/numel(test_accuracy_DT);
    test_accuracy_RF = sum(test_accuracy_RF)/numel(test_accuracy_RF);
    test_accuracy_NN = sum(test_accuracy_NN)/numel(test_accuracy_NN);
    test_accuracy_SVM = sum(test_accuracy_SVM)/numel(test_accuracy_SVM);

    test_accuracies(:,test_set_id) = [test_accuracy_KNN; test_accuracy_DT; test_accuracy_RF; test_accuracy_NN; test_accuracy_SVM];
    
    C_KNN{test_set_id} = confusionmat(final_labels_test, predictions_KNN, 'Order', group_order);
    C_DT{test_set_id} = confusionmat(final_labels_test, predictions_DT, 'Order', group_order);
    C_RF{test_set_id} = confusionmat(final_labels_test, predictions_RF, 'Order', group_order);
    C_NN{test_set_id} = confusionmat(final_labels_test, predictions_NN, 'Order', group_order);
    C_SVM{test_set_id} = confusionmat(final_labels_test, predictions_SVM, 'Order', group_order);

    toc

    test_set_id
end

dataset_names
test_accuracies

mean_per_classifier = mean(test_accuracies,2);
mean_per_dataset = mean(test_accuracies,1);

cd(path_extracted)
save('results','test_accuracies', 'dataset_names', 'mean_per_classifier', 'mean_per_dataset', 'C_KNN', 'C_DT', 'C_RF', 'C_NN', 'C_SVM')

%% Plot confusion matrices

clearvars -except path_*

cd(path_extracted)
load('results.mat')

figure()
for i = 1:6
    subplot(2,3,i)
    confusionchart(C_SVM{i})
    title(dataset_names{i})
end

figure()

test_results_table = table(test_accuracies(:,1),test_accuracies(:,2),test_accuracies(:,3),test_accuracies(:,4),test_accuracies(:,5),test_accuracies(:,6),'VariableNames',dataset_names,'RowNames',{'kNN', 'DT', 'ensemble', 'NN', 'SVM'})
uitable('Data',test_results_table{:,:},'ColumnName',test_results_table.Properties.VariableNames,'RowName',test_results_table.Properties.RowNames,'Units', 'Normalized', 'Position',[0, 0, 1, 1]);

