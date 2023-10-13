clc;clear;close all;

path_root = 'C:\Users\wksadmin\Documents\MLFS project\Unpacked';

% resample AESDD to 16 kHz

data_path = 'C:\Users\wksadmin\Documents\MLFS project\Unpacked\AESDD';
subfolders = {'anger', 'disgust', 'fear', 'happiness', 'sadness'};

f_target = 16000;

labels = {};
data = {};

for subfolder = 1:length(subfolders)
    
    
    
    % get emotion
    emotion = subfolders{subfolder};
    
    cd(fullfile(data_path,subfolders{subfolder}))
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
        % if sampling rate is not 16 kHz, resample to 16 kHz
        if FS ~= 16000
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
        label.utterance = utterance
        labels{numel(labels)+1} = label;
        % save waveform to struct
        data{numel(data)+1} = y;
        
    end
    
    
end

%cd(data_paths{1})


