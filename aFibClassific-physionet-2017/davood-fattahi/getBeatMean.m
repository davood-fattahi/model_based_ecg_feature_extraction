function [BeatMean, BeatMedian, tstmp] = getBeatMean(filesFolder, filesName , fs)

rpr = .35; 
for i=1:length(filesName)
    clc; i
    %% load data
    load([filesFolder{i} '\' filesName{i}]);
%     fs = ???
    
    %% preprocessing
    %%% baseline removing
    w1=0.25;
    w2=0.35;
    ppecg=val-(BaseLine1(BaseLine1(val, round(w1*fs), 'md'), round(w2*fs), 'mn'));
    
    %%% R peak detection  
    hr=72/60;
%     peak_detector_params.saturate = 1;
%     peak_detector_params.hist_search_th = 0.9;
%     peak_detector_params.rpeak_search_wlen = 0.3; % MAX detectable HR (in BPM) = 60/rpeak_search_wlen
%     peak_detector_params.filter_type = 'MULT_MATCHED_FILTER';%'BANDPASS_FILTER', 'MATCHED_FILTER', 'MULT_MATCHED_FILTER'
%     peaks = PeakDetectionProbabilistic(ppecg, fs, peak_detector_params);
    peaks = PeakDetection20(val,hr/fs,.3);
    
    %% mean and median beat calculation
    [beatMean, beatMedian, rp, ~] = ecgmean(ppecg, fs, peaks, abs(rpr), 'rw');
    tstmp = ((1:length(beatMean))-rp)/fs; % time stamp
    %%% inversion detection and correction
    if beatMean(rp)<0
        beatMean = - beatMean;
        beatMedian = - beatMedian;
    end
    
    BeatMean{i,:}=beatMean;
    BeatMedian{i,:}=beatMedian;
    save tempGetBeatMean.mat
end

save('BeatMean.mat', 'BeatMean', 'BeatMedian')   
