clc
clear
close all

files = dir('..\training2017\*.mat');
f_MB = getModelBasedFeatures({files.folder}, {files.name}, 300);
f_DD = getDataDerivenFeatures({files.folder}, {files.name}, 300);
[~, ~, ~] = getBeatMean({files.folder}, {files.name} , 300);
saveEcgBeats({files.folder}, {files.name} , 300, 'segmentedBeats/');

