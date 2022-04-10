function f = getDataDerivenFeatures(filesFolder, filesName , sampleRate)

f = table;
feature = table;
for i=1:length(filesName)
    i
    %% load data
    load([filesFolder{i} '\' filesName{i}]);
    feature.Variables =  feature_extraction_BLACKSWAN(val, sampleRate);
    f= [f; feature];
    save temp_getDataDerivenFeatures.mat
end

writetable(f,'DataDerivenFeatures.csv')   

