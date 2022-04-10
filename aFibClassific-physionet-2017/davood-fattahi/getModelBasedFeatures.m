function f = getModelBasedFeatures(filesFolder, filesName , fs)

soiSt = -.35; 
soiEnd = .15;
f = table;
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
    [beatMean, beatMedian, rp, ~] = ecgmean(ppecg, fs, peaks, abs(soiSt), 'rw');
    tstmp = ((1:length(beatMean))-rp)/fs; % time stamp
    soi= (floor(soiSt*length(beatMean)):floor(soiEnd*length(beatMean)))+rp+1;
    %%% inversion detection and correction
    if beatMean(rp)<0
        beatMean = - beatMean;
        beatMedian = - beatMedian;
    end
%     
%     figure
%     plot(ppecg)
%     hold on
%     plot(find(peaks),ppecg(logical(peaks)),'r*')
%     figure
%     plot(beatMean);
%     hold on
%     plot(beatMedian)
%     
    %% Mean feature extraction
    %%% piecewise polynomials fitting with predefined knots
    knots = floor([-.35 -0.11; -0.11 -0.04; -0.04 0.03; 0.03 .1 ]*length(beatMean))+rp+1;
    [f_pppk2, ~, ~]=ppolyfit(tstmp, beatMean, 2, knots);
    [f_pppk3, ~, ~]=ppolyfit(tstmp, beatMean, 3, knots);
    [f_pppk4, ~, ~]=ppolyfit(tstmp, beatMean, 4, knots);
    [f_pppk5, ~, ~]=ppolyfit(tstmp, beatMean, 5, knots);
    [f_pppk6, ~, ~]=ppolyfit(tstmp, beatMean, 6, knots);
    [f_pppk7, ~, ~]=ppolyfit(tstmp, beatMean, 7, knots);
%     
%     [S, t, SS]=ppolyval(f_pppk5,pct,fs);
%     figure
%     plot(t,S); hold on
%     plot(tstmp(soi),beatMean(soi))
%     plot(tstmp(knots(:)), beatMean(knots(:)), '*');    
    
    %%% piecewise polynomials fitting with auto knots finding
    knots = findknots(beatMean(soi),12)+ soi(1) -1;
    warning off
    [f_ppak2, ~, ~]=ppolyfit(tstmp, beatMean, 2, knots);
    [f_ppak3, ~, ~]=ppolyfit(tstmp, beatMean, 3, knots);
    [f_ppak4, ~, ~]=ppolyfit(tstmp, beatMean, 4, knots);
    [f_ppak5, ~, ~]=ppolyfit(tstmp, beatMean, 5, knots);
    [f_ppak6, ~, ~]=ppolyfit(tstmp, beatMean, 6, knots);
    [f_ppak7, ~, ~]=ppolyfit(tstmp, beatMean, 7, knots);
    warning on
    % Note: the warning about 'Polynomial is not unique; degree >= number of
    % data points' can be ignored, since the polyfit function always returns 
    % the unique solution in which the unnecessary high order coefficients are set to zero. 

% 
%     [S, t, SS]=ppolyval(f_ppak5,pct,fs);
%     figure
%     plot(t,S); hold on
%     plot(tstmp(soi),beatMean(soi))
%     plot(tstmp(knots(:)), beatMean(knots(:)), '*');

    %%% piecewise polynomials fitting with fiducial points knots
    knots = ecgfid_1beat(beatMean,fs,fs/length(beatMean)); knots = knots([1,3],1:4)';
    [f_ppfk2, ~, ~]=ppolyfit(tstmp, beatMean, 2, knots);
    [f_ppfk3, ~, ~]=ppolyfit(tstmp, beatMean, 3, knots);
    [f_ppfk4, ~, ~]=ppolyfit(tstmp, beatMean, 4, knots);
    [f_ppfk5, ~, ~]=ppolyfit(tstmp, beatMean, 5, knots);
    [f_ppfk6, ~, ~]=ppolyfit(tstmp, beatMean, 6, knots);
    [f_ppfk7, ~, ~]=ppolyfit(tstmp, beatMean, 7, knots);

    %%% spline fitting with eqi-spaces knots
    f_spes2_45 = splinefit(tstmp(soi), beatMean(soi), 45, 2, .5);
    f_spes3_30 = splinefit(tstmp(soi), beatMean(soi), 30, 3, .5);
    f_spes4_22 = splinefit(tstmp(soi), beatMean(soi), 22, 4, .5);
    f_spes5_18 = splinefit(tstmp(soi), beatMean(soi), 18, 5, .5);
    
    %%% spline fitting with predefined knots
    knots = floor([ -.35 -.20  -0.15 -.1 -.07 -0.05 -0.03 -0.01 0 0.01 0.03 0.05 0.07 0.1 0.15]*length(beatMean))+rp+1;
    knots(knots<1)=1;
    f_sppk2 = splinefit(tstmp(soi), beatMean(soi), tstmp(knots(:)), 2, 0.5 );
    f_sppk3 = splinefit(tstmp(soi), beatMean(soi), tstmp(knots(:)), 3, 0.5 );
    f_sppk4 = splinefit(tstmp(soi), beatMean(soi), tstmp(knots(:)), 4, 0.5 );
    f_sppk5 = splinefit(tstmp(soi), beatMean(soi), tstmp(knots(:)), 5, 0.5 );
    f_sppk6 = splinefit(tstmp(soi), beatMean(soi), tstmp(knots(:)), 6, 0.5 );
    f_sppk7 = splinefit(tstmp(soi), beatMean(soi), tstmp(knots(:)), 7, 0.5 );
%     
%         
%     figure
%     plot(tstmp(soi),ppval(f_sppk5,tstmp(soi))); 
%     hold on
%     plot(tstmp(soi),beatMean(soi))
%     plot(tstmp(knots(:)), beatMean(knots(:)), '*');

    
    %%% spline fitting with auto knots finding
    knots=findknots(beatMean(soi),12)+ soi(1) -1;
    f_spak2 = splinefit(tstmp(soi), beatMean(soi), tstmp(knots(:)), 2, .5 );
    f_spak3 = splinefit(tstmp(soi), beatMean(soi), tstmp(knots(:)), 3, .5 );
    f_spak4 = splinefit(tstmp(soi), beatMean(soi), tstmp(knots(:)), 4, .5 );
    f_spak5 = splinefit(tstmp(soi), beatMean(soi), tstmp(knots(:)), 5, .5 );
    f_spak6 = splinefit(tstmp(soi), beatMean(soi), tstmp(knots(:)), 6, .5 );
    f_spak7 = splinefit(tstmp(soi), beatMean(soi), tstmp(knots(:)), 7, .5 );
       


    %%% spline fitting with auto fiducial points knots
    knots = ecgfid_1beat(beatMean,fs,fs/length(beatMean)); knots = knots(:,1:4)';
    knots=sort(unique(knots(:))); soi_temp=(knots(1):knots(end));
    f_spfk2 = splinefit(tstmp(soi_temp), beatMean(soi_temp), tstmp(knots(:)), 2, .5 );
    f_spfk3 = splinefit(tstmp(soi_temp), beatMean(soi_temp), tstmp(knots(:)), 3, .5 );
    f_spfk4 = splinefit(tstmp(soi_temp), beatMean(soi_temp), tstmp(knots(:)), 4, .5 );
    f_spfk5 = splinefit(tstmp(soi_temp), beatMean(soi_temp), tstmp(knots(:)), 5, .5 );
    f_spfk6 = splinefit(tstmp(soi_temp), beatMean(soi_temp), tstmp(knots(:)), 6, .5 );
    f_spfk7 = splinefit(tstmp(soi_temp), beatMean(soi_temp), tstmp(knots(:)), 7, .5 );
%     
%     figure
%     plot(tstmp(soi),ppval(f_spfk2,tstmp(soi))); 
%     hold on
%     plot(tstmp(soi),beatMean(soi))
%     plot(tstmp(knots(:)), beatMean(knots(:)), '*');

       
    %% sum of Gaussians fitting without constrains
    knots=findknots(beatMean(soi), 10) + soi(1) -1; knots=knots(2:end-1);
    a0=beatMean(knots); b0=0.1*ones(size(a0)); c0=tstmp(knots);
    p0{1}=[a0(:) b0(:) c0(:)]';
    
    knots=findknots(beatMean(soi), 10, 'Value') + soi(1) -1; knots=knots(2:end-1);
    a0=beatMean(knots); b0=0.1*ones(size(a0)); c0=tstmp(knots);
    p0{2}=[a0(:) b0(:) c0(:)]';
    
    knots = ecgfid_1beat(beatMean,fs,fs/length(beatMean)); knots = knots([2,3],1:4)'; knots = sort(knots(:));
    a0=beatMean(knots); b0=0.05*ones(size(a0)); c0=tstmp(knots);
    p0{3}=[a0(:) b0(:) c0(:)]';
    
    alb = -inf*ones(size(a0)); blb = zeros(size(b0)); clb = soiSt*ones(size(c0));
    lb=[alb(:) blb(:) clb(:)]';
    
    aub = inf*ones(size(a0)); bub = 0.1*ones(size(b0)); cub = soiEnd*ones(size(c0));
    ub = [aub(:) bub(:) cub(:)]';
    
    options = struct('SpecifyObjectiveGradient',true);%, 'FunctionTolerance', 1e-8, 'Display','final-detailed' ,'MaxFunctionEvaluations',10000, 'MaxIterations', 10000, 'OptimalityTolerance', 1e-12, 'StepTolerance', 1e-12);
    [f_g{1},resnorm(1),~,~,~] = GausFit(tstmp(soi), beatMean(soi), p0{1}, lb, ub, options);
    [f_g{2},resnorm(2),~,~,~] = GausFit(tstmp(soi), beatMean(soi), p0{2}, lb, ub, options);
    [f_g{3},resnorm(3),~,~,~] = GausFit(tstmp(soi), beatMean(soi), p0{3}, lb, ub, options);
    [~, I] = min(resnorm(:));
    f_gauss = f_g{I};
    
%     
%     figure; plot(tstmp(soi), GausVal(tstmp(soi), f_gauss));
%     hold on;
%     plot(tstmp(soi),beatMean(soi))
%     plot(tstmp(knots(:)), beatMean(knots(:)), '*');    
    %%
    fname = ["f_meanBeat_pppk2" , "f_meanBeat_pppk3" , "f_meanBeat_pppk4" , "f_meanBeat_pppk5" , "f_meanBeat_pppk6" , "f_meanBeat_pppk7", ...
        "f_meanBeat_ppak2" , "f_meanBeat_ppak3" , "f_meanBeat_ppak4" , "f_meanBeat_ppak5" , "f_meanBeat_ppak6" , "f_meanBeat_ppak7", ...
        "f_meanBeat_ppfk2" , "f_meanBeat_ppfk3" , "f_meanBeat_ppfk4" , "f_meanBeat_ppfk5" , "f_meanBeat_ppfk6" , "f_meanBeat_ppfk7", ...
        "f_meanBeat_spes2_45" , "f_meanBeat_spes3_30" , "f_meanBeat_spes4_22" , "f_meanBeat_spes5_18" , ...
        "f_meanBeat_sppk2" , "f_meanBeat_sppk3" , "f_meanBeat_sppk4" , "f_meanBeat_sppk5" , "f_meanBeat_sppk6" , "f_meanBeat_sppk7", ...
        "f_meanBeat_spak2" , "f_meanBeat_spak3" , "f_meanBeat_spak4" , "f_meanBeat_spak5" , "f_meanBeat_spak6" , "f_meanBeat_spak7", ...
        "f_meanBeat_spfk2" , "f_meanBeat_spfk3" , "f_meanBeat_spfk4" , "f_meanBeat_spfk5" , "f_meanBeat_spfk6" , "f_meanBeat_spfk7", ...
        "f_meanBeat_gauss"];
    
    
    
    fmean = table(reshape(cell2mat(f_pppk2),1,[]),   reshape(cell2mat(f_pppk3),1,[]),   reshape(cell2mat(f_pppk4),1,[]),   reshape(cell2mat(f_pppk5),1,[]),   reshape(cell2mat(f_pppk6),1,[]),   reshape(cell2mat(f_pppk7),1,[]),   ...
        reshape(cell2mat(f_ppak2),1,[]),   reshape(cell2mat(f_ppak3),1,[]),   reshape(cell2mat(f_ppak4),1,[]),   reshape(cell2mat(f_ppak5),1,[]),   reshape(cell2mat(f_ppak6),1,[]),   reshape(cell2mat(f_ppak7),1,[]),   ...
        reshape(cell2mat(f_ppfk2),1,[]),   reshape(cell2mat(f_ppfk3),1,[]),   reshape(cell2mat(f_ppfk4),1,[]),   reshape(cell2mat(f_ppfk5),1,[]),   reshape(cell2mat(f_ppfk6),1,[]),   reshape(cell2mat(f_ppfk7),1,[]),   ...
        f_spes2_45.coefs(:)',   f_spes3_30.coefs(:)',   f_spes4_22.coefs(:)',   f_spes5_18.coefs(:)',   ...
        f_sppk2.coefs(:)',   f_sppk3.coefs(:)',   f_sppk4.coefs(:)',   f_sppk5.coefs(:)',   f_sppk6.coefs(:)',   f_sppk7.coefs(:)',   ...
        f_spak2.coefs(:)',   f_spak3.coefs(:)',   f_spak4.coefs(:)',   f_spak5.coefs(:)',   f_spak6.coefs(:)',   f_spak7.coefs(:)',   ...
        f_spfk2.coefs(:)',   f_spfk3.coefs(:)',   f_spfk4.coefs(:)',   f_spfk5.coefs(:)',   f_spfk6.coefs(:)',   f_spfk7.coefs(:)',   ...
        f_gauss(:)', 'VariableNames', fname);
    
    %% Median feature extraction
    soi = find(tstmp>=soiSt & tstmp<=soiEnd); % segment of interest
    
    %%% piecewise polynomials fitting with predefined knots
    knots = floor([-.35 -0.11; -0.11 -0.04; -0.04 0.03; 0.03 .1 ]*length(beatMedian))+rp+1;
    [f_pppk2, ~, ~]=ppolyfit(tstmp, beatMedian, 2, knots);
    [f_pppk3, ~, ~]=ppolyfit(tstmp, beatMedian, 3, knots);
    [f_pppk4, ~, ~]=ppolyfit(tstmp, beatMedian, 4, knots);
    [f_pppk5, ~, ~]=ppolyfit(tstmp, beatMedian, 5, knots);
    [f_pppk6, ~, ~]=ppolyfit(tstmp, beatMedian, 6, knots);
    [f_pppk7, ~, ~]=ppolyfit(tstmp, beatMedian, 7, knots);
    
    %%% piecewise polynomials fitting with auto knots finding
    knots=(findknots(beatMedian(soi),12))+ soi(1) -1;
    warning off
    [f_ppak2, ~, ~]=ppolyfit(tstmp, beatMedian, 2, knots);
    [f_ppak3, ~, ~]=ppolyfit(tstmp, beatMedian, 3, knots);
    [f_ppak4, ~, ~]=ppolyfit(tstmp, beatMedian, 4, knots);
    [f_ppak5, ~, ~]=ppolyfit(tstmp, beatMedian, 5, knots);
    [f_ppak6, ~, ~]=ppolyfit(tstmp, beatMedian, 6, knots);
    [f_ppak7, ~, ~]=ppolyfit(tstmp, beatMedian, 7, knots);
    warning on
    % Note: the warning about 'Polynomial is not unique; degree >= number of
    % data points' can be ignored, since the polyfit function always returns 
    % the unique solution in which the unnecessary high order coefficients are set to zero. 

    
    %%% piecewise polynomials fitting with fiducial points knots
    knots = ecgfid_1beat(beatMedian,fs,fs/length(beatMedian)); knots = knots([1,3],1:4)';
    [f_ppfk2, ~, ~]=ppolyfit(tstmp, beatMedian, 2, knots);
    [f_ppfk3, ~, ~]=ppolyfit(tstmp, beatMedian, 3, knots);
    [f_ppfk4, ~, ~]=ppolyfit(tstmp, beatMedian, 4, knots);
    [f_ppfk5, ~, ~]=ppolyfit(tstmp, beatMedian, 5, knots);
    [f_ppfk6, ~, ~]=ppolyfit(tstmp, beatMedian, 6, knots);
    [f_ppfk7, ~, ~]=ppolyfit(tstmp, beatMedian, 7, knots);

    %%% spline fitting with eqi-spaces knots
    f_spes2_45 = splinefit(tstmp(soi), beatMedian(soi), 45, 2, .5);
    f_spes3_30 = splinefit(tstmp(soi), beatMedian(soi), 30, 3, .5);
    f_spes4_22 = splinefit(tstmp(soi), beatMedian(soi), 22, 4, .5);
    f_spes5_18 = splinefit(tstmp(soi), beatMedian(soi), 18, 5, .5);
    
    %%% spline fitting with predefined knots
    knots = floor([ -.35 -.20  -0.15 -.1 -.07 -0.05 -0.03 -0.01 0 0.01 0.03 0.05 0.07 0.1 0.15]*length(beatMedian))+rp+1;
    soi_temp=(knots(1):knots(end));
    f_sppk2 = splinefit(tstmp(soi_temp), beatMedian(soi_temp), tstmp(knots(:)), 2, 0.5 );
    f_sppk3 = splinefit(tstmp(soi_temp), beatMedian(soi_temp), tstmp(knots(:)), 3, 0.5 );
    f_sppk4 = splinefit(tstmp(soi_temp), beatMedian(soi_temp), tstmp(knots(:)), 4, 0.5 );
    f_sppk5 = splinefit(tstmp(soi_temp), beatMedian(soi_temp), tstmp(knots(:)), 5, 0.5 );
    f_sppk6 = splinefit(tstmp(soi_temp), beatMedian(soi_temp), tstmp(knots(:)), 6, 0.5 );
    f_sppk7 = splinefit(tstmp(soi_temp), beatMedian(soi_temp), tstmp(knots(:)), 7, 0.5 );
    
    %%% spline fitting with auto knots finding
    knots=(findknots(beatMedian(soi),12))+ soi(1) -1;
    f_spak2 = splinefit(tstmp(soi), beatMedian(soi), tstmp(knots(:)), 2, .5 );
    f_spak3 = splinefit(tstmp(soi), beatMedian(soi), tstmp(knots(:)), 3, .5 );
    f_spak4 = splinefit(tstmp(soi), beatMedian(soi), tstmp(knots(:)), 4, .5 );
    f_spak5 = splinefit(tstmp(soi), beatMedian(soi), tstmp(knots(:)), 5, .5 );
    f_spak6 = splinefit(tstmp(soi), beatMedian(soi), tstmp(knots(:)), 6, .5 );
    f_spak7 = splinefit(tstmp(soi), beatMedian(soi), tstmp(knots(:)), 7, .5 );
        
    %%% spline fitting with auto fiducial points knots
    knots = ecgfid_1beat(beatMedian,fs,fs/length(beatMedian)); knots = knots(:,1:4)';
    knots=sort(unique(knots));     soi_temp=(knots(1):knots(end));
    f_spfk2 = splinefit(tstmp(soi_temp), beatMedian(soi_temp), tstmp(knots(:)), 2, .5 );
    f_spfk3 = splinefit(tstmp(soi_temp), beatMedian(soi_temp), tstmp(knots(:)), 3, .5 );
    f_spfk4 = splinefit(tstmp(soi_temp), beatMedian(soi_temp), tstmp(knots(:)), 4, .5 );
    f_spfk5 = splinefit(tstmp(soi_temp), beatMedian(soi_temp), tstmp(knots(:)), 5, .5 );
    f_spfk6 = splinefit(tstmp(soi_temp), beatMedian(soi_temp), tstmp(knots(:)), 6, .5 );
    f_spfk7 = splinefit(tstmp(soi_temp), beatMedian(soi_temp), tstmp(knots(:)), 7, .5 );
       
    %% sum of Gaussians fitting without constrains
    knots=findknots(beatMedian(soi), 10) + soi(1) -1; knots=knots(2:end-1);
    a0=beatMedian(knots); b0=0.1*ones(size(a0)); c0=tstmp(knots);
    p0{1}=[a0(:) b0(:) c0(:)]';
    
    knots=findknots(beatMedian(soi), 10, 'Value') + soi(1) -1; knots=knots(2:end-1);
    a0=beatMedian(knots); b0=0.1*ones(size(a0)); c0=tstmp(knots);
    p0{2}=[a0(:) b0(:) c0(:)]';
    
    knots = ecgfid_1beat(beatMedian,fs,fs/length(beatMedian)); knots = knots([2,3],1:4)'; knots = sort(knots(:));
    a0=beatMedian(knots); b0=0.05*ones(size(a0)); c0=tstmp(knots);
    p0{3}=[a0(:) b0(:) c0(:)]';
    
    alb = -inf*ones(size(a0)); blb = zeros(size(b0)); clb = soiSt*ones(size(c0));
    lb=[alb(:) blb(:) clb(:)]';
    
    aub = inf*ones(size(a0)); bub = 0.1*ones(size(b0)); cub = soiEnd*ones(size(c0));
    ub = [aub(:) bub(:) cub(:)]';
    
    options = struct('SpecifyObjectiveGradient',true);%, 'FunctionTolerance', 1e-8, 'Display','final-detailed' ,'MaxFunctionEvaluations',10000, 'MaxIterations', 10000, 'OptimalityTolerance', 1e-12, 'StepTolerance', 1e-12);
    [f_g{1},resnorm(1),~,~,~] = GausFit(tstmp(soi), beatMedian(soi), p0{1}, lb, ub, options);
    [f_g{2},resnorm(2),~,~,~] = GausFit(tstmp(soi), beatMedian(soi), p0{2}, lb, ub, options);
    [f_g{3},resnorm(3),~,~,~] = GausFit(tstmp(soi), beatMedian(soi), p0{3}, lb, ub, options);
    [~, I] = min(resnorm(:));
    f_gauss = f_g{I};
    
    
%     close all
%     figure; plot(tstmp(soi), GausVal(tstmp(soi), f_gauss));
%     hold on;
%     plot(tstmp(soi),beatMedian(soi))
%     plot(tstmp(knots(:)), beatMedian(knots(:)), '*');   
    
    %%
    fname = ["f_medBeat_pppk2" , "f_medBeat_pppk3" , "f_medBeat_pppk4" , "f_medBeat_pppk5" , "f_medBeat_pppk6" , "f_medBeat_pppk7", ...
        "f_medBeat_ppak2" , "f_medBeat_ppak3" , "f_medBeat_ppak4" , "f_medBeat_ppak5" , "f_medBeat_ppak6" , "f_medBeat_ppak7", ...
        "f_medBeat_ppfk2" , "f_medBeat_ppfk3" , "f_medBeat_ppfk4" , "f_medBeat_ppfk5" , "f_medBeat_ppfk6" , "f_medBeat_ppfk7", ...
        "f_medBeat_spes2_45" , "f_medBeat_spes3_30" , "f_medBeat_spes4_22" , "f_medBeat_spes5_18" , ...
        "f_medBeat_sppk2" , "f_medBeat_sppk3" , "f_medBeat_sppk4" , "f_medBeat_sppk5" , "f_medBeat_sppk6" , "f_medBeat_sppk7", ...
        "f_medBeat_spak2" , "f_medBeat_spak3" , "f_medBeat_spak4" , "f_medBeat_spak5" , "f_medBeat_spak6" , "f_medBeat_spak7", ...
        "f_medBeat_spfk2" , "f_medBeat_spfk3" , "f_medBeat_spfk4" , "f_medBeat_spfk5" , "f_medBeat_spfk6" , "f_medBeat_spfk7", ...
        "f_medBeat_gauss"];
    
    
    
    fmed = table(reshape(cell2mat(f_pppk2),1,[]),   reshape(cell2mat(f_pppk3),1,[]),   reshape(cell2mat(f_pppk4),1,[]),   reshape(cell2mat(f_pppk5),1,[]),   reshape(cell2mat(f_pppk6),1,[]),   reshape(cell2mat(f_pppk7),1,[]),   ...
        reshape(cell2mat(f_ppak2),1,[]),   reshape(cell2mat(f_ppak3),1,[]),   reshape(cell2mat(f_ppak4),1,[]),   reshape(cell2mat(f_ppak5),1,[]),   reshape(cell2mat(f_ppak6),1,[]),   reshape(cell2mat(f_ppak7),1,[]),   ...
        reshape(cell2mat(f_ppfk2),1,[]),   reshape(cell2mat(f_ppfk3),1,[]),   reshape(cell2mat(f_ppfk4),1,[]),   reshape(cell2mat(f_ppfk5),1,[]),   reshape(cell2mat(f_ppfk6),1,[]),   reshape(cell2mat(f_ppfk7),1,[]),   ...
        f_spes2_45.coefs(:)',   f_spes3_30.coefs(:)',   f_spes4_22.coefs(:)',   f_spes5_18.coefs(:)',   ...
        f_sppk2.coefs(:)',   f_sppk3.coefs(:)',   f_sppk4.coefs(:)',   f_sppk5.coefs(:)',   f_sppk6.coefs(:)',   f_sppk7.coefs(:)',   ...
        f_spak2.coefs(:)',   f_spak3.coefs(:)',   f_spak4.coefs(:)',   f_spak5.coefs(:)',   f_spak6.coefs(:)',   f_spak7.coefs(:)',   ...
        f_spfk2.coefs(:)',   f_spfk3.coefs(:)',   f_spfk4.coefs(:)',   f_spfk5.coefs(:)',   f_spfk6.coefs(:)',   f_spfk7.coefs(:)',   ...
        f_gauss(:)', 'VariableNames', fname);
    
    f = [f; [fmean fmed]];
    save temp_getModelBasedFeatures.mat
end

writetable(f,'ModelBasedFeatures.csv')   

