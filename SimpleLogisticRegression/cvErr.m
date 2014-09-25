function cvErr = cvErr()
    load hw1.mat % X, y, labels, featureNames
    rand('seed', 1); K = 5; N = length(y);
    indices = crossvalind('Kfold', N, K);
    err = zeros(1,length(y));
    for k=1:K
        testX = X(:,indices == k);
        testY = y(indices == k);
        trainX = X(:,indices ~= k);
        trainY = y(indices ~= k);
        [beta0, beta] = fitLogLikLogReg(trainY,trainX);
        for i=1:length(testY)
            predY = predictY(testX(:, i), beta0, beta);
            score = 0;
            if predY ~= testY(i)
                score = 1;
            end
            err(k) = err(k) + score;
        end
    end
    cvErr = sum(err)/length(y);
end