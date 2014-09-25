function cvErr = p11_thru_15()
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
            predY = predictY(testX(:, i), beta0, beta); % Filled in.
            score = 0;
            if predY ~= testY(i)
                score = 1;
            end
            err(k) = err(k) + score;
        end
    end
    cvErr = sum(err)/length(y);
end

function prob = probLogReg(y, X, beta0, beta)
    prob = 1 / (1 + exp(-y * (beta0 + dot(beta, X))));
end

function logP = logProbLogReg(y, X, beta0, beta)
    logP = -log(1 + exp(-y * (beta0 + dot(beta, X))));
end

function predY = predictY(X, beta0, beta)
    logProbY = logProbLogReg(1, X, beta0, beta);
    if logProbY > .5
        predY = 1;
    else
        predY = -1;
    end
end

function val = LogLikLogReg(y, X, beta0, beta)
    val = 0;
    for i=1:length(y)
        val = val + logProbLogReg(y(i), X(:, i), beta0, beta);
    end
end

function [dbeta0, dbeta] = dLogLikLogReg(y, X, beta0, beta)  
    dbeta0 = 0;
    for i = 1:length(X(1,:))
        dbeta0 = dbeta0 + (1 - probLogReg(y(i), X(:, i), beta0, beta))* y(i);
    end
    
    for p = 1:length(beta)
        val = 0;
        for i = 1:length(X(1,:))
            val = val + (1 - probLogReg(y(i), X(:, i), beta0, beta))* y(i) * X(p, i);
        end
        dbeta(p, 1) = val;
    end
end

function [beta0,beta] = fitLogLikLogReg(trainY, trainX)
    s = 1e-5;
    iter = 2000;
    beta0 = 0;
    beta = zeros(length(trainX(:,1)),1);  % Start @ 0, should be random?
    
    for z = 1:iter
        [dbeta0, dbeta] = dLogLikLogReg(trainY, trainX, beta0, beta);
        beta0 = beta0 + s * dbeta0;
        beta = beta + s * dbeta;
        display(z)
    end
end