function [beta0, beta] = fitLogLikLogReg(trainY, trainX)
    s = 1e-5;
    iter = 2000;
    beta0 = 0;  % No prior knowledge.
    beta = zeros(length(trainX(:,1)),1);
    prevLogLik = logLikLogReg(trainY, trainX, beta0, beta);
    
    for z = 1:iter
        [dbeta0, dbeta] = dLogLikLogReg(trainY, trainX, beta0, beta);
        beta0 = beta0 + s * dbeta0;
        beta = beta + s * dbeta;
        
        % Progress check.
        logLik = logLikLogReg(trainY, trainX, beta0, beta);
        if logLik <= prevLogLik
            disp('Log-likelihood not increasing');
            break; 
        end
        
        prevLogLik = logLik;
    end
end
