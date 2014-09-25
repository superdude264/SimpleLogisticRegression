function predY = predictY(X, beta0, beta)
    logProbY = logProbLogReg(1, X, beta0, beta);
    if logProbY > log(.5)  % Use log-probability.
        predY = 1;
    else
        predY = -1;
    end
end