function logP = logProbLogReg(y, X, beta0, beta)
    logP = -log(1 + exp(-y * (beta0 + dot(beta, X))));
end
