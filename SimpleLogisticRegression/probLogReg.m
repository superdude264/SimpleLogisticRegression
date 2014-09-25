function prob = probLogReg(y, X, beta0, beta)
    prob = 1 / (1 + exp(-y * (beta0 + dot(beta, X))));
end