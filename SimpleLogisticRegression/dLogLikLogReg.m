function [dbeta0, dbeta] = dLogLikLogReg(y, X, beta0, beta)
    dbetaHat = zeros(length(X(:,1))+1,1);
    for i = 1:length(X(1,:))
        dbetaHat = dbetaHat + (1 - probLogReg(y(i), X(:, i), beta0, beta))* y(i) * [1; X(:,i)];
    end
    dbeta0 = dbetaHat(1);
    dbeta = dbetaHat(2:length(dbetaHat));
end

