% to be completed
function W = LDA(feature, label)
    label_count = histc(label, unique(label));
    % X - F*n
    X = feature';
    [F,n] = size(X);
    M = zeros(n,n);
    
    start=1;
    for i=1:size(label_count)
        M(start:start+label_count(i)-1,start:start+label_count(i)-1) = ones(label_count(i),label_count(i))/label_count(i);
        start = start + label_count(i);
    end
    
    % Xw - F*n
    Xw = X*(eye(n)-M);
    % K - n*n
    K = Xw'*Xw;
    [V,lambda] = eig(K);
    % decending eigenvalues
    for i = 1:floor(n/2)
       temp_v = V(:,i);
       V(:,i) = V(:,n+1-i);
       V(:,n+1-i) = temp_v;
       
       temp_d = lambda(i,i);
       lambda(i,i) = lambda(n+1-i,n+1-i);
       lambda(n+1-i,n+1-i) = temp_d;
    end
    for i = 1:n-size(label_count)
       lambda(i,i) = lambda(i,i)^-1; 
    end
    U = Xw*V*lambda;
    
    % U - f * n-C
    U = U(:,1:n-size(label_count));
    % Xb_ - n-C * n
    Xb_ = U'*X*M;
    
    K2 = Xb_*Xb_';
    
    Q = PCA(K2, size(label_count)-1);
    W = U*Q;
    
end