% to be completed
function U_reduce = PCA(features, Size)
    FT = features';
    % data centering
    [~,N] = size(FT);
    X = FT*(eye(N)-1/N*ones(N,N));
    % dot product matrix
    K = X'*X;
    K = 1/2*(K+K');      % make K symmetric
    % eigen analysis
    [V, lambda] = eig(K);
    for i = 1:floor(N/2)
       temp_v = V(:,i);
       V(:,i) = V(:,N+1-i);
       V(:,N+1-i) = temp_v;
       
       temp_d = lambda(i,i);
       lambda(i,i) = lambda(N+1-i,N+1-i);
       lambda(N+1-i,N+1-i) = temp_d;
    end
    for i = 1:N-1
       lambda(i,i) = lambda(i,i)^-0.5; 
    end
    U = X*V*lambda;
    % keep components
    U_reduce = U(:,1:Size);
end