function [] = partII()

    % generate the data

    rng(1); 
    r = sqrt(rand(100,1)); 
    t = 2*pi*rand(100,1);  
    data1 = [r.*cos(t), r.*sin(t)]; 

    r2 = sqrt(3*rand(100,1)+1); 
    t2 = 2*pi*rand(100,1);      
    data2 = [r2.*cos(t2), r2.*sin(t2)]; 

    % plot the data

    figure;
    plot(data1(:,1),data1(:,2),'r.','MarkerSize',15)
    hold on
    plot(data2(:,1),data2(:,2),'b.','MarkerSize',15)
    axis equal
    hold on

    % work on class 1
    [a1, R1] = calcRandCentre(data1);

    % work on class 2
    [a2, R2] = calcRandCentre(data2);

    % plot centre and radius for class 1
    plot(a1(1), a1(2), 'rx', 'MarkerSize', 15);
    viscircles(a1', R1, 'Color', 'r', 'LineWidth', 1);
    hold on

    % plot centre and radius for class 2
    plot(a2(1), a2(2), 'bx', 'MarkerSize', 15);
    viscircles(a2', R2, 'Color', 'b', 'LineWidth', 1);

end

function [a, R] = calcRandCentre(data)
    X = data';
    [~,n] = size(X);
    k = zeros(n,1);
    for i=1:n
        k(i) = X(:,i)'*X(:,i);
    end
    H = 2*(X'*X);
    f = -k;
    
    A = zeros(1,n);
    c = 0;
    
    A_e = ones(1,n);
    c_e = 1;
    
    g_l = zeros(n,1);
    g_u = 100000*ones(n,1);
    
    t = quadprog(H, f, A, c, A_e, c_e, g_l, g_u);
    a = X*t
    % calculate R by averaging support vectors
    index = find(t > 1e-9);
    
    R_sum = 0;
    for i = 1:size(index,1)
        R_sum = R_sum + sqrt((X(:,index(i))-a)'*(X(:,index(i))-a));
    end
    R = R_sum/size(index,1)
    
    
end