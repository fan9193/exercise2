function [out1,out2] = gibbsbiprobit(k,Y,X,beta_0,B_0)
% k: number of iterations;
% Y: dependent variable;
% X: independent variables;

%% prior: ~N(beta_0,B_0)
% beta_0: mean of prior distribution of beta;
% B_0: variance matrix of prior distribution of beta;

   n = size(Y,1); 
   d = size(X,2); %X:n*d

   
   % store estimates for each iteration
   beta_store = zeros(k,d);
   
   % iteration for posterior
   beta = zeros(d,1);
   temp1=0;
   Z = zeros(n,1);
   betadraw = (X'*X)^(-1)*X'*Y; %initial draw of beta

   for t = 1:k
       for i = 1:n
           a=X(i,:)*betadraw;
           
           % truncated normal
           pd = makedist('normal','mu',a,'sigma',1);
           t1=truncate(pd,0,inf);
           t2 = truncate(pd,-inf,0);
           temp = random(t1)*Y(i)+random(t2)*(1-Y(i));
           temp1 = [temp1,temp];
       end
       Z = (temp1(2:n+1))';
       
       % posterial distribution ~ N(beta,B)
       beta = (inv(B_0) + X'*X)\(inv(B_0)*beta_0 + X'*Z);
       B = inv(inv(B_0)+ X'*X);
       betadraw = (mvnrnd(beta,B))';
       beta_store(t,:) = betadraw;
   end
   out1 = beta_store;
   out2 = B;
end


