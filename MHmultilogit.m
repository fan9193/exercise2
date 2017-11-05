function [out1,out2] = MHmultilogit
%% Metropolis-Hastings Algorithm for multinomial Logit model
% I assume number of options are the same for all customers, and there's no outside option 


% k: number of iterations;
% Y: dependent variable; nJ*1
% X: independent variables; nJ*d
% J: number of choices

% here I use a normal prior
% beta_0: mean of prior distribution of beta;
% B_0: variance matrix of prior distribution of beta;

global k X Y J beta_0 B_0;
n = length(Y)/J;
d = size(X,2); % number of parameters

      
%% calculate prior density given beta value
    function out = priorden(beta) %scalar given beta.
        out = mvnpdf(beta,beta_0,B_0);
    end

%% calculate likelihood function f(y|beta)
% store variable
    function out = likelihood_multilogit(beta)
        choiceprob = zeros(n,J); % choice probability
        denorm = zeros(1,n); %denominator
        P = zeros(1,n);
      for i=1:n
          denorm(i) = sum(exp(X((i-1)*J+1:i*J,:)*beta)); % denorm is 1*n
          for j =1:J
              if (Y((i-1)*J+j)==1)
                 choiceprob(i,j) = exp(X((i-1)*J+j,:)*beta)/denorm(i);
              else
                  choiceprob(i,j) = 1/denorm(i);
              end
          end
          P(i)= prod(choiceprob(i,:).^double(Y((i-1)*J+1:i*J)'));
      end
      out = sum(log(P)); % use log because multiply p for 1000 times would be very very small
    end
      
%% posterial density of beta
    function out = posterialden(beta) 
        out = likelihood_multilogit(beta)+ log(priorden(beta)); % scalar
    end

%% acceptance rate
    function out = accept(beta,beta_new) % scalar
        out = min(1,exp(posterialden(beta_new)-posterialden(beta)));
    end

% here I use random walk as q(beta,beta')
%% MH algorithm
   % store estimates for each iteration
   beta_store = zeros(k,d);
   beta = zeros(1,d);%initial value
   beta_new = zeros(1,d);% define beta_new
   
   for t = 1:k
       %note parameters be updated one by one.
       for m = 1:d %parameters
         beta_new(m) = beta(m) + normrnd(0,2);
         r=rand();
         a = accept(beta',beta_new');
         if r < a
           beta(m) = beta_new(m);
         else
           beta_new(m) = beta(m);
         end  
       end
       beta_store(t,:) = beta_new;
   end
   
   B = cov(beta_store);
   
   out1 = beta_store;
   out2 = B;
end


