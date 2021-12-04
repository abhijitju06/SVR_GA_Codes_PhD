% clc
% clear all

% load('new_bilogical_data_random_order.mat')

% x_v = zeros(1,107);
% u = zeros(1, 27);                     
% dataset_helicopter=zeros(1,241);
% T  = 10;
% Ts = 0.0002;
% Tspan = [0 Ts];
% t = [0:Ts:T]';
% i=1;
% no_epoch = 20000;
% 
% b1=1;
% a1=0.1;
% 
% for epoch =1:no_epoch
%     flag=1;
%     x_v(i, :) = ((b1-a1).*rand(1,107) + a1);
%     [xd,ui] = CCM_three_time_scale_expression_1(t,x_v(i, :));
%     u(i, :)= real(ui(end, :));
%     while(flag)
%         [tsim, xsim] = ode23tb('CCM_three_time_scale_expression_1', Tspan, x_v(i,:));
%         x_v(i+1, :) = real(xsim(end, :));
%         for j = 1:107
%            if (x_v(i+1, j))<0.1
%                x_v(i+1, j) = 0.1;
%            end
%         end
%    
%         for j = 1:107
%             if (x_v(i+1, j))>1.0
%                 x_v(i+1, j) = 1.0;
%             end   
%         end
%         [xd,ui] = CCM_three_time_scale_expression_1(t,x_v(i+1, :));
%         u(i+1, :)= real(ui(end, :));
%         dataset_helicopter(i,:)=[x_v(i, :) u(i, :) x_v(i+1, :)];
%         i=i+1;
%         fprintf('%d SAMPLE GENERATED\n\n',i-1);
%         if(i==(epoch*40)+1)
%             flag=0;
%         end
%     end
% end


% p=1;
% set =1;
% for part = 1:2
% for s = 1:(no_epoch/2)  
%    i = 1;
%    for row=p:p+18
%       train_input(set,i:i+11) = dataset_helicopter(row,1:12);
%       i = i+12;
%    end
%    row = row + 1;
%    train_input(set,i:i+6) = dataset_helicopter(row,1:7);
%    p = p+20;
%    train_output(set,1:5) = dataset_helicopter(row,8:12);
%    
%    i = 1;
%    for row=p:p+18
%       test_input(set,i:i+11) = dataset_helicopter(row,1:12);
%       i = i+12;
%    end
%    row = row + 1;
%    test_input(set,i:i+6) = dataset_helicopter(row,1:7);
%    p = p+20;
%    test_output(set,1:5) = dataset_helicopter(row,8:12);  
%    
%    set = set + 1;
% end
% end
% 
% for row = 1:no_epoch
%    col = 1; 
%    for u=1:7
%       t = 19*12+u;
%       for i=1:20
%          final_test_input(row,col) = test_input(row,t);
%          final_train_input(row,col) = train_input(row,t);
%          t = t - 12;
%          col = col + 1;
%       end     
%    end
%    
%    for y=1:5
%       t = 18*12+u+y;
%       for i=1:19
%          final_test_input(row,col) = test_input(row,t);
%          final_train_input(row,col) = train_input(row,t);
%          t = t - 12;
%          col = col + 1;
%       end    
%    end   
% end
% 
% final_train_data = zeros(10000,241);
% final_test_data = zeros(10000,241);
% 
% final_train_data(:,1:235) = final_train_input;
% final_train_data(:,237:241) = train_output;
% 
% final_test_data(:,1:235) = final_test_input;
% final_test_data(:,237:241) = test_output;

no_epoch = 20000;
p=1;
for set = 1:no_epoch  
   i = 1;
   for row=p:p+18
      train_input(set,i:i+240) = new_bilogical_data_random_order(row,1:241);
      i = i+241;
   end
   row = row + 1;
   train_input(set,i:i+133) = new_bilogical_data_random_order(row,1:134);
   p = p+20;
   train_output(set,1:107) = new_bilogical_data_random_order(row,135:241);
   
   i = 1;
   for row=p:p+18
      test_input(set,i:i+240) = new_bilogical_data_random_order(row,1:241);
      i = i+241;
   end
   row = row + 1;
   test_input(set,i:i+133) = new_bilogical_data_random_order(row,1:134);
   p = p+20;
   test_output(set,1:107) = new_bilogical_data_random_order(row,135:241);   
   
   fprintf('%d SET COMPLETED\n\n',set);
end


for row = 1:no_epoch
   col = 1; 
   for u=1:134
      t = 19*241+u;
      for i=1:20
         final_test_input(row,col) = test_input(row,t);
         final_train_input(row,col) = train_input(row,t);
         t = t - 241;
         col = col + 1;
      end     
   end
   
   for y=1:107
      t = 18*241+u+y;
      for i=1:19
         final_test_input(row,col) = test_input(row,t);
         final_train_input(row,col) = train_input(row,t);
         t = t - 241;
         col = col + 1;
      end    
   end   
   fprintf('%d ROW COMPLETED\n\n',row);
   
end

final_train_data = zeros(20000,4821);
final_test_data = zeros(20000,4821);

final_train_data(:,1:4713) = final_train_input;
final_train_data(:,4715:4821) = train_output;

final_test_data(:,1:4713) = final_test_input;
final_test_data(:,4715:4821) = test_output;


% %%%% WRITE DATA INTO EXCEL FILE
% train_filename = 'Final_biological_train_dataset.xlsx';
% test_filename = 'Final_biological_test_dataset.xlsx';
% 
% xlswrite(train_filename,final_train_data);
% xlswrite(test_filename,final_test_data);

