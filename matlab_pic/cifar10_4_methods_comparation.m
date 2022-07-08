set(gcf,'unit','centimeters','position',[10 5 14 8]);
set(gcf,'Color',[0.9 0.9 0.9]);
marksize = 4;
linewidth = 0.7;


CR = 100;
y_low = 0;
y_up = 1.0;
% dataset number 
% 1 --> cifar
% 2 --> cifar100
% 3 --> gtsrb
dataset_num = 3;

if dataset_num == 1
    source_data = importdata('exp1_cifar.csv');
    dataset_name = 'cifar';
    filename='x4_lines_cifar10.pdf';
elseif dataset_num == 2
    source_data = importdata('exp1_cifar100.csv');
    dataset_name = 'cifar100';
    filename='x4_lines_cifar100.pdf';
elseif dataset_num ==3
    source_data = importdata('exp1_gtsrb.csv');
    dataset_name = 'gtsrb';
    filename='x4_lines_gtsrb.pdf';
end



pic_data = source_data.data;

x = 1:1:CR;
our_train_accuracy = pic_data(1:CR, 1);
our_test_accuracy = pic_data(1:CR, 2);
traditionalFL_train_accuracy = pic_data(1:CR, 3);
traditionalFL_test_accuracy = pic_data(1:CR, 4);
FL_mobile_train_accuracy = pic_data(1:CR, 5);
FL_mobile_test_accuracy = pic_data(1:CR, 6);
centralized_train_accuracy = pic_data(1:CR, 7);
centralized_test_accuracy = pic_data(1:CR, 8);

plot(x, our_train_accuracy, '-ok','LineWidth',linewidth,'MarkerSize',marksize);hold on;
plot(x, traditionalFL_train_accuracy, '-*r','LineWidth',linewidth,'MarkerSize',marksize);hold on;
plot(x, FL_mobile_train_accuracy, '-xb','LineWidth',linewidth,'MarkerSize',marksize);hold on;
plot(x, centralized_train_accuracy, '-sg','LineWidth',linewidth,'MarkerSize',marksize);hold on;

legend('PractFL','Traditional FL','Mobile FL','Centralized Learning','Location','Southeast','Orientation','vertical');



axis([1 CR y_low y_up]);
figure_FontSize=10;
set(findobj('FontSize',10),'FontSize',figure_FontSize);
figure_FontName='Arial';
set(findobj('FontName', 'Helvetica'), 'FontName', figure_FontName);
xlabel('Communication round');
ylabel('Accuracy');
%set(gca,'Position',[.13 .17 .80 .74]);
set(findobj(get(gca,'Children'),'LineWidth',0.5),'LineWidth',0.5);
set(gcf,'Units','Inches');
pos = get(gcf,'Position');
set(gcf,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)]);
%filename = './fedcav_mnist_0.3.pdf'; % 设定导出文件名
print(gcf,filename,'-dpdf','-r0')