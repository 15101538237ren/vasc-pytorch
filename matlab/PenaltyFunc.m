clear
close all
% Determine the function for feature distance first
feature_dists = [0:0.02:3];
mu_f = 1;
figure(1)

subplot(1,2,1);
f_penalties = exp(-(feature_dists./mu_f).^4);
plot(feature_dists, f_penalties, '-ro');
text(1.5,0.85,'$penalty(D) = e^{-\big(\frac{D}{1}\big)^4}$','interpreter','latex');
xlabel('Feature Distance');
ylabel('Penalty');

spatial_dists = [0:1:300];

subplot(1,2,2);
mu_s = 80;
s_penalties = 1 - exp(-(spatial_dists./mu_s).^4);
plot(spatial_dists, s_penalties, '-bo');
text(100,0.6,'$penalty(D) = 1 - e^{-\big(\frac{D}{80}\big)^4}$','interpreter','latex');
xlabel('Spatial Distance');
ylabel('Penalty');

fig = gcf;
fig.PaperUnits = 'inches';
fig.PaperSize = [8 4];
fig.PaperPosition = [0 0 8 4];
print(strcat('../figures/drosophila/penalty.pdf'),'-dpdf')
close