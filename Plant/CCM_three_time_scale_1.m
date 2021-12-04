clc
clear
close all

warning('off','all');


% CCM_three_time_scale_initial_value();

%%%%% METABOLITES INITIAL CONCENTRATION in nM %%%%%
%%% m39 is not included intentionally

m1	  =	0.7333	;	% glc_1P arbitary
m2	  =	0.7152	;	% glc_6P
m3	  =	0.6143	;	% glc arbitary
m4	  =	0.922	;	% f_6P
m5	  =	0.6691	;	% f_16_BP 
m6	  =	0.1878	;	% f_26_BP arbitary
m7	  =	0.3506	;	% dhap arbitary
m8	  =	0.5922	;	% ga_3P
m9	  =	0.9618	;	% BPG_13
m10	  =	0.9684	;	% PG_3
m11	  =	0.2419	;	% PG_2
m12	  =	0.9735	;	% PEP
m13	  =	0.9615	;	% PYR arbitary
m14	  =	0.5368	;	% LACTATE arbitary (only produced)
m15	  =	0.8203	;	% Ace_coa % – 0.006 mM
m16	  =	0.4277	;	% OAA%0.006 mM [2,11]
m17	  =	0.4796	;	% Citrate arbitary
m18	  =	0.9242	;	% RIsocitrate arbitary
m19	  =	0.813	;	% Ralpha_KG
m20	  =	0.9635	;	% Succ_coa % – 0.91 mM [2,11]
m21	  =	0.6902	;	% succinate arbitary
m22	  =	0.2321	;	% fumarate arbitary
m23	  =	0.8642	;	% malate
m24	  =	0.9406	;	% gl_15_l6P arbitary
m25	  =	0.7109	;	% gl_6P
m26	  =	0.782	;	% rbls_5P
m27	  =	0.7688	;	% rbs_5P
m28	  =	0.453	;	% xyl_5P
m29	  =	0.6899	;	% sdhl_7P
m30	  =	0.2541	;	% eryt_4P
m31	  =	0.7354	;	% ATP
m32	  =	0.5286	;	% ADP arbitary
m33	  =	0.3492	;	% NADH
m34	  =	0.1416	;	% NAD
m35	  =	0.1874	;	% COA % – 0.79 mM [2,11]
m36	  =	0.8411	;	% CO2 arbitary (only produced)
m37	  =	0.7253	;	% FFA arbitary
m38	  =	0.3854	;	% FAD arbitary
m40	  =	0.9552	;	% FADH
m41	  =	0.131	;	% COA_SH
m42	  =	0.4949	;	% NADPH arbitary

%%%%% SIGNALLING MOLECULE INITIAL CONCENTRATION s in nM %%%%%

s1	 =	0.5316	;	%	STAT3	
s2	 =	0.8212	;	%	NFkB	
s3	 =	0.3051	;	%	ERK	
s4	 =	0.5483	;	%	P13K	
s5	 =	0.9108	;	%	AKT	
s6	 =	0.6172	;	%	mTOR	
s7	 =	0.8607	;	%	MNK	
s8	 =	0.3648	;	%	 HIF_1	
s9	 =	0.6274	;	%	elf4E	
s10	 =	0.3221	;	%	Ras	
s11	 =	0.6998	;	%	TRAF6	
s12	 =	0.1751	;	%	Raf_1	
s13	 =	0.6634	;	%	ROS	
s14	 =	0.6949	;	%	Rheb	
s15	 =	0.7568	;	%	PTEN	
s16	 =	0.5017	;	%	VHL	
s17	 =	0.5841	;	%	 PHD	
s18	 =	0.7921	;	%	AMP	
s19	 =	0.6233	;	%	PDK1	
s20	 =	0.9355	;	%	PKCs	(only produced)
s21	 =	0.6221	;	%	BCL_X	(only produced)
s22	 =	0.1153	;	%	FASL	(only produced)
s23	 =	0.2088	;	%	MDM2	
s24	 =	0.8764	;	%	p53	
s25	 =	0.5359	;	%	MDMX	
s26	 =	0.8604	;	%	MYC	
s27	 =	0.2885	;	%	TIGAR	(only produced)
s28	 =	0.5971	;	%	MTORC2	
s29	 =	0.6669	;	%	MTORC1	


%%%%% ENZYME/GENE INITIAL CONCENTRATION g in nM (Assumed all) %%%%%

g1	 =	0.968	;	%Glut 1
g2	 =	0.4129	;	%Phosphoglucomutase_1
g3	 =	0.735	;	%Hexokinase
g4	 =	0.6275	;	%Glucose-6-phosphatase
g5	 =	0.3359	;	%Phosphoglucoisomerase
g6	 =	0.74	;	%Phosphofructokinase_1
g7	 =	0.7794	;	%Fructose-1,6-bisphosphatase
g8	 =	0.3185	;	%Phosphofructokinase_2
g9	 =	0.4982	;	%Fructose-2,6-bisphosphatase
g10	 =	0.719	;	%Aldolase arbitrary 0.809 actual
g11	 =	0.4233	;	%Triose_phosphate_isomerase
g12	 =	0.7627	;	%Glyceraldehyde-3-phosphate_dehydrogenase
g13	 =	0.4552	;	%Phosphoglycerate_kinase
g14	 =	0.7151	;	%Phosphoglycerate_mutase
g15	 =	0.7336	;	%Enolase
g16	 =	0.6981	;	%Pyruvate_kinase
g17	 =	0.1176	;	%Lactate_dehydrogenase
g18	 =	0.3978	;	%Pyruvate_dehydrogenase             %%% Perturbation point for warburg effect... but basal production should not be changed I think
g19	 =	0.4819	;	%Acyl-CoA_synthetase                %%% Perturbation point for warburg effect... but basal production should not be changed I think
g20	 =	0.3432	;	%Fatty_acid_synthase                %%% Perturbation point for warburg effect... but basal production should not be changed I think
g21	 =	0.8773	;	%Pyruvate_carboxylase               %%% Perturbation point for warburg effect... but basal production should not be changed I think 
g22	 =	0.8395	;	%Phosphoenolpyruvate_carboxykinase_1 %%% Perturbation point for warburg effect... but basal production should not be changed I think
g23	 =	0.8869	;	%Citrate_synthase 
g24	 =	0.899	;	%Aconitase_1 arbitrary
g25	 =	0.8521	;	%Isocitrate_dehydrogenase
g26	 =	0.7922	;	%alpha-Ketoglutarate_dehydrogenase_complex 
g27	 =	0.8971	;	%Succinyl_CoA_synthetase            %%% Perturbation point for warburg effect... but basal production should not be changed I think
g28	 =	0.8277	;	%Succinate_dehydrogenase 
g29	 =	0.7796	;	%Fumarase 
g30	 =	0.4397	;	%Malate_dehydrogenase 
g31	 =	0.2944	;	%Glucose-6-phosphate_dehydrogenase 
g32	 =	0.8114	;	%6-phosphogluconolactonase 
g33	 =	0.9544	;	%Phosphogluconate_dehydrogenase
g34	 =	0.3948	;	%Ribose 5-phosphate_isomerise_A
g35	 =	0.7041	;	%Ribulose-5-phosphate-3-epimerase
g36	 =	0.4948	;	%Transketolase 
g37	 =	0.8502	;	%Transaldolase_1

% g3 2,3BPG	


x_v = zeros(1,107);
x_v(1:41) = [m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32 m33 m34 m35 m36 m37 m38 m40 m41 m42];
x_v(42:70) = [s8 s17 s1 s2 s3 s4 s5 s6 s7  s9 s10 s11 s12 s13 s14 s15 s16 s18 s19 s20 s21 s22 s23 s24 s25 s26 s27 s28 s29];
x_v(71:107) = [g1 g2 g3 g4 g5 g6 g7 g8 g9 g10 g11 g12 g13 g14 g15 g16 g17 g18 g19 g20 g21 g22 g23 g24 g25 g26 g27 g28 g29 g30 g31 g32 g33 g34 g35 g36 g37];

uu = zeros(1, 27); 

T  = 100;
Ts = 0.0002;
Tspan = [0 Ts];
t = [0:Ts:T]';


for ii = 1:(T/Ts)

[xd, ui] = CCM_three_time_scale_expression_1(t,x_v(ii,:));                                    % fetching (dx/dt) 
uu(ii, :)= real(ui(end, :));                                                                   
%Xdot(ii, :)= real(xd(end, :));                                                               
[tsim, xsim] = ode23tb('CCM_three_time_scale_expression_1', Tspan, x_v(ii,:));


%sysd = c2d([tsim, xsim],(1/50),'zoh');

x_v(ii+1, :) = real(xsim(end, :));

for j = 1:107
   if (x_v(ii+1, j))<0.1
      x_v(ii+1, j) = 0.1;
   end
end
   
for j = 1:107
   if (x_v(ii+1, j))>1.0
      x_v(ii+1, j) = 1.0;
   end   
end
 if (ii<(T/Ts))
        fprintf('Iteration %d is completed...\n\n',ii);
    else
        fprintf('Iteration %d is completed and finally simulation is completed.\n\n',ii);
 end

end