function [ xdot, u ] = CCM_three_time_scale_expression_1( t,x_v)

NOM = 107;              % the number of molecules
xdot = zeros(NOM, 1);


%%%%% METABOLITES INITIAL CONCENTRATION in nM %%%%%
%%% m39 is not included intentionally

m1	=	x_v(1);     %	glc_1P				
m2	=	x_v(2);     %	glc_6P					
m3	=	x_v(3);     %	glc				
m4	=	x_v(4);     %	f_6P					
m5	=	x_v(5);     %	f_16_BP					
m6	=	x_v(6);     %	f_26_BP				
m7	=	x_v(7);     %	dhap				
m8	=	x_v(8);     %	ga_3P					
m9	=	x_v(9);     %	BPG_13					
m10	=	x_v(10);	%	PG_3					
m11	=	x_v(11);	%	PG_2					
m12	=	x_v(12);	%	PEP					
m13	=	x_v(13);	%	PYR				
m14	=	x_v(14);	%	LACTATE		
m15	=	x_v(15);	%	Ace_coa
m16	=	x_v(16);	%	OAA		
m17	=	x_v(17);	%	Citrate			
m18	=	x_v(18);	%	Isocitrate				
m19	=	x_v(19);	%	alpha_KG					
m20	=	x_v(20);	%	Succ_coa
m21	=	x_v(21);	%	succinate				
m22	=	x_v(22);	%	fumarate				
m23	=	x_v(23);	%	malate					
m24	=	x_v(24);	%	gl_15_l6P				
m25	=	x_v(25);	%	gl_6P					
m26	=	x_v(26);	%	rbls_5P					
m27	=	x_v(27);	%	rbs_5P					
m28	=	x_v(28);	%	xyl_5P					
m29	=	x_v(29);	%	sdhl_7P					
m30	=	x_v(30);	%	eryt_4P					
m31	=	x_v(31);	%	ATP					
m32	=	x_v(32);	%	ADP			
m33	=	x_v(33);	%	NADH					
m34	=	x_v(34);	%	NAD					
m35	=	x_v(35);	%	COA
m36	=	x_v(36);	%	CO2		
m37	=	x_v(37);	%	FFA				
m38	=	x_v(38);	%	FAD				
m40	=	x_v(39);	%	FADH					
m41	=	x_v(40);	%	COA_SH					
m42	=	x_v(41);	%	NADPH				



%%%%% SIGNALLING MOLECULE INITIAL CONCENTRATION s in nM %%%%%

s8	=	x_v(42);	%	HIF_1
s17	=	x_v(43);	%	PHD

s1	=	x_v(44);	%	STAT3		
s2	=	x_v(45);	%	NFkB		
s3	=	x_v(46);	%	ERK		
s4	=	x_v(47);	%	P13K		
s5	=	x_v(48);	%	AKT		
s6	=	x_v(49);	%	mTOR		
s7	=	x_v(50);	%	MNK		
s9	=	x_v(51);	%	elf4E		
s10	=	x_v(52);	%	Ras		
s11	=	x_v(53);	%	TRAF6		
s12	=	x_v(54);	%	Raf_1		
s13	=	x_v(55);	%	ROS		
s14	=	x_v(56);	%	Rheb		
s15	=	x_v(57);	%	PTEN		
s16	=	x_v(58);	%	VHL		
s18	=	x_v(59);	%	AMP		
s19	=	x_v(60);	%	PDK1		
s20	=	x_v(61);	%	PKCs
s21	=	x_v(62);	%	BCL_X
s22	=	x_v(63);	%	FASL
s23	=	x_v(64);	%	MDM2		
s24	=	x_v(65);	%	p53		
s25	=	x_v(66);	%	MDMX		
s26	=	x_v(67);	%	MYC		
s27	=	x_v(68);	%	TIGAR
s28	=	x_v(69);	%	MTORC2		
s29	=	x_v(70);	%	MTORC1		


%%%%% ENZYME/GENE INITIAL CONCENTRATION g in nM  %%%%%

g1	=	x_v(71);	%Glut	1			
g2	=	x_v(72);	%Phosphoglucomutase_1				
g3	=	x_v(73);	%Hexokinase				
g4	=	x_v(74);	%Glucose-6-phosphatase				
g5	=	x_v(75);	%Phosphoglucoisomerase				
g6	=	x_v(76);	%Phosphofructokinase_1				
g7	=	x_v(77);	%Fructose-1,6-bisphosphatase				
g8	=	x_v(78);	%Phosphofructokinase_2				
g9	=	x_v(79);	%Fructose-2,6-bisphosphatase				
g10	=	x_v(80);	%Aldolase
g11	=	x_v(81);	%Triose_phosphate_isomerase				
g12	=	x_v(82);	%Glyceraldehyde-3-phosphate_dehydrogenase				
g13	=	x_v(83);	%Phosphoglycerate_kinase				
g14	=	x_v(84);	%Phosphoglycerate_mutase				
g15	=	x_v(85);	%Enolase				
g16	=	x_v(86);	%Pyruvate_kinase				
g17	=	x_v(87);	%Lactate_dehydrogenase				
g18	=	x_v(88);	%Pyruvate_dehydrogenase				
g19	=	x_v(89);	%Acyl-CoA_synthetase				
g20	=	x_v(90);	%Fatty_acid_synthase				
g21	=	x_v(91);	%Pyruvate_carboxylase				
g22	=	x_v(92);	%Phosphoenolpyruvate_carboxykinase_1				
g23	=	x_v(93);	%Citrate_synthase				
g24	=	x_v(94);	%Aconitase_1			
g25	=	x_v(95);	%Isocitrate_dehydrogenase				
g26	=	x_v(96);	%alpha-Ketoglutarate_dehydrogenase_complex				
g27	=	x_v(97);	%Succinyl_CoA_synthetase				
g28	=	x_v(98);	%Succinate_dehydrogenase				
g29	=	x_v(99);	%Fumarase				
g30	=	x_v(100);	%Malate_dehydrogenase				
g31	=	x_v(101);	%Glucose-6-phosphate_dehydrogenase				
g32	=	x_v(102);	%6-phosphogluconolactonase				
g33	=	x_v(103);	%Phosphogluconate_dehydrogenase				
g34	=	x_v(104);	%Ribose	5-phosphate_isomerise_A			
g35	=	x_v(105);	%Ribulose-5-phosphate-3-epimerase				
g36	=	x_v(106);	%Transketolase				
g37	=	x_v(107);	%Transaldolase_1				

% g3 2,3BPG	


%%%%%%%%%%%%%%%%  CONSTANT PARTS  %%%%%%%%%%%%%%%%%%%%%

%%%%% METABOLIC REACTION SUBSTRATE CONSTANT km in nM %%%%%

km0	 =	0.05564	;
km1	 =	0.4479	;
km2	 =	0.05966	;
km3	 =	0.052076	;
km4	 =	0.4479	;
km5	 =	0.052373	;
km6	 =	0.4289	;
km7	 =	0.054248	;
km8	 =	0.4654	;
km9	 =	0.053261	;
km10	 =	0.07412	;
km11	 =	0.052933	;
km12	 =	0.4415	;
km13	 =	0.09692	;
km14	 =	0.41931	;
km15	 =	0.051036	;
km16	 =	0.4729	;
km17	 =	0.13405	;
km18	 =	0.4658	;
km19	 =	0.053377	;
km20	 =	0.3951	;
km21	 =	0.073413	;
km22	 =	0.42947	;
km23	 =	0.13745	;
km24	 =	0.49851	;
km25	 =	0.054972	;
km26	 =	0.45491484	;
km27	 =	0.1576	;
km28	 =	0.074994	;
km29	 =	0.4786	;
km30	 =	0.052321	;
km31	 =	0.052518	;
km32	 =	0.42146	;
km33	 =	0.053936	;
km34	 =	0.063326	;
km35	 =	0.050974	;
km36	 =	0.4698	;
km37	 =	0.054877	;
km38	 =	0.051364	;
km39	 =	0.051125	;
km40	 =	0.093633	;
km41	 =	0.4922	;
km42	 =	0.062864	;
km43	 =	0.052887	;
km44	 =	0.084375	;
km45	 =	0.092682	;
km46	 =	0.052271	;
km47	 =	0.073521	;
km48	 =	0.063836	;

%%%%% METABOLIC REACTION ENZYME RATE CONATANT K in S^-1  %%%%%

K1	 =	0.09	;	% glut1
K2	 =	0.01	;	% pgm_1
K3	 =	0.09	;	% hk
K4	 =	0.01	;	% g6Pase
K5	 =	0.0812	;	% pgi
K6	 =	0.09	;	% pfk1
K7	 =	0; %0.017	;	% f16Bpase
K8	 =	0.09	;	% pfk2
K9	 =	0; %0.0766	;	% f26Bpase
K10	 =	0.0826	;	% ald
K11	 =	0.0899	;	% tpi
K12	 =	0.076	;	% gcld3PD
K13	 =	0.0532	;	% pglc_kn
K14	 =	0.0755	;	% pglc_m
K15	 =	0.0727	;	% enl
K16	 =	0.05	;	% pyrk
K17	 =	0.01	;	% lacd
K18	 =	0;%0.0809	;	% pyrd                      %%% Perturbation point for warburg effect %%%%%%% I have just stop the metabolic reaction. I do not change gene production and signaling molecules interaction.
K19	 =	0.0643	;	% acyl_cos_synthase         %%% Perturbation point for warburg effect %%%%%%% I have just stop the metabolic reaction. I do not change gene production and signaling molecules interaction.
K20	 =	0.09	;	% fa_synthase               %%% Perturbation point for warburg effect %%%%%%% I have just stop the metabolic reaction. I do not change gene production and signaling molecules interaction.
K21	 =	0;%0.0852	;	% pyr_crbxylase             %%% Perturbation point for warburg effect %%%%%%% I have just stop the metabolic reaction. I do not change gene production and signaling molecules interaction.
K22	 =	0;%0.0296	;	% pep_crbxykinase1          %%% Perturbation point for warburg effect %%%%%%% I have just stop the metabolic reaction. I do not change gene production and signaling molecules interaction.
K23	 =	0.0818	;	% cit_synthase
K24	 =	0.0844	;	% actnase
K25	 =	0.0881	;	% isocit_deh
K26	 =	0.0848	;	% KG_deh_cmp
K27	 =	0.09	;	% succ_coa_synthase             %%% Perturbation point for warburg effect %%%%%%% I have just stop the metabolic reaction. I do not change gene production and signaling molecules interaction.
K28	 =	0.0757	;	% succ_deh
K29	 =	0.0766	;	% frmase
K30	 =	0.0752	;	% mal_deh
K31	 =	0.0603	;	% g6P_deh
K32	 =	0.0191	;	% pglc6
K33	 =	0.0658	;	% phglc_deh
K34	 =	0.0792	;	% rbs_5Pis_A
K35	 =	0.069	;	% rbls_5P_3ep
K36	 =	0.0892	;	% trkl
K37	 =	0.0722	;	% trsdl_1

%%%%% SIGNALLING MOLECULE BINDING RATE CONSTANT KS in per second %%%%%
%%% Randomly generated between 0.001 to 0.009 
KS1	 =	0.0013	;
KS2	 =	0.0059	;
KS3	 =	0.0039	;
KS4	 =	0.0014	;
KS5	 =	0.0049	;
KS6	 =	0.0025	;
KS7	 =	0.002	;
KS8	 =	0.0016	;
KS9	 =	0.0022	;
KS10	 =	0.0025	;
KS11	 =	0.0013	;
KS12	 =	0.0061	;
KS13	 =	0.0033	;
KS14	 =	0.0053	;
KS15	 =	0.0066	;
KS16	 =	0.005	;
KS17	 =	0.0053	;
KS18	 =	0.0046	;
KS19	 =	0.002	;
KS20	 =	0.0049	;
KS21	 =	0.0078	;
KS22	 =	0.008	;
KS23	 =	0.0032	;
KS24	 =	0.0027	;
KS25	 =	0.0055	;
KS26	 =	0.0061	;
KS27	 =	0.0043	;
KS28	 =	0.0026	;
KS29	 =	0.0086	;
KS30	 =	0.0017	;
KS31	 =	0.0018	;
KS32	 =	0.0021	;
KS33	 =	0.0023	;
KS34	 =	0.006	;
KS35	 =	0.0056	;
KS36	 =	0.0014	;
KS37	 =	0.0084	;
KS38	 =	0.0068	;
KS39	 =	0.0069	;
KS43	 =	0.0015	;
KS44	 =	0.0079	;
KS45	 =	0.0085	;
KS46	 =	0.0089	;
KS47	 =	0.0079	;
KS48	 =	0.0073	;
KS49	 =	0.0051	;
KS50	 =	0.0024	;
KS52	 =	0.0042	;
KS53	 =	0.0021	;
KS54	 =	0.0012	;

%%%%% GENE BASAL PRODUCTION RATE CONSTANT kbg in per second %%%%%

kbg1	 =	0.0004939	;	%	glut1
kbg2	 =	0.0004301	;	%	pgm_1
kbg3	 =	0.0004296	;	%	hk
kbg4	 =	0.0004333	;	%	g6Pase
kbg5	 =	0.0004467	;	%	pgi
kbg6	 =	0.0004648	;	%	pfk1
kbg7	 =	0.0004025	;	%	f16Bpase
kbg8	 =	0.0004842	;	%	pfk2
kbg9	 =	0.0004559	;	%	f26Bpase
kbg10	 =	0.0004854	;	%	ald
kbg11	 =	0.0004348	;	%	tpi
kbg12	 =	0.0004446	;	%	gcld3PD
kbg13	 =	0.0004054	;	%	pglc_kn
kbg14	 =	0.0004177	;	%	pglc_m
kbg15	 =	0.0004663	;	%	enl
kbg16	 =	0.0004331	;	%	pyrk
kbg17	 =	0.0004898	;	%	lacd
kbg18	 =	0.0004118	;	%	pyrd                          %%% Perturbation point for warburg effect... but basal production should not be changed I think
kbg19	 =	0.0004988	;	%	acyl_cos_synthase             %%% Perturbation point for warburg effect... but basal production should not be changed I think
kbg20	 =	0.000454	;	%	fa_synthase                   %%% Perturbation point for warburg effect... but basal production should not be changed I think
kbg21	 =	0.0004707	;	%	pyr_crbxylase                 %%% Perturbation point for warburg effect... but basal production should not be changed I think
kbg22	 =	0.0004999	;	%	pep_crbxykinase1              %%% Perturbation point for warburg effect... but basal production should not be changed I think
kbg23	 =	0.0004288	;	%	cit_synthase
kbg24	 =	0.0004415	;	%	actnase
kbg25	 =	0.0004465	;	%	isocit_deh
kbg26	 =	0.0004764	;	%	KG_deh_cmp
kbg27	 =	0.0004818	;	%	succ_coa_synthase             %%% Perturbation point for warburg effect... but basal production should not be changed I think
kbg28	 =	0.00041	;	%	succ_deh
kbg29	 =	0.0004178	;	%	frmase
kbg30	 =	0.000436	;	%	mal_deh
kbg31	 =	0.0004057	;	%	g6P_deh
kbg32	 =	0.0004522	;	%	pglc6
kbg33	 =	0.0004336	;	%	phglc_deh
kbg34	 =	0.0004176	;	%	rbs_5Pis_A
kbg35	 =	0.0004209	;	%	rbls_5P_3ep
kbg36	 =	0.0004905	;	%	trkl
kbg37	 =	0.0004675	;	%	trsdl_1


%%%%% GENE TRANSCRIPTION FACTORS BINDING RATE CONSTANT kg in per second %%%%%

kg1	=	0.0007405	;
kg2	=	0.0008736	;
kg3	=	0.0006312	;
kg4	=	0.0008237	;
kg5	=	0.0008209	;
kg6	=	0.0007686	;
kg7	=	0.0006553	;
kg8	=	0.0007792	;
kg9	=	0.00069	;
kg10	=	0.0006402	;
kg11	=	0.0006638	;
kg12	=	0.0008685	;
kg13	=	0.0006214	;
kg14	=	0.0006727	;
kg15	=	0.0006161	;
kg16	=	0.0007325	;
kg17	=	0.000604	;
kg18	=	0.0008692	;   %%% Perturbation point for warburg effect... but production should not be changed I think
kg19	=	0.000659	;   %%% Perturbation point for warburg effect... but production should not be changed I think                         
kg20	=	0.000628	;   %%% Perturbation point for warburg effect... but production should not be changed I think
kg21	=	0.0006922	;   %%% Perturbation point for warburg effect... but production should not be changed I think
kg22	=	0.0007368	;   %%% Perturbation point for warburg effect... but production should not be changed I think
kg23	=	0.0006305	;
kg24	=	0.0008986	;
kg25	=	0.0006996	;
kg26	=	0.0006892	;
kg27	=	0.0006186	;   %%% Perturbation point for warburg effect... but production should not be changed I think
kg28	=	0.0006895	;
kg29	=	0.0006139	;
kg30	=	0.0007516	;
kg31	=	0.0008284	;
kg32	=	0.0007893	;
kg33	=	0.000627	;
kg34	=	0.0006243	;
kg35	=	0.0008332	;
kg36	=	0.0008715	;
kg37	=	0.0007601	;

%%%%% GENE DECAY RATE CONSTANT lg in per second %%%%%

lg1=	0.0001109	;	%	glut1
lg2=	0.0001826	;	%	pgm_1
lg3=	0.0001338	;	%	hk
lg4=	0.0001294	;	%	g6Pase
lg5=	0.0001746	;	%	pgi
lg6=	0.000101	;	%	pfk1
lg7=	0.0001048	;	%	f16Bpase
lg8=	0.0001668	;	%	pfk2
lg9=	0.0001603	;	%	f26Bpase
lg10=	0.0001526	;	%	ald
lg11=	0.000173	;	%	tpi
lg12=	0.0001707	;	%	gcld3PD
lg13=	0.0001781	;	%	pglc_kn
lg14=	0.0001288	;	%	pglc_m
lg15=	0.0001693	;	%	enl
lg16=	0.0001557	;	%	pyrk
lg17=	0.0001397	;	%	lacd
lg18=	0.0001062	;	%	pyrd
lg19=	0.000178	;	%	acyl_cos_synthase
lg20=	0.0001338	;	%	fa_synthase
lg21=	0.0001608	;	%	pyr_crbxylase
lg22=	0.0001741	;	%	pep_crbxykinase1
lg23=	0.0001105	;	%	cit_synthase
lg24=	0.0001128	;	%	actnase
lg25=	0.000155	;	%	isocit_deh
lg26=	0.0001485	;	%	KG_deh_cmp
lg27=	0.000189	;	%	succ_coa_synthase
lg28=	0.0001799	;	%	succ_deh
lg29=	0.0001734	;	%	frmase
lg30=	0.0001051	;	%	mal_deh
lg31=	0.0001073	;	%	g6P_deh
lg32=	0.0001089	;	%	pglc6
lg33=	0.0001798	;	%	phglc_deh
lg34=	0.0001943	;	%	rbs_5Pis_A
lg35=	0.0001684	;	%	rbls_5P_3ep
lg36=	0.0001132	;	%	trkl
lg37=	0.0001723	;	%	trsdl_1
%%%%% Feedback parameters for Metabolic Network %%%%%%
%%%% F49, F50 is not included intentionally
F1	=	0.9;% 0.2435;
F2	=	0.9;%0.9293;
F3	=	0.9;%0.3500;
F4	=	0.9;%0.1966;
F5	=	0.9;%0.2511;
F6	=	0.9;%0.6160;
F7	=	0.9;%0.4733;
F8	=	0.9;%0.3517;
F9	=	0.9;%0.8308;
F10	=	0.9;%0.5853;
F11	=	0.9;%0.5497;
F12	=	0.9;%0.9172;
F13	=	0.9;%0.2858;
F14_modified =	0.9;%0.7572;
F15	=	0.9;%0.7537;
F16	=	0.9;%0.3804;
F17	=	0.9;%0.5678;
F18	=	0.9;%0.0759;
F19	=	0.9;%0.0540;
F20	=	0.9;%0.5308;
F21	=	0.9;%0.7792;
F22	=	0.9;%0.9340;
F23	=	0.9;%0.1299;
F24	=	0.9;%0.5688;
F25	=	0.9;%0.4694;
F26	=	0.9;%0.0119;
F27	=	0.9;%0.3371;
F28	=	0.9;%0.1622;
F29	=	0.9;%0.7943;
F30	=	0.9;%0.3112;
F31	=	0.9;%0.5285;
F32	=	0.9;%0.1656;
F33	=	0.9;%0.6020;
F34	=	0.9;%0.2630;
F35	=	0.9;%0.6541;
F36	=	0.9;%0.6892;
F37	=	0.9;%0.7482;
F38	=	0.9;%0.4505;
F39	=	0.9;%0.0838;
F40	=	0.9;%0.2290;
F41	=	0.9;%0.9133;
F42	=	0.9;%0.1524;
F43	=	0.9;%0.8258;
F44	=	0.9;%0.5383;
F45	=	0.9;%0.9961;
F46	=	0.9;%0.0782;
F47	=	0.9;%0.4427;
F48	=	0.9;%0.1067;

F51	=	0.9;%0.7749;
F52	=	0.9;%0.8173;
F53	=	0.9;%0.8687;
F54	=	0.9;%0.0844;
F55	=	0.9;%0.0716;

%%%%% Feedback parameters for Signalling Network %%%%%%
%%%% FS_12 is not included intentionally
F_S1	=	0.9;%0.3998	;
F_S2	=	0.9;%0.2599	;
F_S3	=	0.9;%0.8001	;
F_S4	=	0.9;%0.4314	;
F_S5	=	0.9;%0.9106	;
F_S6	=	0.9;%0.1818	;
F_S7	=	0.9;%0.2638	;
F_S8	=	0.9;%0.1455	;
F_S9	=	0.9;%0.1361	;
F_S10	=	0.9;%0.8693	;
F_S11	=	0.9;%0.5797	;
F_S13	=	0.9;%0.145	;
F_S14	=	0.9;%0.853	;
F_S15	=	0.9;%0.6221	;
F_S16	=	0.9;%0.351	;
F_S17	=	0.9;%0.5132	;
F_S18	=	0.9;%0.4018	;
F_S19	=	0.9;%0.076	;
F_S20	=	0.9;%0.2399	;
F_S21	=	0.9;%0.1233	;
F_S22	=	0.9;%0.1839	;
F_S23	=	0.9;%0.24	;
F_S24	=	0.9;%0.4173	;
F_S25	=	0.9;%0.0497	;
F_S26	=	0.9;%0.9027	;
F_S27	=	0.9;%0.9448	;
F_S28	=	0.9;%0.4909	;
F_S29	=	0.9;%0.4893	;
F_S30	=	0.9;%0.3377	;
F_S31	=	0.9;%0.9001	;
F_S32	=	0.9;%0.3692	;


%%%%% Feedback parameters for Gene Regulatory Network %%%%%%

FG1	= 0.9;%0.5289;
FG2	= 0.9;%0.6944;
FG3	= 0.9;%0.2124;
FG4	= 0.9;%0.5433;
FG5	= 0.9;%0.7025;
FG6	= 0.9;%0.9564;
FG7	= 0.9;%0.4445;
FG8	= 0.9;%0.0854;
FG9	= 0.9;%0.0573;
FG10 = 0.9;%0.6295;
FG11 = 0.9;%0.7962;
FG12 = 0.9;%0.6912;
FG13 = 0.9;%0.3453;






%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


constant_terms_in_u18_design = ((KS7*s3)/60 + (KS29*s3)/60 + KS13/(60*(F_S15*s3 + 1)*(F_S14*s18 + 1)) + (KS9*s7)/(60*(F_S8*s3 + 1)*(F_S9*s6 + 1)) - (KS36*(kbg16 - lg16 + kg16*s8))/(60*FG6*kg16*s8) + (KS8*s1*s2*s3*s6*s9)/(60*(F_S6*s16 + 1)*(F_S7*s17 + 1)));

u(1) = -(60*((KS31*(kbg16 - lg16 + kg16*s8))/(60*FG6*kg16*s8) - (KS8*s1*s2*s3*s6*s9)/(60*(F_S6*s16 + 1)*(F_S7*s17 + 1))))/KS1;
u(2) = ((kg28*s5*s24*s26*(kbg9 + kbg20 + kbg21 + kbg27 + kbg29 + kbg30 + kbg37 - lg9 - lg20 - lg21 - lg27 - lg29 - lg30 - lg37 + KS21/(60*F_S24*s23 + 60) - (KS19*s5*s24*s25)/((F_S20*s26 + 1)*(60*F_S21*s10 + 60))))/((kbg28 - lg28)*(kg30*s3 + kg9*s3*s26 + kg27*s3*s26 + kg37*s1*s26 + kg29*s2*s24*s26 + kg21*s1*s2*s24*s26 + (kg20*s3*s5*s26)/(FG9*s18 + 1) + (F_S22*KS19*s5*s24*s25)/((F_S20*s26 + 1)*(60*F_S21*s10 + 60)))) - 1)/FG13;
u(3) = -(60*((KS29*s3)/60 + (KS19*s5*s24*s25)/(60*(F_S21*s10 + 1)*(F_S20*s26 + 1)*((F_S22*(kbg9 + kbg20 + kbg21 + kbg27 + kbg29 + kbg30 + kbg37 - lg9 - lg20 - lg21 - lg27 - lg29 - lg30 - lg37 + KS21/(60*F_S24*s23 + 60) - (KS19*s5*s24*s25)/((F_S20*s26 + 1)*(60*F_S21*s10 + 60))))/(kg30*s3 + kg9*s3*s26 + kg27*s3*s26 + kg37*s1*s26 + kg29*s2*s24*s26 + kg21*s1*s2*s24*s26 + (kg20*s3*s5*s26)/(FG9*s18 + 1) + (F_S22*KS19*s5*s24*s25)/((F_S20*s26 + 1)*(60*F_S21*s10 + 60))) - 1))))/KS54;
u(4) = -((KS49*((KS33*((KS29*s3)/60 + (KS19*s5*s24*s25)/(60*(F_S21*s10 + 1)*(F_S20*s26 + 1)*((F_S22*(kbg9 + kbg20 + kbg21 + kbg27 + kbg29 + kbg30 + kbg37 - lg9 - lg20 - lg21 - lg27 - lg29 - lg30 - lg37 + KS21/(60*F_S24*s23 + 60) - (KS19*s5*s24*s25)/((F_S20*s26 + 1)*(60*F_S21*s10 + 60))))/(kg30*s3 + kg9*s3*s26 + kg27*s3*s26 + kg37*s1*s26 + kg29*s2*s24*s26 + kg21*s1*s2*s24*s26 + (kg20*s3*s5*s26)/(FG9*s18 + 1) + (F_S22*KS19*s5*s24*s25)/((F_S20*s26 + 1)*(60*F_S21*s10 + 60))) - 1))))/KS54 - (KS32*((kg28*s5*s24*s26*(kbg9 + kbg20 + kbg21 + kbg27 + kbg29 + kbg30 + kbg37 - lg9 - lg20 - lg21 - lg27 - lg29 - lg30 - lg37 + KS21/(60*F_S24*s23 + 60) - (KS19*s5*s24*s25)/((F_S20*s26 + 1)*(60*F_S21*s10 + 60))))/((kbg28 - lg28)*(kg30*s3 + kg9*s3*s26 + kg27*s3*s26 + kg37*s1*s26 + kg29*s2*s24*s26 + kg21*s1*s2*s24*s26 + (kg20*s3*s5*s26)/(FG9*s18 + 1) + (F_S22*KS19*s5*s24*s25)/((F_S20*s26 + 1)*(60*F_S21*s10 + 60)))) - 1))/(60*FG13) - (KS2*s4*s11)/60 + (KS17*s2)/(60*(F_S17*s4 + 1)*(F_S18*s5 + 1)) + (KS8*s1*s2*s3*s6*s9)/(60*(F_S6*s16 + 1)*(F_S7*s17 + 1))))/KS35 - (KS2*s4*s11)/60)/(KS48/60 - (KS34*KS49)/(60*KS35));
u(5) = (60*((KS2*s4*s11)/60 + (KS48*((KS49*((KS33*((KS29*s3)/60 + (KS19*s5*s24*s25)/(60*(F_S21*s10 + 1)*(F_S20*s26 + 1)*((F_S22*(kbg9 + kbg20 + kbg21 + kbg27 + kbg29 + kbg30 + kbg37 - lg9 - lg20 - lg21 - lg27 - lg29 - lg30 - lg37 + KS21/(60*F_S24*s23 + 60) - (KS19*s5*s24*s25)/((F_S20*s26 + 1)*(60*F_S21*s10 + 60))))/(kg30*s3 + kg9*s3*s26 + kg27*s3*s26 + kg37*s1*s26 + kg29*s2*s24*s26 + kg21*s1*s2*s24*s26 + (kg20*s3*s5*s26)/(FG9*s18 + 1) + (F_S22*KS19*s5*s24*s25)/((F_S20*s26 + 1)*(60*F_S21*s10 + 60))) - 1))))/KS54 - (KS32*((kg28*s5*s24*s26*(kbg9 + kbg20 + kbg21 + kbg27 + kbg29 + kbg30 + kbg37 - lg9 - lg20 - lg21 - lg27 - lg29 - lg30 - lg37 + KS21/(60*F_S24*s23 + 60) - (KS19*s5*s24*s25)/((F_S20*s26 + 1)*(60*F_S21*s10 + 60))))/((kbg28 - lg28)*(kg30*s3 + kg9*s3*s26 + kg27*s3*s26 + kg37*s1*s26 + kg29*s2*s24*s26 + kg21*s1*s2*s24*s26 + (kg20*s3*s5*s26)/(FG9*s18 + 1) + (F_S22*KS19*s5*s24*s25)/((F_S20*s26 + 1)*(60*F_S21*s10 + 60)))) - 1))/(60*FG13) - (KS2*s4*s11)/60 + (KS17*s2)/(60*(F_S17*s4 + 1)*(F_S18*s5 + 1)) + (KS8*s1*s2*s3*s6*s9)/(60*(F_S6*s16 + 1)*(F_S7*s17 + 1))))/KS35 - (KS2*s4*s11)/60))/(60*(KS48/60 - (KS34*KS49)/(60*KS35)))))/KS49;
u(6) = (kbg16 - lg16 + kg16*s8)/(FG6*kg16*s8);
u(7) = (60*((KS2*s4*s11)/60 - (KS4*s10)/(60*(F_S3*s15 + 1)) + (KS18*s4)/(60*(F_S19*s5 + 1)) + (KS5*s4*s28)/(60*(F_S4*s15 + 1)) + (KS14*s4*s8)/(60*(F_S16*s15 + 1)) + (KS17*s2)/(60*(F_S17*s4 + 1)*(F_S18*s5 + 1)) - (KS37*(kbg16 - lg16 + kg16*s8))/(60*FG6*kg16*s8) - (KS52*(kbg16 - lg16 + kg16*s8))/(60*FG6*kg16*s8) + (KS6*s4*s5*s13*s14)/(60*((60*(KS24/(60*F_S28*s13 + 60) - (KS28*m31*s29)/60 + KS23/((F_S26*m22 + 1)*(60*F_S25*m21 + 60)*(F_S27*s13 + 1)) + (KS6*s4*s5*s13*s14)/60))/(KS6*s4*s5*s13*s14) + 1))))/KS38;
u(8) = -((K9*g9*m6)/((F23*m4 + 1)*(3600*km10 + 3600*m6)) - (K8*g8*m4*m31*(F20*m4 + 1))/(3600*km9 + 3600*m4*m31))/((F22*K9*g9*m6)/((F23*m4 + 1)*(3600*km10 + 3600*m6)) + (F21*K8*g8*m4*m31*(F20*m4 + 1))/(3600*km9 + 3600*m4*m31));
u(9) = (60*(KS24/(60*F_S28*s13 + 60) - (KS28*m31*s29)/60 + KS23/((F_S26*m22 + 1)*(60*F_S25*m21 + 60)*(F_S27*s13 + 1)) + (KS6*s4*s5*s13*s14)/60))/(F_S5*KS6*s4*s5*s13*s14);
u(10) = -(kbg23 - lg23)/kg23;
u(11) = (60*((KS47*((KS49*((KS33*((KS29*s3)/60 + (KS19*s5*s24*s25)/(60*(F_S21*s10 + 1)*(F_S20*s26 + 1)*((F_S22*(kbg9 + kbg20 + kbg21 + kbg27 + kbg29 + kbg30 + kbg37 - lg9 - lg20 - lg21 - lg27 - lg29 - lg30 - lg37 + KS21/(60*F_S24*s23 + 60) - (KS19*s5*s24*s25)/((F_S20*s26 + 1)*(60*F_S21*s10 + 60))))/(kg30*s3 + kg9*s3*s26 + kg27*s3*s26 + kg37*s1*s26 + kg29*s2*s24*s26 + kg21*s1*s2*s24*s26 + (kg20*s3*s5*s26)/(FG9*s18 + 1) + (F_S22*KS19*s5*s24*s25)/((F_S20*s26 + 1)*(60*F_S21*s10 + 60))) - 1))))/KS54 - (KS32*((kg28*s5*s24*s26*(kbg9 + kbg20 + kbg21 + kbg27 + kbg29 + kbg30 + kbg37 - lg9 - lg20 - lg21 - lg27 - lg29 - lg30 - lg37 + KS21/(60*F_S24*s23 + 60) - (KS19*s5*s24*s25)/((F_S20*s26 + 1)*(60*F_S21*s10 + 60))))/((kbg28 - lg28)*(kg30*s3 + kg9*s3*s26 + kg27*s3*s26 + kg37*s1*s26 + kg29*s2*s24*s26 + kg21*s1*s2*s24*s26 + (kg20*s3*s5*s26)/(FG9*s18 + 1) + (F_S22*KS19*s5*s24*s25)/((F_S20*s26 + 1)*(60*F_S21*s10 + 60)))) - 1))/(60*FG13) - (KS2*s4*s11)/60 + (KS17*s2)/(60*(F_S17*s4 + 1)*(F_S18*s5 + 1)) + (KS8*s1*s2*s3*s6*s9)/(60*(F_S6*s16 + 1)*(F_S7*s17 + 1))))/KS35 - (KS2*s4*s11)/60))/(60*(KS48/60 - (KS34*KS49)/(60*KS35))) - KS13/(60*(F_S15*s3 + 1)*(F_S14*s18 + 1)) + (KS6*s4*s5*s13*s14)/(60*((60*(KS24/(60*F_S28*s13 + 60) - (KS28*m31*s29)/60 + KS23/((F_S26*m22 + 1)*(60*F_S25*m21 + 60)*(F_S27*s13 + 1)) + (KS6*s4*s5*s13*s14)/60))/(KS6*s4*s5*s13*s14) + 1))))/KS46;
u(12) = (60*((KS22*s24)/60 + (KS25*s24)/60 - (KS20*s8*s18)/(60*((60*F_S11*((KS2*s4*s11)/60 - (KS4*s10)/(60*(F_S3*s15 + 1)) + (KS18*s4)/(60*(F_S19*s5 + 1)) + (KS5*s4*s28)/(60*(F_S4*s15 + 1)) + (KS14*s4*s8)/(60*(F_S16*s15 + 1)) + (KS17*s2)/(60*(F_S17*s4 + 1)*(F_S18*s5 + 1)) - (KS37*(kbg16 - lg16 + kg16*s8))/(60*FG6*kg16*s8) - (KS52*(kbg16 - lg16 + kg16*s8))/(60*FG6*kg16*s8) + (KS6*s4*s5*s13*s14)/(60*((60*(KS24/(60*F_S28*s13 + 60) - (KS28*m31*s29)/60 + KS23/((F_S26*m22 + 1)*(60*F_S25*m21 + 60)*(F_S27*s13 + 1)) + (KS6*s4*s5*s13*s14)/60))/(KS6*s4*s5*s13*s14) + 1))))/KS38 + 1)*(F_S23*s23 + 1)) - (KS19*s5*s24*s25)/(60*(F_S21*s10 + 1)*(F_S20*s26 + 1)*((F_S22*(kbg9 + kbg20 + kbg21 + kbg27 + kbg29 + kbg30 + kbg37 - lg9 - lg20 - lg21 - lg27 - lg29 - lg30 - lg37 + KS21/(60*F_S24*s23 + 60) - (KS19*s5*s24*s25)/((F_S20*s26 + 1)*(60*F_S21*s10 + 60))))/(kg30*s3 + kg9*s3*s26 + kg27*s3*s26 + kg37*s1*s26 + kg29*s2*s24*s26 + kg21*s1*s2*s24*s26 + (kg20*s3*s5*s26)/(FG9*s18 + 1) + (F_S22*KS19*s5*s24*s25)/((F_S20*s26 + 1)*(60*F_S21*s10 + 60))) - 1))))/KS50;
u(13) = ((constant_terms_in_u18_design*(60*constant_terms_in_u18_design + KS11*s10 + 60*F_S13*constant_terms_in_u18_design*s5))/(15*(F_S13*s5 + 1)))^(1/2)/(2*F_S1*constant_terms_in_u18_design);
u(14) = ((60*KS3*s12*(F_S13*s5 + 1))/(KS11*((30*((constant_terms_in_u18_design*(60*constant_terms_in_u18_design + KS11*s10 + 60*F_S13*constant_terms_in_u18_design*s5))/(15*(F_S13*s5 + 1)))^(1/2))/constant_terms_in_u18_design + 60)) - 1)/F_S2;
u(15) = (KS10/((KS43*((K9*g9*m6)/((F23*m4 + 1)*(3600*km10 + 3600*m6)) - (K8*g8*m4*m31*(F20*m4 + 1))/(3600*km9 + 3600*m4*m31)))/((F22*K9*g9*m6)/((F23*m4 + 1)*(3600*km10 + 3600*m6)) + (F21*K8*g8*m4*m31*(F20*m4 + 1))/(3600*km9 + 3600*m4*m31)) + (KS4*s10)/(F_S3*s15 + 1) + (KS11*s10)/(F_S13*s5 + 1) - (KS39*(kbg16 - lg16 + kg16*s8))/(FG6*kg16*s8) + (KS11*s10*((30*((constant_terms_in_u18_design*(60*constant_terms_in_u18_design + KS11*s10 + 60*F_S13*constant_terms_in_u18_design*s5))/(15*(F_S13*s5 + 1)))^(1/2))/constant_terms_in_u18_design + 60))/(60*(((constant_terms_in_u18_design*(60*constant_terms_in_u18_design + KS11*s10 + 60*F_S13*constant_terms_in_u18_design*s5))/(15*(F_S13*s5 + 1)))^(1/2)/(2*constant_terms_in_u18_design) + 1)*(F_S13*s5 + 1)) - (KS19*s5*s24*s25)/((F_S21*s10 + 1)*(F_S20*s26 + 1)*((F_S22*(kbg9 + kbg20 + kbg21 + kbg27 + kbg29 + kbg30 + kbg37 - lg9 - lg20 - lg21 - lg27 - lg29 - lg30 - lg37 + KS21/(60*F_S24*s23 + 60) - (KS19*s5*s24*s25)/((F_S20*s26 + 1)*(60*F_S21*s10 + 60))))/(kg30*s3 + kg9*s3*s26 + kg27*s3*s26 + kg37*s1*s26 + kg29*s2*s24*s26 + kg21*s1*s2*s24*s26 + (kg20*s3*s5*s26)/(FG9*s18 + 1) + (F_S22*KS19*s5*s24*s25)/((F_S20*s26 + 1)*(60*F_S21*s10 + 60))) - 1))) - 1)/F_S10;
u(16) = -(kbg9 + kbg20 + kbg21 + kbg27 + kbg29 + kbg30 + kbg37 - lg9 - lg20 - lg21 - lg27 - lg29 - lg30 - lg37 + KS21/(60*F_S24*s23 + 60) - (KS19*s5*s24*s25)/((F_S20*s26 + 1)*(60*F_S21*s10 + 60)))/(kg30*s3 + kg9*s3*s26 + kg27*s3*s26 + kg37*s1*s26 + kg29*s2*s24*s26 + kg21*s1*s2*s24*s26 + (kg20*s3*s5*s26)/(FG9*s18 + 1) + (F_S22*KS19*s5*s24*s25)/((F_S20*s26 + 1)*(60*F_S21*s10 + 60)));
u(17) = -(km0*((K4*g4*m2*(F7*m2 + 1))/(km4 + m2) - (K3*g3*m3*m31*(F4*m32 + 1)*((3600*(km3 + m3*m31)*(F5*m2 + 1)*(F6*m15 + 1)*((K2*g2*m1*((F1*((K2*g2*m1)/(3600*(km1 + m1)) - (K2*g2*m2)/(3600*(km2 + m2))))/((F1*K2*g2*m1)/(3600*(km1 + m1)) - (F2*K2*g2*m2)/(3600*(km2 + m2))) - 1))/(3600*(km1 + m1)) - (K2*g2*m2*((F2*((K2*g2*m1)/(3600*(km1 + m1)) - (K2*g2*m2)/(3600*(km2 + m2))))/((F1*K2*g2*m1)/(3600*(km1 + m1)) - (F2*K2*g2*m2)/(3600*(km2 + m2))) - 1))/(3600*(km2 + m2)) + (K4*g4*m2*(F7*m2 + 1))/(3600*(km4 + m2)) + (K5*g5*m2)/(3600*(km5 + m2)*(F8*m5 + 1)*(F9*m25 + 1)*(F10*m30 + 1)) - (K5*g5*m4)/(3600*(km6 + m4)*(F11*m5 + 1)*(F12*m25 + 1)*(F13*m30 + 1)) - (K3*g3*m3*m31*(F4*m32 + 1))/(3600*(km3 + m3*m31)*(F5*m2 + 1)*(F6*m15 + 1)) + (K31*g31*km40*km42*m2*(2*K31*g31*km42*m2 - 3600*((4*K31^2*g31^2*km42^2*m2^2*m26^2 + K33^2*g33^2*km40^2*m25^2*m26^2 + 4*K31^2*g31^2*km42^2*km43*km44*m2^2 + K33^2*g33^2*km40^2*km43*km44*m25^2 + 4*K31^2*g31^2*km42^2*km43*m2^2*m26 + 4*K31^2*g31^2*km42^2*km44*m2^2*m26 + K33^2*g33^2*km40^2*km43*m25^2*m26 + K33^2*g33^2*km40^2*km44*m25^2*m26 + 8*K31*K34*g31*g34*km42^2*m2^2*m26^2 - 8*K31*K35*g31*g35*km42^2*m2^2*m26^2 + 4*K33*K34*g33*g34*km40^2*m25^2*m26^2 - 4*K33*K35*g33*g35*km40^2*m25^2*m26^2 + F52^2*K33^2*g33^2*km40^2*m25^2*m26^2*m42^2 + 2*F52*K33^2*g33^2*km40^2*m25^2*m26^2*m42 + 8*K31*K34*g31*g34*km42^2*km44*m2^2*m26 - 8*K31*K35*g31*g35*km42^2*km43*m2^2*m26 + 4*K33*K34*g33*g34*km40^2*km44*m25^2*m26 - 4*K33*K35*g33*g35*km40^2*km43*m25^2*m26 + F52^2*K33^2*g33^2*km40^2*km43*km44*m25^2*m42^2 + F52^2*K33^2*g33^2*km40^2*km43*m25^2*m26*m42^2 + F52^2*K33^2*g33^2*km40^2*km44*m25^2*m26*m42^2 + 2*F52*K33^2*g33^2*km40^2*km43*km44*m25^2*m42 + 2*F52*K33^2*g33^2*km40^2*km43*m25^2*m26*m42 + 2*F52*K33^2*g33^2*km40^2*km44*m25^2*m26*m42 + 4*K31*K33*g31*g33*km40*km42*m2*m25*m26^2 + 4*F52^2*K33*K34*g33*g34*km40^2*m25^2*m26^2*m42^2 - 4*F52^2*K33*K35*g33*g35*km40^2*m25^2*m26^2*m42^2 + 8*F52*K31*K34*g31*g34*km42^2*m2^2*m26^2*m42 - 8*F52*K31*K35*g31*g35*km42^2*m2^2*m26^2*m42 + 8*F52*K33*K34*g33*g34*km40^2*m25^2*m26^2*m42 - 8*F52*K33*K35*g33*g35*km40^2*m25^2*m26^2*m42 + 4*K31*K33*g31*g33*km40*km42*km43*km44*m2*m25 + 4*K31*K33*g31*g33*km40*km42*km43*m2*m25*m26 + 4*K31*K33*g31*g33*km40*km42*km44*m2*m25*m26 + 4*F52^2*K33*K34*g33*g34*km40^2*km44*m25^2*m26*m42^2 - 4*F52^2*K33*K35*g33*g35*km40^2*km43*m25^2*m26*m42^2 + 8*F52*K31*K34*g31*g34*km42^2*km44*m2^2*m26*m42 - 8*F52*K31*K35*g31*g35*km42^2*km43*m2^2*m26*m42 + 8*F52*K33*K34*g33*g34*km40^2*km44*m25^2*m26*m42 - 8*F52*K33*K35*g33*g35*km40^2*km43*m25^2*m26*m42 + 4*F52*K31*K33*g31*g33*km40*km42*m2*m25*m26^2*m42 + 4*F52*K31*K33*g31*g33*km40*km42*km43*km44*m2*m25*m42 + 4*F52*K31*K33*g31*g33*km40*km42*km43*m2*m25*m26*m42 + 4*F52*K31*K33*g31*g33*km40*km42*km44*m2*m25*m26*m42)/(12960000*(km43 + m26)*(km44 + m26)))^(1/2) + K33*g33*km40*m25 + F52*K33*g33*km40*m25*m42))/(7200*(F52*m42 + 1)*(km40 + (km40*km42*m2*(2*K31*g31*km42*m2 - 3600*((4*K31^2*g31^2*km42^2*m2^2*m26^2 + K33^2*g33^2*km40^2*m25^2*m26^2 + 4*K31^2*g31^2*km42^2*km43*km44*m2^2 + K33^2*g33^2*km40^2*km43*km44*m25^2 + 4*K31^2*g31^2*km42^2*km43*m2^2*m26 + 4*K31^2*g31^2*km42^2*km44*m2^2*m26 + K33^2*g33^2*km40^2*km43*m25^2*m26 + K33^2*g33^2*km40^2*km44*m25^2*m26 + 8*K31*K34*g31*g34*km42^2*m2^2*m26^2 - 8*K31*K35*g31*g35*km42^2*m2^2*m26^2 + 4*K33*K34*g33*g34*km40^2*m25^2*m26^2 - 4*K33*K35*g33*g35*km40^2*m25^2*m26^2 + F52^2*K33^2*g33^2*km40^2*m25^2*m26^2*m42^2 + 2*F52*K33^2*g33^2*km40^2*m25^2*m26^2*m42 + 8*K31*K34*g31*g34*km42^2*km44*m2^2*m26 - 8*K31*K35*g31*g35*km42^2*km43*m2^2*m26 + 4*K33*K34*g33*g34*km40^2*km44*m25^2*m26 - 4*K33*K35*g33*g35*km40^2*km43*m25^2*m26 + F52^2*K33^2*g33^2*km40^2*km43*km44*m25^2*m42^2 + F52^2*K33^2*g33^2*km40^2*km43*m25^2*m26*m42^2 + F52^2*K33^2*g33^2*km40^2*km44*m25^2*m26*m42^2 + 2*F52*K33^2*g33^2*km40^2*km43*km44*m25^2*m42 + 2*F52*K33^2*g33^2*km40^2*km43*m25^2*m26*m42 + 2*F52*K33^2*g33^2*km40^2*km44*m25^2*m26*m42 + 4*K31*K33*g31*g33*km40*km42*m2*m25*m26^2 + 4*F52^2*K33*K34*g33*g34*km40^2*m25^2*m26^2*m42^2 - 4*F52^2*K33*K35*g33*g35*km40^2*m25^2*m26^2*m42^2 + 8*F52*K31*K34*g31*g34*km42^2*m2^2*m26^2*m42 - 8*F52*K31*K35*g31*g35*km42^2*m2^2*m26^2*m42 + 8*F52*K33*K34*g33*g34*km40^2*m25^2*m26^2*m42 - 8*F52*K33*K35*g33*g35*km40^2*m25^2*m26^2*m42 + 4*K31*K33*g31*g33*km40*km42*km43*km44*m2*m25 + 4*K31*K33*g31*g33*km40*km42*km43*m2*m25*m26 + 4*K31*K33*g31*g33*km40*km42*km44*m2*m25*m26 + 4*F52^2*K33*K34*g33*g34*km40^2*km44*m25^2*m26*m42^2 - 4*F52^2*K33*K35*g33*g35*km40^2*km43*m25^2*m26*m42^2 + 8*F52*K31*K34*g31*g34*km42^2*km44*m2^2*m26*m42 - 8*F52*K31*K35*g31*g35*km42^2*km43*m2^2*m26*m42 + 8*F52*K33*K34*g33*g34*km40^2*km44*m25^2*m26*m42 - 8*F52*K33*K35*g33*g35*km40^2*km43*m25^2*m26*m42 + 4*F52*K31*K33*g31*g33*km40*km42*m2*m25*m26^2*m42 + 4*F52*K31*K33*g31*g33*km40*km42*km43*km44*m2*m25*m42 + 4*F52*K31*K33*g31*g33*km40*km42*km43*m2*m25*m26*m42 + 4*F52*K31*K33*g31*g33*km40*km42*km44*m2*m25*m26*m42)/(12960000*(km43 + m26)*(km44 + m26)))^(1/2) + K33*g33*km40*m25 + F52*K33*g33*km40*m25*m42))/(2*(2*K31*g31*km42^2*m2^2 + K33*g33*km40^2*m25^2 + F52*K33*g33*km40^2*m25^2*m42)))*(2*K31*g31*km42^2*m2^2 + K33*g33*km40^2*m25^2 + F52*K33*g33*km40^2*m25^2*m42))))/(K3*g3*m3*m31*(F4*m32 + 1)) + 1))/((km3 + m3*m31)*(F5*m2 + 1)*(F6*m15 + 1))))/(K1*g1*(((K4*g4*m2*(F7*m2 + 1))/(km4 + m2) - (K3*g3*m3*m31*(F4*m32 + 1)*((3600*(km3 + m3*m31)*(F5*m2 + 1)*(F6*m15 + 1)*((K2*g2*m1*((F1*((K2*g2*m1)/(3600*(km1 + m1)) - (K2*g2*m2)/(3600*(km2 + m2))))/((F1*K2*g2*m1)/(3600*(km1 + m1)) - (F2*K2*g2*m2)/(3600*(km2 + m2))) - 1))/(3600*(km1 + m1)) - (K2*g2*m2*((F2*((K2*g2*m1)/(3600*(km1 + m1)) - (K2*g2*m2)/(3600*(km2 + m2))))/((F1*K2*g2*m1)/(3600*(km1 + m1)) - (F2*K2*g2*m2)/(3600*(km2 + m2))) - 1))/(3600*(km2 + m2)) + (K4*g4*m2*(F7*m2 + 1))/(3600*(km4 + m2)) + (K5*g5*m2)/(3600*(km5 + m2)*(F8*m5 + 1)*(F9*m25 + 1)*(F10*m30 + 1)) - (K5*g5*m4)/(3600*(km6 + m4)*(F11*m5 + 1)*(F12*m25 + 1)*(F13*m30 + 1)) - (K3*g3*m3*m31*(F4*m32 + 1))/(3600*(km3 + m3*m31)*(F5*m2 + 1)*(F6*m15 + 1)) + (K31*g31*km40*km42*m2*(2*K31*g31*km42*m2 - 3600*((4*K31^2*g31^2*km42^2*m2^2*m26^2 + K33^2*g33^2*km40^2*m25^2*m26^2 + 4*K31^2*g31^2*km42^2*km43*km44*m2^2 + K33^2*g33^2*km40^2*km43*km44*m25^2 + 4*K31^2*g31^2*km42^2*km43*m2^2*m26 + 4*K31^2*g31^2*km42^2*km44*m2^2*m26 + K33^2*g33^2*km40^2*km43*m25^2*m26 + K33^2*g33^2*km40^2*km44*m25^2*m26 + 8*K31*K34*g31*g34*km42^2*m2^2*m26^2 - 8*K31*K35*g31*g35*km42^2*m2^2*m26^2 + 4*K33*K34*g33*g34*km40^2*m25^2*m26^2 - 4*K33*K35*g33*g35*km40^2*m25^2*m26^2 + F52^2*K33^2*g33^2*km40^2*m25^2*m26^2*m42^2 + 2*F52*K33^2*g33^2*km40^2*m25^2*m26^2*m42 + 8*K31*K34*g31*g34*km42^2*km44*m2^2*m26 - 8*K31*K35*g31*g35*km42^2*km43*m2^2*m26 + 4*K33*K34*g33*g34*km40^2*km44*m25^2*m26 - 4*K33*K35*g33*g35*km40^2*km43*m25^2*m26 + F52^2*K33^2*g33^2*km40^2*km43*km44*m25^2*m42^2 + F52^2*K33^2*g33^2*km40^2*km43*m25^2*m26*m42^2 + F52^2*K33^2*g33^2*km40^2*km44*m25^2*m26*m42^2 + 2*F52*K33^2*g33^2*km40^2*km43*km44*m25^2*m42 + 2*F52*K33^2*g33^2*km40^2*km43*m25^2*m26*m42 + 2*F52*K33^2*g33^2*km40^2*km44*m25^2*m26*m42 + 4*K31*K33*g31*g33*km40*km42*m2*m25*m26^2 + 4*F52^2*K33*K34*g33*g34*km40^2*m25^2*m26^2*m42^2 - 4*F52^2*K33*K35*g33*g35*km40^2*m25^2*m26^2*m42^2 + 8*F52*K31*K34*g31*g34*km42^2*m2^2*m26^2*m42 - 8*F52*K31*K35*g31*g35*km42^2*m2^2*m26^2*m42 + 8*F52*K33*K34*g33*g34*km40^2*m25^2*m26^2*m42 - 8*F52*K33*K35*g33*g35*km40^2*m25^2*m26^2*m42 + 4*K31*K33*g31*g33*km40*km42*km43*km44*m2*m25 + 4*K31*K33*g31*g33*km40*km42*km43*m2*m25*m26 + 4*K31*K33*g31*g33*km40*km42*km44*m2*m25*m26 + 4*F52^2*K33*K34*g33*g34*km40^2*km44*m25^2*m26*m42^2 - 4*F52^2*K33*K35*g33*g35*km40^2*km43*m25^2*m26*m42^2 + 8*F52*K31*K34*g31*g34*km42^2*km44*m2^2*m26*m42 - 8*F52*K31*K35*g31*g35*km42^2*km43*m2^2*m26*m42 + 8*F52*K33*K34*g33*g34*km40^2*km44*m25^2*m26*m42 - 8*F52*K33*K35*g33*g35*km40^2*km43*m25^2*m26*m42 + 4*F52*K31*K33*g31*g33*km40*km42*m2*m25*m26^2*m42 + 4*F52*K31*K33*g31*g33*km40*km42*km43*km44*m2*m25*m42 + 4*F52*K31*K33*g31*g33*km40*km42*km43*m2*m25*m26*m42 + 4*F52*K31*K33*g31*g33*km40*km42*km44*m2*m25*m26*m42)/(12960000*(km43 + m26)*(km44 + m26)))^(1/2) + K33*g33*km40*m25 + F52*K33*g33*km40*m25*m42))/(7200*(F52*m42 + 1)*(km40 + (km40*km42*m2*(2*K31*g31*km42*m2 - 3600*((4*K31^2*g31^2*km42^2*m2^2*m26^2 + K33^2*g33^2*km40^2*m25^2*m26^2 + 4*K31^2*g31^2*km42^2*km43*km44*m2^2 + K33^2*g33^2*km40^2*km43*km44*m25^2 + 4*K31^2*g31^2*km42^2*km43*m2^2*m26 + 4*K31^2*g31^2*km42^2*km44*m2^2*m26 + K33^2*g33^2*km40^2*km43*m25^2*m26 + K33^2*g33^2*km40^2*km44*m25^2*m26 + 8*K31*K34*g31*g34*km42^2*m2^2*m26^2 - 8*K31*K35*g31*g35*km42^2*m2^2*m26^2 + 4*K33*K34*g33*g34*km40^2*m25^2*m26^2 - 4*K33*K35*g33*g35*km40^2*m25^2*m26^2 + F52^2*K33^2*g33^2*km40^2*m25^2*m26^2*m42^2 + 2*F52*K33^2*g33^2*km40^2*m25^2*m26^2*m42 + 8*K31*K34*g31*g34*km42^2*km44*m2^2*m26 - 8*K31*K35*g31*g35*km42^2*km43*m2^2*m26 + 4*K33*K34*g33*g34*km40^2*km44*m25^2*m26 - 4*K33*K35*g33*g35*km40^2*km43*m25^2*m26 + F52^2*K33^2*g33^2*km40^2*km43*km44*m25^2*m42^2 + F52^2*K33^2*g33^2*km40^2*km43*m25^2*m26*m42^2 + F52^2*K33^2*g33^2*km40^2*km44*m25^2*m26*m42^2 + 2*F52*K33^2*g33^2*km40^2*km43*km44*m25^2*m42 + 2*F52*K33^2*g33^2*km40^2*km43*m25^2*m26*m42 + 2*F52*K33^2*g33^2*km40^2*km44*m25^2*m26*m42 + 4*K31*K33*g31*g33*km40*km42*m2*m25*m26^2 + 4*F52^2*K33*K34*g33*g34*km40^2*m25^2*m26^2*m42^2 - 4*F52^2*K33*K35*g33*g35*km40^2*m25^2*m26^2*m42^2 + 8*F52*K31*K34*g31*g34*km42^2*m2^2*m26^2*m42 - 8*F52*K31*K35*g31*g35*km42^2*m2^2*m26^2*m42 + 8*F52*K33*K34*g33*g34*km40^2*m25^2*m26^2*m42 - 8*F52*K33*K35*g33*g35*km40^2*m25^2*m26^2*m42 + 4*K31*K33*g31*g33*km40*km42*km43*km44*m2*m25 + 4*K31*K33*g31*g33*km40*km42*km43*m2*m25*m26 + 4*K31*K33*g31*g33*km40*km42*km44*m2*m25*m26 + 4*F52^2*K33*K34*g33*g34*km40^2*km44*m25^2*m26*m42^2 - 4*F52^2*K33*K35*g33*g35*km40^2*km43*m25^2*m26*m42^2 + 8*F52*K31*K34*g31*g34*km42^2*km44*m2^2*m26*m42 - 8*F52*K31*K35*g31*g35*km42^2*km43*m2^2*m26*m42 + 8*F52*K33*K34*g33*g34*km40^2*km44*m25^2*m26*m42 - 8*F52*K33*K35*g33*g35*km40^2*km43*m25^2*m26*m42 + 4*F52*K31*K33*g31*g33*km40*km42*m2*m25*m26^2*m42 + 4*F52*K31*K33*g31*g33*km40*km42*km43*km44*m2*m25*m42 + 4*F52*K31*K33*g31*g33*km40*km42*km43*m2*m25*m26*m42 + 4*F52*K31*K33*g31*g33*km40*km42*km44*m2*m25*m26*m42)/(12960000*(km43 + m26)*(km44 + m26)))^(1/2) + K33*g33*km40*m25 + F52*K33*g33*km40*m25*m42))/(2*(2*K31*g31*km42^2*m2^2 + K33*g33*km40^2*m25^2 + F52*K33*g33*km40^2*m25^2*m42)))*(2*K31*g31*km42^2*m2^2 + K33*g33*km40^2*m25^2 + F52*K33*g33*km40^2*m25^2*m42))))/(K3*g3*m3*m31*(F4*m32 + 1)) + 1))/((km3 + m3*m31)*(F5*m2 + 1)*(F6*m15 + 1)))/(K1*g1) + 1));
u(18) =  (km40*km42*(2*K31*g31*km42*m2 - 3600*((4*K31^2*g31^2*km42^2*m2^2*m26^2 + K33^2*g33^2*km40^2*m25^2*m26^2 + 4*K31^2*g31^2*km42^2*km43*km44*m2^2 + K33^2*g33^2*km40^2*km43*km44*m25^2 + 4*K31^2*g31^2*km42^2*km43*m2^2*m26 + 4*K31^2*g31^2*km42^2*km44*m2^2*m26 + K33^2*g33^2*km40^2*km43*m25^2*m26 + K33^2*g33^2*km40^2*km44*m25^2*m26 + 8*K31*K34*g31*g34*km42^2*m2^2*m26^2 - 8*K31*K35*g31*g35*km42^2*m2^2*m26^2 + 4*K33*K34*g33*g34*km40^2*m25^2*m26^2 - 4*K33*K35*g33*g35*km40^2*m25^2*m26^2 + F52^2*K33^2*g33^2*km40^2*m25^2*m26^2*m42^2 + 2*F52*K33^2*g33^2*km40^2*m25^2*m26^2*m42 + 8*K31*K34*g31*g34*km42^2*km44*m2^2*m26 - 8*K31*K35*g31*g35*km42^2*km43*m2^2*m26 + 4*K33*K34*g33*g34*km40^2*km44*m25^2*m26 - 4*K33*K35*g33*g35*km40^2*km43*m25^2*m26 + F52^2*K33^2*g33^2*km40^2*km43*km44*m25^2*m42^2 + F52^2*K33^2*g33^2*km40^2*km43*m25^2*m26*m42^2 + F52^2*K33^2*g33^2*km40^2*km44*m25^2*m26*m42^2 + 2*F52*K33^2*g33^2*km40^2*km43*km44*m25^2*m42 + 2*F52*K33^2*g33^2*km40^2*km43*m25^2*m26*m42 + 2*F52*K33^2*g33^2*km40^2*km44*m25^2*m26*m42 + 4*K31*K33*g31*g33*km40*km42*m2*m25*m26^2 + 4*F52^2*K33*K34*g33*g34*km40^2*m25^2*m26^2*m42^2 - 4*F52^2*K33*K35*g33*g35*km40^2*m25^2*m26^2*m42^2 + 8*F52*K31*K34*g31*g34*km42^2*m2^2*m26^2*m42 - 8*F52*K31*K35*g31*g35*km42^2*m2^2*m26^2*m42 + 8*F52*K33*K34*g33*g34*km40^2*m25^2*m26^2*m42 - 8*F52*K33*K35*g33*g35*km40^2*m25^2*m26^2*m42 + 4*K31*K33*g31*g33*km40*km42*km43*km44*m2*m25 + 4*K31*K33*g31*g33*km40*km42*km43*m2*m25*m26 + 4*K31*K33*g31*g33*km40*km42*km44*m2*m25*m26 + 4*F52^2*K33*K34*g33*g34*km40^2*km44*m25^2*m26*m42^2 - 4*F52^2*K33*K35*g33*g35*km40^2*km43*m25^2*m26*m42^2 + 8*F52*K31*K34*g31*g34*km42^2*km44*m2^2*m26*m42 - 8*F52*K31*K35*g31*g35*km42^2*km43*m2^2*m26*m42 + 8*F52*K33*K34*g33*g34*km40^2*km44*m25^2*m26*m42 - 8*F52*K33*K35*g33*g35*km40^2*km43*m25^2*m26*m42 + 4*F52*K31*K33*g31*g33*km40*km42*m2*m25*m26^2*m42 + 4*F52*K31*K33*g31*g33*km40*km42*km43*km44*m2*m25*m42 + 4*F52*K31*K33*g31*g33*km40*km42*km43*m2*m25*m26*m42 + 4*F52*K31*K33*g31*g33*km40*km42*km44*m2*m25*m26*m42)/(12960000*(km43 + m26)*(km44 + m26)))^(1/2) + K33*g33*km40*m25 + F52*K33*g33*km40*m25*m42))/(2*(2*K31*g31*km42^2*m2^2 + K33*g33*km40^2*m25^2 + F52*K33*g33*km40^2*m25^2*m42));
u(19) = (3600*(km3 + m3*m31)*(F5*m2 + 1)*(F6*m15 + 1)*((K2*g2*m1*((F1*((K2*g2*m1)/(3600*(km1 + m1)) - (K2*g2*m2)/(3600*(km2 + m2))))/((F1*K2*g2*m1)/(3600*(km1 + m1)) - (F2*K2*g2*m2)/(3600*(km2 + m2))) - 1))/(3600*(km1 + m1)) - (K2*g2*m2*((F2*((K2*g2*m1)/(3600*(km1 + m1)) - (K2*g2*m2)/(3600*(km2 + m2))))/((F1*K2*g2*m1)/(3600*(km1 + m1)) - (F2*K2*g2*m2)/(3600*(km2 + m2))) - 1))/(3600*(km2 + m2)) + (K4*g4*m2*(F7*m2 + 1))/(3600*(km4 + m2)) + (K5*g5*m2)/(3600*(km5 + m2)*(F8*m5 + 1)*(F9*m25 + 1)*(F10*m30 + 1)) - (K5*g5*m4)/(3600*(km6 + m4)*(F11*m5 + 1)*(F12*m25 + 1)*(F13*m30 + 1)) - (K3*g3*m3*m31*(F4*m32 + 1))/(3600*(km3 + m3*m31)*(F5*m2 + 1)*(F6*m15 + 1)) + (K31*g31*km40*km42*m2*(2*K31*g31*km42*m2 - 3600*((4*K31^2*g31^2*km42^2*m2^2*m26^2 + K33^2*g33^2*km40^2*m25^2*m26^2 + 4*K31^2*g31^2*km42^2*km43*km44*m2^2 + K33^2*g33^2*km40^2*km43*km44*m25^2 + 4*K31^2*g31^2*km42^2*km43*m2^2*m26 + 4*K31^2*g31^2*km42^2*km44*m2^2*m26 + K33^2*g33^2*km40^2*km43*m25^2*m26 + K33^2*g33^2*km40^2*km44*m25^2*m26 + 8*K31*K34*g31*g34*km42^2*m2^2*m26^2 - 8*K31*K35*g31*g35*km42^2*m2^2*m26^2 + 4*K33*K34*g33*g34*km40^2*m25^2*m26^2 - 4*K33*K35*g33*g35*km40^2*m25^2*m26^2 + F52^2*K33^2*g33^2*km40^2*m25^2*m26^2*m42^2 + 2*F52*K33^2*g33^2*km40^2*m25^2*m26^2*m42 + 8*K31*K34*g31*g34*km42^2*km44*m2^2*m26 - 8*K31*K35*g31*g35*km42^2*km43*m2^2*m26 + 4*K33*K34*g33*g34*km40^2*km44*m25^2*m26 - 4*K33*K35*g33*g35*km40^2*km43*m25^2*m26 + F52^2*K33^2*g33^2*km40^2*km43*km44*m25^2*m42^2 + F52^2*K33^2*g33^2*km40^2*km43*m25^2*m26*m42^2 + F52^2*K33^2*g33^2*km40^2*km44*m25^2*m26*m42^2 + 2*F52*K33^2*g33^2*km40^2*km43*km44*m25^2*m42 + 2*F52*K33^2*g33^2*km40^2*km43*m25^2*m26*m42 + 2*F52*K33^2*g33^2*km40^2*km44*m25^2*m26*m42 + 4*K31*K33*g31*g33*km40*km42*m2*m25*m26^2 + 4*F52^2*K33*K34*g33*g34*km40^2*m25^2*m26^2*m42^2 - 4*F52^2*K33*K35*g33*g35*km40^2*m25^2*m26^2*m42^2 + 8*F52*K31*K34*g31*g34*km42^2*m2^2*m26^2*m42 - 8*F52*K31*K35*g31*g35*km42^2*m2^2*m26^2*m42 + 8*F52*K33*K34*g33*g34*km40^2*m25^2*m26^2*m42 - 8*F52*K33*K35*g33*g35*km40^2*m25^2*m26^2*m42 + 4*K31*K33*g31*g33*km40*km42*km43*km44*m2*m25 + 4*K31*K33*g31*g33*km40*km42*km43*m2*m25*m26 + 4*K31*K33*g31*g33*km40*km42*km44*m2*m25*m26 + 4*F52^2*K33*K34*g33*g34*km40^2*km44*m25^2*m26*m42^2 - 4*F52^2*K33*K35*g33*g35*km40^2*km43*m25^2*m26*m42^2 + 8*F52*K31*K34*g31*g34*km42^2*km44*m2^2*m26*m42 - 8*F52*K31*K35*g31*g35*km42^2*km43*m2^2*m26*m42 + 8*F52*K33*K34*g33*g34*km40^2*km44*m25^2*m26*m42 - 8*F52*K33*K35*g33*g35*km40^2*km43*m25^2*m26*m42 + 4*F52*K31*K33*g31*g33*km40*km42*m2*m25*m26^2*m42 + 4*F52*K31*K33*g31*g33*km40*km42*km43*km44*m2*m25*m42 + 4*F52*K31*K33*g31*g33*km40*km42*km43*m2*m25*m26*m42 + 4*F52*K31*K33*g31*g33*km40*km42*km44*m2*m25*m26*m42)/(12960000*(km43 + m26)*(km44 + m26)))^(1/2) + K33*g33*km40*m25 + F52*K33*g33*km40*m25*m42))/(7200*(F52*m42 + 1)*(km40 + (km40*km42*m2*(2*K31*g31*km42*m2 - 3600*((4*K31^2*g31^2*km42^2*m2^2*m26^2 + K33^2*g33^2*km40^2*m25^2*m26^2 + 4*K31^2*g31^2*km42^2*km43*km44*m2^2 + K33^2*g33^2*km40^2*km43*km44*m25^2 + 4*K31^2*g31^2*km42^2*km43*m2^2*m26 + 4*K31^2*g31^2*km42^2*km44*m2^2*m26 + K33^2*g33^2*km40^2*km43*m25^2*m26 + K33^2*g33^2*km40^2*km44*m25^2*m26 + 8*K31*K34*g31*g34*km42^2*m2^2*m26^2 - 8*K31*K35*g31*g35*km42^2*m2^2*m26^2 + 4*K33*K34*g33*g34*km40^2*m25^2*m26^2 - 4*K33*K35*g33*g35*km40^2*m25^2*m26^2 + F52^2*K33^2*g33^2*km40^2*m25^2*m26^2*m42^2 + 2*F52*K33^2*g33^2*km40^2*m25^2*m26^2*m42 + 8*K31*K34*g31*g34*km42^2*km44*m2^2*m26 - 8*K31*K35*g31*g35*km42^2*km43*m2^2*m26 + 4*K33*K34*g33*g34*km40^2*km44*m25^2*m26 - 4*K33*K35*g33*g35*km40^2*km43*m25^2*m26 + F52^2*K33^2*g33^2*km40^2*km43*km44*m25^2*m42^2 + F52^2*K33^2*g33^2*km40^2*km43*m25^2*m26*m42^2 + F52^2*K33^2*g33^2*km40^2*km44*m25^2*m26*m42^2 + 2*F52*K33^2*g33^2*km40^2*km43*km44*m25^2*m42 + 2*F52*K33^2*g33^2*km40^2*km43*m25^2*m26*m42 + 2*F52*K33^2*g33^2*km40^2*km44*m25^2*m26*m42 + 4*K31*K33*g31*g33*km40*km42*m2*m25*m26^2 + 4*F52^2*K33*K34*g33*g34*km40^2*m25^2*m26^2*m42^2 - 4*F52^2*K33*K35*g33*g35*km40^2*m25^2*m26^2*m42^2 + 8*F52*K31*K34*g31*g34*km42^2*m2^2*m26^2*m42 - 8*F52*K31*K35*g31*g35*km42^2*m2^2*m26^2*m42 + 8*F52*K33*K34*g33*g34*km40^2*m25^2*m26^2*m42 - 8*F52*K33*K35*g33*g35*km40^2*m25^2*m26^2*m42 + 4*K31*K33*g31*g33*km40*km42*km43*km44*m2*m25 + 4*K31*K33*g31*g33*km40*km42*km43*m2*m25*m26 + 4*K31*K33*g31*g33*km40*km42*km44*m2*m25*m26 + 4*F52^2*K33*K34*g33*g34*km40^2*km44*m25^2*m26*m42^2 - 4*F52^2*K33*K35*g33*g35*km40^2*km43*m25^2*m26*m42^2 + 8*F52*K31*K34*g31*g34*km42^2*km44*m2^2*m26*m42 - 8*F52*K31*K35*g31*g35*km42^2*km43*m2^2*m26*m42 + 8*F52*K33*K34*g33*g34*km40^2*km44*m25^2*m26*m42 - 8*F52*K33*K35*g33*g35*km40^2*km43*m25^2*m26*m42 + 4*F52*K31*K33*g31*g33*km40*km42*m2*m25*m26^2*m42 + 4*F52*K31*K33*g31*g33*km40*km42*km43*km44*m2*m25*m42 + 4*F52*K31*K33*g31*g33*km40*km42*km43*m2*m25*m26*m42 + 4*F52*K31*K33*g31*g33*km40*km42*km44*m2*m25*m26*m42)/(12960000*(km43 + m26)*(km44 + m26)))^(1/2) + K33*g33*km40*m25 + F52*K33*g33*km40*m25*m42))/(2*(2*K31*g31*km42^2*m2^2 + K33*g33*km40^2*m25^2 + F52*K33*g33*km40^2*m25^2*m42)))*(2*K31*g31*km42^2*m2^2 + K33*g33*km40^2*m25^2 + F52*K33*g33*km40^2*m25^2*m42))))/(F3*K3*g3*m3*m31*(F4*m32 + 1));
u(20) = (3600*(km29 + m16*m31)*((3600*F35*(km3 + m3*m31)*(F5*m2 + 1)*(F6*m15 + 1)*((K2*g2*m1*((F1*((K2*g2*m1)/(3600*(km1 + m1)) - (K2*g2*m2)/(3600*(km2 + m2))))/((F1*K2*g2*m1)/(3600*(km1 + m1)) - (F2*K2*g2*m2)/(3600*(km2 + m2))) - 1))/(3600*(km1 + m1)) - (K2*g2*m2*((F2*((K2*g2*m1)/(3600*(km1 + m1)) - (K2*g2*m2)/(3600*(km2 + m2))))/((F1*K2*g2*m1)/(3600*(km1 + m1)) - (F2*K2*g2*m2)/(3600*(km2 + m2))) - 1))/(3600*(km2 + m2)) + (K4*g4*m2*(F7*m2 + 1))/(3600*(km4 + m2)) + (K5*g5*m2)/(3600*(km5 + m2)*(F8*m5 + 1)*(F9*m25 + 1)*(F10*m30 + 1)) - (K5*g5*m4)/(3600*(km6 + m4)*(F11*m5 + 1)*(F12*m25 + 1)*(F13*m30 + 1)) - (K3*g3*m3*m31*(F4*m32 + 1))/(3600*(km3 + m3*m31)*(F5*m2 + 1)*(F6*m15 + 1)) + (K31*g31*km40*km42*m2*(2*K31*g31*km42*m2 - 3600*((4*K31^2*g31^2*km42^2*m2^2*m26^2 + K33^2*g33^2*km40^2*m25^2*m26^2 + 4*K31^2*g31^2*km42^2*km43*km44*m2^2 + K33^2*g33^2*km40^2*km43*km44*m25^2 + 4*K31^2*g31^2*km42^2*km43*m2^2*m26 + 4*K31^2*g31^2*km42^2*km44*m2^2*m26 + K33^2*g33^2*km40^2*km43*m25^2*m26 + K33^2*g33^2*km40^2*km44*m25^2*m26 + 8*K31*K34*g31*g34*km42^2*m2^2*m26^2 - 8*K31*K35*g31*g35*km42^2*m2^2*m26^2 + 4*K33*K34*g33*g34*km40^2*m25^2*m26^2 - 4*K33*K35*g33*g35*km40^2*m25^2*m26^2 + F52^2*K33^2*g33^2*km40^2*m25^2*m26^2*m42^2 + 2*F52*K33^2*g33^2*km40^2*m25^2*m26^2*m42 + 8*K31*K34*g31*g34*km42^2*km44*m2^2*m26 - 8*K31*K35*g31*g35*km42^2*km43*m2^2*m26 + 4*K33*K34*g33*g34*km40^2*km44*m25^2*m26 - 4*K33*K35*g33*g35*km40^2*km43*m25^2*m26 + F52^2*K33^2*g33^2*km40^2*km43*km44*m25^2*m42^2 + F52^2*K33^2*g33^2*km40^2*km43*m25^2*m26*m42^2 + F52^2*K33^2*g33^2*km40^2*km44*m25^2*m26*m42^2 + 2*F52*K33^2*g33^2*km40^2*km43*km44*m25^2*m42 + 2*F52*K33^2*g33^2*km40^2*km43*m25^2*m26*m42 + 2*F52*K33^2*g33^2*km40^2*km44*m25^2*m26*m42 + 4*K31*K33*g31*g33*km40*km42*m2*m25*m26^2 + 4*F52^2*K33*K34*g33*g34*km40^2*m25^2*m26^2*m42^2 - 4*F52^2*K33*K35*g33*g35*km40^2*m25^2*m26^2*m42^2 + 8*F52*K31*K34*g31*g34*km42^2*m2^2*m26^2*m42 - 8*F52*K31*K35*g31*g35*km42^2*m2^2*m26^2*m42 + 8*F52*K33*K34*g33*g34*km40^2*m25^2*m26^2*m42 - 8*F52*K33*K35*g33*g35*km40^2*m25^2*m26^2*m42 + 4*K31*K33*g31*g33*km40*km42*km43*km44*m2*m25 + 4*K31*K33*g31*g33*km40*km42*km43*m2*m25*m26 + 4*K31*K33*g31*g33*km40*km42*km44*m2*m25*m26 + 4*F52^2*K33*K34*g33*g34*km40^2*km44*m25^2*m26*m42^2 - 4*F52^2*K33*K35*g33*g35*km40^2*km43*m25^2*m26*m42^2 + 8*F52*K31*K34*g31*g34*km42^2*km44*m2^2*m26*m42 - 8*F52*K31*K35*g31*g35*km42^2*km43*m2^2*m26*m42 + 8*F52*K33*K34*g33*g34*km40^2*km44*m25^2*m26*m42 - 8*F52*K33*K35*g33*g35*km40^2*km43*m25^2*m26*m42 + 4*F52*K31*K33*g31*g33*km40*km42*m2*m25*m26^2*m42 + 4*F52*K31*K33*g31*g33*km40*km42*km43*km44*m2*m25*m42 + 4*F52*K31*K33*g31*g33*km40*km42*km43*m2*m25*m26*m42 + 4*F52*K31*K33*g31*g33*km40*km42*km44*m2*m25*m26*m42)/(12960000*(km43 + m26)*(km44 + m26)))^(1/2) + K33*g33*km40*m25 + F52*K33*g33*km40*m25*m42))/(7200*(F52*m42 + 1)*(km40 + (km40*km42*m2*(2*K31*g31*km42*m2 - 3600*((4*K31^2*g31^2*km42^2*m2^2*m26^2 + K33^2*g33^2*km40^2*m25^2*m26^2 + 4*K31^2*g31^2*km42^2*km43*km44*m2^2 + K33^2*g33^2*km40^2*km43*km44*m25^2 + 4*K31^2*g31^2*km42^2*km43*m2^2*m26 + 4*K31^2*g31^2*km42^2*km44*m2^2*m26 + K33^2*g33^2*km40^2*km43*m25^2*m26 + K33^2*g33^2*km40^2*km44*m25^2*m26 + 8*K31*K34*g31*g34*km42^2*m2^2*m26^2 - 8*K31*K35*g31*g35*km42^2*m2^2*m26^2 + 4*K33*K34*g33*g34*km40^2*m25^2*m26^2 - 4*K33*K35*g33*g35*km40^2*m25^2*m26^2 + F52^2*K33^2*g33^2*km40^2*m25^2*m26^2*m42^2 + 2*F52*K33^2*g33^2*km40^2*m25^2*m26^2*m42 + 8*K31*K34*g31*g34*km42^2*km44*m2^2*m26 - 8*K31*K35*g31*g35*km42^2*km43*m2^2*m26 + 4*K33*K34*g33*g34*km40^2*km44*m25^2*m26 - 4*K33*K35*g33*g35*km40^2*km43*m25^2*m26 + F52^2*K33^2*g33^2*km40^2*km43*km44*m25^2*m42^2 + F52^2*K33^2*g33^2*km40^2*km43*m25^2*m26*m42^2 + F52^2*K33^2*g33^2*km40^2*km44*m25^2*m26*m42^2 + 2*F52*K33^2*g33^2*km40^2*km43*km44*m25^2*m42 + 2*F52*K33^2*g33^2*km40^2*km43*m25^2*m26*m42 + 2*F52*K33^2*g33^2*km40^2*km44*m25^2*m26*m42 + 4*K31*K33*g31*g33*km40*km42*m2*m25*m26^2 + 4*F52^2*K33*K34*g33*g34*km40^2*m25^2*m26^2*m42^2 - 4*F52^2*K33*K35*g33*g35*km40^2*m25^2*m26^2*m42^2 + 8*F52*K31*K34*g31*g34*km42^2*m2^2*m26^2*m42 - 8*F52*K31*K35*g31*g35*km42^2*m2^2*m26^2*m42 + 8*F52*K33*K34*g33*g34*km40^2*m25^2*m26^2*m42 - 8*F52*K33*K35*g33*g35*km40^2*m25^2*m26^2*m42 + 4*K31*K33*g31*g33*km40*km42*km43*km44*m2*m25 + 4*K31*K33*g31*g33*km40*km42*km43*m2*m25*m26 + 4*K31*K33*g31*g33*km40*km42*km44*m2*m25*m26 + 4*F52^2*K33*K34*g33*g34*km40^2*km44*m25^2*m26*m42^2 - 4*F52^2*K33*K35*g33*g35*km40^2*km43*m25^2*m26*m42^2 + 8*F52*K31*K34*g31*g34*km42^2*km44*m2^2*m26*m42 - 8*F52*K31*K35*g31*g35*km42^2*km43*m2^2*m26*m42 + 8*F52*K33*K34*g33*g34*km40^2*km44*m25^2*m26*m42 - 8*F52*K33*K35*g33*g35*km40^2*km43*m25^2*m26*m42 + 4*F52*K31*K33*g31*g33*km40*km42*m2*m25*m26^2*m42 + 4*F52*K31*K33*g31*g33*km40*km42*km43*km44*m2*m25*m42 + 4*F52*K31*K33*g31*g33*km40*km42*km43*m2*m25*m26*m42 + 4*F52*K31*K33*g31*g33*km40*km42*km44*m2*m25*m26*m42)/(12960000*(km43 + m26)*(km44 + m26)))^(1/2) + K33*g33*km40*m25 + F52*K33*g33*km40*m25*m42))/(2*(2*K31*g31*km42^2*m2^2 + K33*g33*km40^2*m25^2 + F52*K33*g33*km40^2*m25^2*m42)))*(2*K31*g31*km42^2*m2^2 + K33*g33*km40^2*m25^2 + F52*K33*g33*km40^2*m25^2*m42))))/(F3*K3*g3*m3*m31*(F4*m32 + 1)) + 1)*((K30*g30*m23*m34)/(3600*(km39 + m23*m34)*(F51*m16 + 1)) - (K22*g22*m16*m31)/(3600*(km29 + m16*m31)*((3600*F35*(km3 + m3*m31)*(F5*m2 + 1)*(F6*m15 + 1)*((K2*g2*m1*((F1*((K2*g2*m1)/(3600*(km1 + m1)) - (K2*g2*m2)/(3600*(km2 + m2))))/((F1*K2*g2*m1)/(3600*(km1 + m1)) - (F2*K2*g2*m2)/(3600*(km2 + m2))) - 1))/(3600*(km1 + m1)) - (K2*g2*m2*((F2*((K2*g2*m1)/(3600*(km1 + m1)) - (K2*g2*m2)/(3600*(km2 + m2))))/((F1*K2*g2*m1)/(3600*(km1 + m1)) - (F2*K2*g2*m2)/(3600*(km2 + m2))) - 1))/(3600*(km2 + m2)) + (K4*g4*m2*(F7*m2 + 1))/(3600*(km4 + m2)) + (K5*g5*m2)/(3600*(km5 + m2)*(F8*m5 + 1)*(F9*m25 + 1)*(F10*m30 + 1)) - (K5*g5*m4)/(3600*(km6 + m4)*(F11*m5 + 1)*(F12*m25 + 1)*(F13*m30 + 1)) - (K3*g3*m3*m31*(F4*m32 + 1))/(3600*(km3 + m3*m31)*(F5*m2 + 1)*(F6*m15 + 1)) + (K31*g31*km40*km42*m2*(2*K31*g31*km42*m2 - 3600*((4*K31^2*g31^2*km42^2*m2^2*m26^2 + K33^2*g33^2*km40^2*m25^2*m26^2 + 4*K31^2*g31^2*km42^2*km43*km44*m2^2 + K33^2*g33^2*km40^2*km43*km44*m25^2 + 4*K31^2*g31^2*km42^2*km43*m2^2*m26 + 4*K31^2*g31^2*km42^2*km44*m2^2*m26 + K33^2*g33^2*km40^2*km43*m25^2*m26 + K33^2*g33^2*km40^2*km44*m25^2*m26 + 8*K31*K34*g31*g34*km42^2*m2^2*m26^2 - 8*K31*K35*g31*g35*km42^2*m2^2*m26^2 + 4*K33*K34*g33*g34*km40^2*m25^2*m26^2 - 4*K33*K35*g33*g35*km40^2*m25^2*m26^2 + F52^2*K33^2*g33^2*km40^2*m25^2*m26^2*m42^2 + 2*F52*K33^2*g33^2*km40^2*m25^2*m26^2*m42 + 8*K31*K34*g31*g34*km42^2*km44*m2^2*m26 - 8*K31*K35*g31*g35*km42^2*km43*m2^2*m26 + 4*K33*K34*g33*g34*km40^2*km44*m25^2*m26 - 4*K33*K35*g33*g35*km40^2*km43*m25^2*m26 + F52^2*K33^2*g33^2*km40^2*km43*km44*m25^2*m42^2 + F52^2*K33^2*g33^2*km40^2*km43*m25^2*m26*m42^2 + F52^2*K33^2*g33^2*km40^2*km44*m25^2*m26*m42^2 + 2*F52*K33^2*g33^2*km40^2*km43*km44*m25^2*m42 + 2*F52*K33^2*g33^2*km40^2*km43*m25^2*m26*m42 + 2*F52*K33^2*g33^2*km40^2*km44*m25^2*m26*m42 + 4*K31*K33*g31*g33*km40*km42*m2*m25*m26^2 + 4*F52^2*K33*K34*g33*g34*km40^2*m25^2*m26^2*m42^2 - 4*F52^2*K33*K35*g33*g35*km40^2*m25^2*m26^2*m42^2 + 8*F52*K31*K34*g31*g34*km42^2*m2^2*m26^2*m42 - 8*F52*K31*K35*g31*g35*km42^2*m2^2*m26^2*m42 + 8*F52*K33*K34*g33*g34*km40^2*m25^2*m26^2*m42 - 8*F52*K33*K35*g33*g35*km40^2*m25^2*m26^2*m42 + 4*K31*K33*g31*g33*km40*km42*km43*km44*m2*m25 + 4*K31*K33*g31*g33*km40*km42*km43*m2*m25*m26 + 4*K31*K33*g31*g33*km40*km42*km44*m2*m25*m26 + 4*F52^2*K33*K34*g33*g34*km40^2*km44*m25^2*m26*m42^2 - 4*F52^2*K33*K35*g33*g35*km40^2*km43*m25^2*m26*m42^2 + 8*F52*K31*K34*g31*g34*km42^2*km44*m2^2*m26*m42 - 8*F52*K31*K35*g31*g35*km42^2*km43*m2^2*m26*m42 + 8*F52*K33*K34*g33*g34*km40^2*km44*m25^2*m26*m42 - 8*F52*K33*K35*g33*g35*km40^2*km43*m25^2*m26*m42 + 4*F52*K31*K33*g31*g33*km40*km42*m2*m25*m26^2*m42 + 4*F52*K31*K33*g31*g33*km40*km42*km43*km44*m2*m25*m42 + 4*F52*K31*K33*g31*g33*km40*km42*km43*m2*m25*m26*m42 + 4*F52*K31*K33*g31*g33*km40*km42*km44*m2*m25*m26*m42)/(12960000*(km43 + m26)*(km44 + m26)))^(1/2) + K33*g33*km40*m25 + F52*K33*g33*km40*m25*m42))/(7200*(F52*m42 + 1)*(km40 + (km40*km42*m2*(2*K31*g31*km42*m2 - 3600*((4*K31^2*g31^2*km42^2*m2^2*m26^2 + K33^2*g33^2*km40^2*m25^2*m26^2 + 4*K31^2*g31^2*km42^2*km43*km44*m2^2 + K33^2*g33^2*km40^2*km43*km44*m25^2 + 4*K31^2*g31^2*km42^2*km43*m2^2*m26 + 4*K31^2*g31^2*km42^2*km44*m2^2*m26 + K33^2*g33^2*km40^2*km43*m25^2*m26 + K33^2*g33^2*km40^2*km44*m25^2*m26 + 8*K31*K34*g31*g34*km42^2*m2^2*m26^2 - 8*K31*K35*g31*g35*km42^2*m2^2*m26^2 + 4*K33*K34*g33*g34*km40^2*m25^2*m26^2 - 4*K33*K35*g33*g35*km40^2*m25^2*m26^2 + F52^2*K33^2*g33^2*km40^2*m25^2*m26^2*m42^2 + 2*F52*K33^2*g33^2*km40^2*m25^2*m26^2*m42 + 8*K31*K34*g31*g34*km42^2*km44*m2^2*m26 - 8*K31*K35*g31*g35*km42^2*km43*m2^2*m26 + 4*K33*K34*g33*g34*km40^2*km44*m25^2*m26 - 4*K33*K35*g33*g35*km40^2*km43*m25^2*m26 + F52^2*K33^2*g33^2*km40^2*km43*km44*m25^2*m42^2 + F52^2*K33^2*g33^2*km40^2*km43*m25^2*m26*m42^2 + F52^2*K33^2*g33^2*km40^2*km44*m25^2*m26*m42^2 + 2*F52*K33^2*g33^2*km40^2*km43*km44*m25^2*m42 + 2*F52*K33^2*g33^2*km40^2*km43*m25^2*m26*m42 + 2*F52*K33^2*g33^2*km40^2*km44*m25^2*m26*m42 + 4*K31*K33*g31*g33*km40*km42*m2*m25*m26^2 + 4*F52^2*K33*K34*g33*g34*km40^2*m25^2*m26^2*m42^2 - 4*F52^2*K33*K35*g33*g35*km40^2*m25^2*m26^2*m42^2 + 8*F52*K31*K34*g31*g34*km42^2*m2^2*m26^2*m42 - 8*F52*K31*K35*g31*g35*km42^2*m2^2*m26^2*m42 + 8*F52*K33*K34*g33*g34*km40^2*m25^2*m26^2*m42 - 8*F52*K33*K35*g33*g35*km40^2*m25^2*m26^2*m42 + 4*K31*K33*g31*g33*km40*km42*km43*km44*m2*m25 + 4*K31*K33*g31*g33*km40*km42*km43*m2*m25*m26 + 4*K31*K33*g31*g33*km40*km42*km44*m2*m25*m26 + 4*F52^2*K33*K34*g33*g34*km40^2*km44*m25^2*m26*m42^2 - 4*F52^2*K33*K35*g33*g35*km40^2*km43*m25^2*m26*m42^2 + 8*F52*K31*K34*g31*g34*km42^2*km44*m2^2*m26*m42 - 8*F52*K31*K35*g31*g35*km42^2*km43*m2^2*m26*m42 + 8*F52*K33*K34*g33*g34*km40^2*km44*m25^2*m26*m42 - 8*F52*K33*K35*g33*g35*km40^2*km43*m25^2*m26*m42 + 4*F52*K31*K33*g31*g33*km40*km42*m2*m25*m26^2*m42 + 4*F52*K31*K33*g31*g33*km40*km42*km43*km44*m2*m25*m42 + 4*F52*K31*K33*g31*g33*km40*km42*km43*m2*m25*m26*m42 + 4*F52*K31*K33*g31*g33*km40*km42*km44*m2*m25*m26*m42)/(12960000*(km43 + m26)*(km44 + m26)))^(1/2) + K33*g33*km40*m25 + F52*K33*g33*km40*m25*m42))/(2*(2*K31*g31*km42^2*m2^2 + K33*g33*km40^2*m25^2 + F52*K33*g33*km40^2*m25^2*m42)))*(2*K31*g31*km42^2*m2^2 + K33*g33*km40^2*m25^2 + F52*K33*g33*km40^2*m25^2*m42))))/(F3*K3*g3*m3*m31*(F4*m32 + 1)) + 1)) + (K21*g21*m13*m31*(F33*m15 + 1))/(3600*(km28 + m13*m31)) - (K23*g23*m15*m16*(F36*m15 + 1)*(F37*m16 + 1))/(3600*(km30 + m15*m16)*(F39*m15 + 1)*(F38*m31 + 1))))/(F34*K22*g22*m16*m31);
u(21) = -((K27*g27*m21)/(3600*(km36 + m21)) - 3*((K27*g27*m20*m32)/(3600*(km35 + m20*m32))) + (K18*g18*m13*m34*m35)/(3600*(km25 + m13*m34*m35)) - (K20*g20*m15*m32*m33*m40)/(3600*(km27 + m15*m32*m33*m40)) + (K19*g19*m31*m34*m37*m38)/(3600*(km26 + m31*m34*m37*m38)) - (K26*g26*m19*m34*m41)/(3600*(km34 + m19*m34*m41)*(F47*m20 + 1)*(F46*m31 + 1)) + (K25*g25*m18*m34*(F42*m18 + 1)*(F43*m32 + 1))/(3600*(km33 + m18*m34)*(F44*m31 + 1)*(F45*m33 + 1)))/((F53*K18*g18*m13*m34*m35)/(3600*(km25 + m13*m34*m35)) - (F55*K26*g26*m19*m34*m41)/(3600*(km34 + m19*m34*m41)*(F47*m20 + 1)*(F46*m31 + 1)) + (F54*K25*g25*m18*m34*(F42*m18 + 1)*(F43*m32 + 1))/(3600*(km33 + m18*m34)*(F44*m31 + 1)*(F45*m33 + 1)));
u(22) = -((K24*g24*m18)/(3600*km32 + 3600*m18) - (K24*g24*m17)/(3600*km31 + 3600*m17) + (K23*g23*m15*m16*(F36*m15 + 1)*(F37*m16 + 1))/((3600*km30 + 3600*m15*m16)*(F39*m15 + 1)*(F38*m31 + 1)))/((F40*K24*g24*m17)/(3600*km31 + 3600*m17) - (F41*K24*g24*m18)/(3600*km32 + 3600*m18));
u(23) = ((3600*km37 + 3600*m21*m38)*((K27*g27*m21)/(3600*km36 + 3600*m21) - (K29*g29*m22)/(3600*km38 + 3600*m22) - 3*((K27*g27*m20*m32)/(3600*km35 + 3600*m20*m32)) + (4*K28*g28*m21*m38)/(3600*km37 + 3600*m21*m38) - (2*K20*g20*m15*m32*m33*m40)/(3600*km27 + 3600*m15*m32*m33*m40) + (2*K19*g19*m31*m34*m37*m38)/(3600*km26 + 3600*m31*m34*m37*m38)))/(4*F48*K28*g28*m21*m38);
u(24) = ((KS27*s5)/(60*(F_S30*s28 + 1)) - (KS26*m31)/(60*(F_S29*s29 + 1)) + (KS5*s4*s28)/(60*(F_S4*s15 + 1)) + (K10*g10*m5)/(3600*km11 + 3600*m5) + (K7*g7*m5)/((F19*m6 + 1)*(3600*km8 + 3600*m5)) - (K10*g10*m7*m8)/(3600*km12 + 3600*m7*m8) - (K6*g6*m4*m31*(F15*m6 + 1))/((3600*km7 + 3600*m4*m31)*(F17*m17 + 1)*(F16*m31 + 1)))/(KS16/60 + (F18*K7*g7*m5)/((F19*m6 + 1)*(3600*km8 + 3600*m5)) + (F14_modified*K6*g6*m4*m31*(F15*m6 + 1))/((3600*km7 + 3600*m4*m31)*(F17*m17 + 1)*(F16*m31 + 1)));
u(25) = ((K16*g16*m12*m32*(F27*m5 + 1)*((3600*F26*(km3 + m3*m31)*(F5*m2 + 1)*(F6*m15 + 1)*((K2*g2*m1*((F1*((K2*g2*m1)/(3600*(km1 + m1)) - (K2*g2*m2)/(3600*(km2 + m2))))/((F1*K2*g2*m1)/(3600*(km1 + m1)) - (F2*K2*g2*m2)/(3600*(km2 + m2))) - 1))/(3600*(km1 + m1)) - (K2*g2*m2*((F2*((K2*g2*m1)/(3600*(km1 + m1)) - (K2*g2*m2)/(3600*(km2 + m2))))/((F1*K2*g2*m1)/(3600*(km1 + m1)) - (F2*K2*g2*m2)/(3600*(km2 + m2))) - 1))/(3600*(km2 + m2)) + (K4*g4*m2*(F7*m2 + 1))/(3600*(km4 + m2)) + (K5*g5*m2)/(3600*(km5 + m2)*(F8*m5 + 1)*(F9*m25 + 1)*(F10*m30 + 1)) - (K5*g5*m4)/(3600*(km6 + m4)*(F11*m5 + 1)*(F12*m25 + 1)*(F13*m30 + 1)) - (K3*g3*m3*m31*(F4*m32 + 1))/(3600*(km3 + m3*m31)*(F5*m2 + 1)*(F6*m15 + 1)) + (K31*g31*km40*km42*m2*(2*K31*g31*km42*m2 - 3600*((4*K31^2*g31^2*km42^2*m2^2*m26^2 + K33^2*g33^2*km40^2*m25^2*m26^2 + 4*K31^2*g31^2*km42^2*km43*km44*m2^2 + K33^2*g33^2*km40^2*km43*km44*m25^2 + 4*K31^2*g31^2*km42^2*km43*m2^2*m26 + 4*K31^2*g31^2*km42^2*km44*m2^2*m26 + K33^2*g33^2*km40^2*km43*m25^2*m26 + K33^2*g33^2*km40^2*km44*m25^2*m26 + 8*K31*K34*g31*g34*km42^2*m2^2*m26^2 - 8*K31*K35*g31*g35*km42^2*m2^2*m26^2 + 4*K33*K34*g33*g34*km40^2*m25^2*m26^2 - 4*K33*K35*g33*g35*km40^2*m25^2*m26^2 + F52^2*K33^2*g33^2*km40^2*m25^2*m26^2*m42^2 + 2*F52*K33^2*g33^2*km40^2*m25^2*m26^2*m42 + 8*K31*K34*g31*g34*km42^2*km44*m2^2*m26 - 8*K31*K35*g31*g35*km42^2*km43*m2^2*m26 + 4*K33*K34*g33*g34*km40^2*km44*m25^2*m26 - 4*K33*K35*g33*g35*km40^2*km43*m25^2*m26 + F52^2*K33^2*g33^2*km40^2*km43*km44*m25^2*m42^2 + F52^2*K33^2*g33^2*km40^2*km43*m25^2*m26*m42^2 + F52^2*K33^2*g33^2*km40^2*km44*m25^2*m26*m42^2 + 2*F52*K33^2*g33^2*km40^2*km43*km44*m25^2*m42 + 2*F52*K33^2*g33^2*km40^2*km43*m25^2*m26*m42 + 2*F52*K33^2*g33^2*km40^2*km44*m25^2*m26*m42 + 4*K31*K33*g31*g33*km40*km42*m2*m25*m26^2 + 4*F52^2*K33*K34*g33*g34*km40^2*m25^2*m26^2*m42^2 - 4*F52^2*K33*K35*g33*g35*km40^2*m25^2*m26^2*m42^2 + 8*F52*K31*K34*g31*g34*km42^2*m2^2*m26^2*m42 - 8*F52*K31*K35*g31*g35*km42^2*m2^2*m26^2*m42 + 8*F52*K33*K34*g33*g34*km40^2*m25^2*m26^2*m42 - 8*F52*K33*K35*g33*g35*km40^2*m25^2*m26^2*m42 + 4*K31*K33*g31*g33*km40*km42*km43*km44*m2*m25 + 4*K31*K33*g31*g33*km40*km42*km43*m2*m25*m26 + 4*K31*K33*g31*g33*km40*km42*km44*m2*m25*m26 + 4*F52^2*K33*K34*g33*g34*km40^2*km44*m25^2*m26*m42^2 - 4*F52^2*K33*K35*g33*g35*km40^2*km43*m25^2*m26*m42^2 + 8*F52*K31*K34*g31*g34*km42^2*km44*m2^2*m26*m42 - 8*F52*K31*K35*g31*g35*km42^2*km43*m2^2*m26*m42 + 8*F52*K33*K34*g33*g34*km40^2*km44*m25^2*m26*m42 - 8*F52*K33*K35*g33*g35*km40^2*km43*m25^2*m26*m42 + 4*F52*K31*K33*g31*g33*km40*km42*m2*m25*m26^2*m42 + 4*F52*K31*K33*g31*g33*km40*km42*km43*km44*m2*m25*m42 + 4*F52*K31*K33*g31*g33*km40*km42*km43*m2*m25*m26*m42 + 4*F52*K31*K33*g31*g33*km40*km42*km44*m2*m25*m26*m42)/(12960000*(km43 + m26)*(km44 + m26)))^(1/2) + K33*g33*km40*m25 + F52*K33*g33*km40*m25*m42))/(7200*(F52*m42 + 1)*(km40 + (km40*km42*m2*(2*K31*g31*km42*m2 - 3600*((4*K31^2*g31^2*km42^2*m2^2*m26^2 + K33^2*g33^2*km40^2*m25^2*m26^2 + 4*K31^2*g31^2*km42^2*km43*km44*m2^2 + K33^2*g33^2*km40^2*km43*km44*m25^2 + 4*K31^2*g31^2*km42^2*km43*m2^2*m26 + 4*K31^2*g31^2*km42^2*km44*m2^2*m26 + K33^2*g33^2*km40^2*km43*m25^2*m26 + K33^2*g33^2*km40^2*km44*m25^2*m26 + 8*K31*K34*g31*g34*km42^2*m2^2*m26^2 - 8*K31*K35*g31*g35*km42^2*m2^2*m26^2 + 4*K33*K34*g33*g34*km40^2*m25^2*m26^2 - 4*K33*K35*g33*g35*km40^2*m25^2*m26^2 + F52^2*K33^2*g33^2*km40^2*m25^2*m26^2*m42^2 + 2*F52*K33^2*g33^2*km40^2*m25^2*m26^2*m42 + 8*K31*K34*g31*g34*km42^2*km44*m2^2*m26 - 8*K31*K35*g31*g35*km42^2*km43*m2^2*m26 + 4*K33*K34*g33*g34*km40^2*km44*m25^2*m26 - 4*K33*K35*g33*g35*km40^2*km43*m25^2*m26 + F52^2*K33^2*g33^2*km40^2*km43*km44*m25^2*m42^2 + F52^2*K33^2*g33^2*km40^2*km43*m25^2*m26*m42^2 + F52^2*K33^2*g33^2*km40^2*km44*m25^2*m26*m42^2 + 2*F52*K33^2*g33^2*km40^2*km43*km44*m25^2*m42 + 2*F52*K33^2*g33^2*km40^2*km43*m25^2*m26*m42 + 2*F52*K33^2*g33^2*km40^2*km44*m25^2*m26*m42 + 4*K31*K33*g31*g33*km40*km42*m2*m25*m26^2 + 4*F52^2*K33*K34*g33*g34*km40^2*m25^2*m26^2*m42^2 - 4*F52^2*K33*K35*g33*g35*km40^2*m25^2*m26^2*m42^2 + 8*F52*K31*K34*g31*g34*km42^2*m2^2*m26^2*m42 - 8*F52*K31*K35*g31*g35*km42^2*m2^2*m26^2*m42 + 8*F52*K33*K34*g33*g34*km40^2*m25^2*m26^2*m42 - 8*F52*K33*K35*g33*g35*km40^2*m25^2*m26^2*m42 + 4*K31*K33*g31*g33*km40*km42*km43*km44*m2*m25 + 4*K31*K33*g31*g33*km40*km42*km43*m2*m25*m26 + 4*K31*K33*g31*g33*km40*km42*km44*m2*m25*m26 + 4*F52^2*K33*K34*g33*g34*km40^2*km44*m25^2*m26*m42^2 - 4*F52^2*K33*K35*g33*g35*km40^2*km43*m25^2*m26*m42^2 + 8*F52*K31*K34*g31*g34*km42^2*km44*m2^2*m26*m42 - 8*F52*K31*K35*g31*g35*km42^2*km43*m2^2*m26*m42 + 8*F52*K33*K34*g33*g34*km40^2*km44*m25^2*m26*m42 - 8*F52*K33*K35*g33*g35*km40^2*km43*m25^2*m26*m42 + 4*F52*K31*K33*g31*g33*km40*km42*m2*m25*m26^2*m42 + 4*F52*K31*K33*g31*g33*km40*km42*km43*km44*m2*m25*m42 + 4*F52*K31*K33*g31*g33*km40*km42*km43*m2*m25*m26*m42 + 4*F52*K31*K33*g31*g33*km40*km42*km44*m2*m25*m26*m42)/(12960000*(km43 + m26)*(km44 + m26)))^(1/2) + K33*g33*km40*m25 + F52*K33*g33*km40*m25*m42))/(2*(2*K31*g31*km42^2*m2^2 + K33*g33*km40^2*m25^2 + F52*K33*g33*km40^2*m25^2*m42)))*(2*K31*g31*km42^2*m2^2 + K33*g33*km40^2*m25^2 + F52*K33*g33*km40^2*m25^2*m42))))/(F3*K3*g3*m3*m31*(F4*m32 + 1)) + 1))/(3600*(km23 + m12*m32)*(F29*m15 + 1)*(F28*m31 + 1)*(F32*m37 + 1)*((3600*F31*(km29 + m16*m31)*((3600*F35*(km3 + m3*m31)*(F5*m2 + 1)*(F6*m15 + 1)*((K2*g2*m1*((F1*((K2*g2*m1)/(3600*(km1 + m1)) - (K2*g2*m2)/(3600*(km2 + m2))))/((F1*K2*g2*m1)/(3600*(km1 + m1)) - (F2*K2*g2*m2)/(3600*(km2 + m2))) - 1))/(3600*(km1 + m1)) - (K2*g2*m2*((F2*((K2*g2*m1)/(3600*(km1 + m1)) - (K2*g2*m2)/(3600*(km2 + m2))))/((F1*K2*g2*m1)/(3600*(km1 + m1)) - (F2*K2*g2*m2)/(3600*(km2 + m2))) - 1))/(3600*(km2 + m2)) + (K4*g4*m2*(F7*m2 + 1))/(3600*(km4 + m2)) + (K5*g5*m2)/(3600*(km5 + m2)*(F8*m5 + 1)*(F9*m25 + 1)*(F10*m30 + 1)) - (K5*g5*m4)/(3600*(km6 + m4)*(F11*m5 + 1)*(F12*m25 + 1)*(F13*m30 + 1)) - (K3*g3*m3*m31*(F4*m32 + 1))/(3600*(km3 + m3*m31)*(F5*m2 + 1)*(F6*m15 + 1)) + (K31*g31*km40*km42*m2*(2*K31*g31*km42*m2 - 3600*((4*K31^2*g31^2*km42^2*m2^2*m26^2 + K33^2*g33^2*km40^2*m25^2*m26^2 + 4*K31^2*g31^2*km42^2*km43*km44*m2^2 + K33^2*g33^2*km40^2*km43*km44*m25^2 + 4*K31^2*g31^2*km42^2*km43*m2^2*m26 + 4*K31^2*g31^2*km42^2*km44*m2^2*m26 + K33^2*g33^2*km40^2*km43*m25^2*m26 + K33^2*g33^2*km40^2*km44*m25^2*m26 + 8*K31*K34*g31*g34*km42^2*m2^2*m26^2 - 8*K31*K35*g31*g35*km42^2*m2^2*m26^2 + 4*K33*K34*g33*g34*km40^2*m25^2*m26^2 - 4*K33*K35*g33*g35*km40^2*m25^2*m26^2 + F52^2*K33^2*g33^2*km40^2*m25^2*m26^2*m42^2 + 2*F52*K33^2*g33^2*km40^2*m25^2*m26^2*m42 + 8*K31*K34*g31*g34*km42^2*km44*m2^2*m26 - 8*K31*K35*g31*g35*km42^2*km43*m2^2*m26 + 4*K33*K34*g33*g34*km40^2*km44*m25^2*m26 - 4*K33*K35*g33*g35*km40^2*km43*m25^2*m26 + F52^2*K33^2*g33^2*km40^2*km43*km44*m25^2*m42^2 + F52^2*K33^2*g33^2*km40^2*km43*m25^2*m26*m42^2 + F52^2*K33^2*g33^2*km40^2*km44*m25^2*m26*m42^2 + 2*F52*K33^2*g33^2*km40^2*km43*km44*m25^2*m42 + 2*F52*K33^2*g33^2*km40^2*km43*m25^2*m26*m42 + 2*F52*K33^2*g33^2*km40^2*km44*m25^2*m26*m42 + 4*K31*K33*g31*g33*km40*km42*m2*m25*m26^2 + 4*F52^2*K33*K34*g33*g34*km40^2*m25^2*m26^2*m42^2 - 4*F52^2*K33*K35*g33*g35*km40^2*m25^2*m26^2*m42^2 + 8*F52*K31*K34*g31*g34*km42^2*m2^2*m26^2*m42 - 8*F52*K31*K35*g31*g35*km42^2*m2^2*m26^2*m42 + 8*F52*K33*K34*g33*g34*km40^2*m25^2*m26^2*m42 - 8*F52*K33*K35*g33*g35*km40^2*m25^2*m26^2*m42 + 4*K31*K33*g31*g33*km40*km42*km43*km44*m2*m25 + 4*K31*K33*g31*g33*km40*km42*km43*m2*m25*m26 + 4*K31*K33*g31*g33*km40*km42*km44*m2*m25*m26 + 4*F52^2*K33*K34*g33*g34*km40^2*km44*m25^2*m26*m42^2 - 4*F52^2*K33*K35*g33*g35*km40^2*km43*m25^2*m26*m42^2 + 8*F52*K31*K34*g31*g34*km42^2*km44*m2^2*m26*m42 - 8*F52*K31*K35*g31*g35*km42^2*km43*m2^2*m26*m42 + 8*F52*K33*K34*g33*g34*km40^2*km44*m25^2*m26*m42 - 8*F52*K33*K35*g33*g35*km40^2*km43*m25^2*m26*m42 + 4*F52*K31*K33*g31*g33*km40*km42*m2*m25*m26^2*m42 + 4*F52*K31*K33*g31*g33*km40*km42*km43*km44*m2*m25*m42 + 4*F52*K31*K33*g31*g33*km40*km42*km43*m2*m25*m26*m42 + 4*F52*K31*K33*g31*g33*km40*km42*km44*m2*m25*m26*m42)/(12960000*(km43 + m26)*(km44 + m26)))^(1/2) + K33*g33*km40*m25 + F52*K33*g33*km40*m25*m42))/(7200*(F52*m42 + 1)*(km40 + (km40*km42*m2*(2*K31*g31*km42*m2 - 3600*((4*K31^2*g31^2*km42^2*m2^2*m26^2 + K33^2*g33^2*km40^2*m25^2*m26^2 + 4*K31^2*g31^2*km42^2*km43*km44*m2^2 + K33^2*g33^2*km40^2*km43*km44*m25^2 + 4*K31^2*g31^2*km42^2*km43*m2^2*m26 + 4*K31^2*g31^2*km42^2*km44*m2^2*m26 + K33^2*g33^2*km40^2*km43*m25^2*m26 + K33^2*g33^2*km40^2*km44*m25^2*m26 + 8*K31*K34*g31*g34*km42^2*m2^2*m26^2 - 8*K31*K35*g31*g35*km42^2*m2^2*m26^2 + 4*K33*K34*g33*g34*km40^2*m25^2*m26^2 - 4*K33*K35*g33*g35*km40^2*m25^2*m26^2 + F52^2*K33^2*g33^2*km40^2*m25^2*m26^2*m42^2 + 2*F52*K33^2*g33^2*km40^2*m25^2*m26^2*m42 + 8*K31*K34*g31*g34*km42^2*km44*m2^2*m26 - 8*K31*K35*g31*g35*km42^2*km43*m2^2*m26 + 4*K33*K34*g33*g34*km40^2*km44*m25^2*m26 - 4*K33*K35*g33*g35*km40^2*km43*m25^2*m26 + F52^2*K33^2*g33^2*km40^2*km43*km44*m25^2*m42^2 + F52^2*K33^2*g33^2*km40^2*km43*m25^2*m26*m42^2 + F52^2*K33^2*g33^2*km40^2*km44*m25^2*m26*m42^2 + 2*F52*K33^2*g33^2*km40^2*km43*km44*m25^2*m42 + 2*F52*K33^2*g33^2*km40^2*km43*m25^2*m26*m42 + 2*F52*K33^2*g33^2*km40^2*km44*m25^2*m26*m42 + 4*K31*K33*g31*g33*km40*km42*m2*m25*m26^2 + 4*F52^2*K33*K34*g33*g34*km40^2*m25^2*m26^2*m42^2 - 4*F52^2*K33*K35*g33*g35*km40^2*m25^2*m26^2*m42^2 + 8*F52*K31*K34*g31*g34*km42^2*m2^2*m26^2*m42 - 8*F52*K31*K35*g31*g35*km42^2*m2^2*m26^2*m42 + 8*F52*K33*K34*g33*g34*km40^2*m25^2*m26^2*m42 - 8*F52*K33*K35*g33*g35*km40^2*m25^2*m26^2*m42 + 4*K31*K33*g31*g33*km40*km42*km43*km44*m2*m25 + 4*K31*K33*g31*g33*km40*km42*km43*m2*m25*m26 + 4*K31*K33*g31*g33*km40*km42*km44*m2*m25*m26 + 4*F52^2*K33*K34*g33*g34*km40^2*km44*m25^2*m26*m42^2 - 4*F52^2*K33*K35*g33*g35*km40^2*km43*m25^2*m26*m42^2 + 8*F52*K31*K34*g31*g34*km42^2*km44*m2^2*m26*m42 - 8*F52*K31*K35*g31*g35*km42^2*km43*m2^2*m26*m42 + 8*F52*K33*K34*g33*g34*km40^2*km44*m25^2*m26*m42 - 8*F52*K33*K35*g33*g35*km40^2*km43*m25^2*m26*m42 + 4*F52*K31*K33*g31*g33*km40*km42*m2*m25*m26^2*m42 + 4*F52*K31*K33*g31*g33*km40*km42*km43*km44*m2*m25*m42 + 4*F52*K31*K33*g31*g33*km40*km42*km43*m2*m25*m26*m42 + 4*F52*K31*K33*g31*g33*km40*km42*km44*m2*m25*m26*m42)/(12960000*(km43 + m26)*(km44 + m26)))^(1/2) + K33*g33*km40*m25 + F52*K33*g33*km40*m25*m42))/(2*(2*K31*g31*km42^2*m2^2 + K33*g33*km40^2*m25^2 + F52*K33*g33*km40^2*m25^2*m42)))*(2*K31*g31*km42^2*m2^2 + K33*g33*km40^2*m25^2 + F52*K33*g33*km40^2*m25^2*m42))))/(F3*K3*g3*m3*m31*(F4*m32 + 1)) + 1)*((K30*g30*m23*m34)/(3600*(km39 + m23*m34)*(F51*m16 + 1)) - (K22*g22*m16*m31)/(3600*(km29 + m16*m31)*((3600*F35*(km3 + m3*m31)*(F5*m2 + 1)*(F6*m15 + 1)*((K2*g2*m1*((F1*((K2*g2*m1)/(3600*(km1 + m1)) - (K2*g2*m2)/(3600*(km2 + m2))))/((F1*K2*g2*m1)/(3600*(km1 + m1)) - (F2*K2*g2*m2)/(3600*(km2 + m2))) - 1))/(3600*(km1 + m1)) - (K2*g2*m2*((F2*((K2*g2*m1)/(3600*(km1 + m1)) - (K2*g2*m2)/(3600*(km2 + m2))))/((F1*K2*g2*m1)/(3600*(km1 + m1)) - (F2*K2*g2*m2)/(3600*(km2 + m2))) - 1))/(3600*(km2 + m2)) + (K4*g4*m2*(F7*m2 + 1))/(3600*(km4 + m2)) + (K5*g5*m2)/(3600*(km5 + m2)*(F8*m5 + 1)*(F9*m25 + 1)*(F10*m30 + 1)) - (K5*g5*m4)/(3600*(km6 + m4)*(F11*m5 + 1)*(F12*m25 + 1)*(F13*m30 + 1)) - (K3*g3*m3*m31*(F4*m32 + 1))/(3600*(km3 + m3*m31)*(F5*m2 + 1)*(F6*m15 + 1)) + (K31*g31*km40*km42*m2*(2*K31*g31*km42*m2 - 3600*((4*K31^2*g31^2*km42^2*m2^2*m26^2 + K33^2*g33^2*km40^2*m25^2*m26^2 + 4*K31^2*g31^2*km42^2*km43*km44*m2^2 + K33^2*g33^2*km40^2*km43*km44*m25^2 + 4*K31^2*g31^2*km42^2*km43*m2^2*m26 + 4*K31^2*g31^2*km42^2*km44*m2^2*m26 + K33^2*g33^2*km40^2*km43*m25^2*m26 + K33^2*g33^2*km40^2*km44*m25^2*m26 + 8*K31*K34*g31*g34*km42^2*m2^2*m26^2 - 8*K31*K35*g31*g35*km42^2*m2^2*m26^2 + 4*K33*K34*g33*g34*km40^2*m25^2*m26^2 - 4*K33*K35*g33*g35*km40^2*m25^2*m26^2 + F52^2*K33^2*g33^2*km40^2*m25^2*m26^2*m42^2 + 2*F52*K33^2*g33^2*km40^2*m25^2*m26^2*m42 + 8*K31*K34*g31*g34*km42^2*km44*m2^2*m26 - 8*K31*K35*g31*g35*km42^2*km43*m2^2*m26 + 4*K33*K34*g33*g34*km40^2*km44*m25^2*m26 - 4*K33*K35*g33*g35*km40^2*km43*m25^2*m26 + F52^2*K33^2*g33^2*km40^2*km43*km44*m25^2*m42^2 + F52^2*K33^2*g33^2*km40^2*km43*m25^2*m26*m42^2 + F52^2*K33^2*g33^2*km40^2*km44*m25^2*m26*m42^2 + 2*F52*K33^2*g33^2*km40^2*km43*km44*m25^2*m42 + 2*F52*K33^2*g33^2*km40^2*km43*m25^2*m26*m42 + 2*F52*K33^2*g33^2*km40^2*km44*m25^2*m26*m42 + 4*K31*K33*g31*g33*km40*km42*m2*m25*m26^2 + 4*F52^2*K33*K34*g33*g34*km40^2*m25^2*m26^2*m42^2 - 4*F52^2*K33*K35*g33*g35*km40^2*m25^2*m26^2*m42^2 + 8*F52*K31*K34*g31*g34*km42^2*m2^2*m26^2*m42 - 8*F52*K31*K35*g31*g35*km42^2*m2^2*m26^2*m42 + 8*F52*K33*K34*g33*g34*km40^2*m25^2*m26^2*m42 - 8*F52*K33*K35*g33*g35*km40^2*m25^2*m26^2*m42 + 4*K31*K33*g31*g33*km40*km42*km43*km44*m2*m25 + 4*K31*K33*g31*g33*km40*km42*km43*m2*m25*m26 + 4*K31*K33*g31*g33*km40*km42*km44*m2*m25*m26 + 4*F52^2*K33*K34*g33*g34*km40^2*km44*m25^2*m26*m42^2 - 4*F52^2*K33*K35*g33*g35*km40^2*km43*m25^2*m26*m42^2 + 8*F52*K31*K34*g31*g34*km42^2*km44*m2^2*m26*m42 - 8*F52*K31*K35*g31*g35*km42^2*km43*m2^2*m26*m42 + 8*F52*K33*K34*g33*g34*km40^2*km44*m25^2*m26*m42 - 8*F52*K33*K35*g33*g35*km40^2*km43*m25^2*m26*m42 + 4*F52*K31*K33*g31*g33*km40*km42*m2*m25*m26^2*m42 + 4*F52*K31*K33*g31*g33*km40*km42*km43*km44*m2*m25*m42 + 4*F52*K31*K33*g31*g33*km40*km42*km43*m2*m25*m26*m42 + 4*F52*K31*K33*g31*g33*km40*km42*km44*m2*m25*m26*m42)/(12960000*(km43 + m26)*(km44 + m26)))^(1/2) + K33*g33*km40*m25 + F52*K33*g33*km40*m25*m42))/(7200*(F52*m42 + 1)*(km40 + (km40*km42*m2*(2*K31*g31*km42*m2 - 3600*((4*K31^2*g31^2*km42^2*m2^2*m26^2 + K33^2*g33^2*km40^2*m25^2*m26^2 + 4*K31^2*g31^2*km42^2*km43*km44*m2^2 + K33^2*g33^2*km40^2*km43*km44*m25^2 + 4*K31^2*g31^2*km42^2*km43*m2^2*m26 + 4*K31^2*g31^2*km42^2*km44*m2^2*m26 + K33^2*g33^2*km40^2*km43*m25^2*m26 + K33^2*g33^2*km40^2*km44*m25^2*m26 + 8*K31*K34*g31*g34*km42^2*m2^2*m26^2 - 8*K31*K35*g31*g35*km42^2*m2^2*m26^2 + 4*K33*K34*g33*g34*km40^2*m25^2*m26^2 - 4*K33*K35*g33*g35*km40^2*m25^2*m26^2 + F52^2*K33^2*g33^2*km40^2*m25^2*m26^2*m42^2 + 2*F52*K33^2*g33^2*km40^2*m25^2*m26^2*m42 + 8*K31*K34*g31*g34*km42^2*km44*m2^2*m26 - 8*K31*K35*g31*g35*km42^2*km43*m2^2*m26 + 4*K33*K34*g33*g34*km40^2*km44*m25^2*m26 - 4*K33*K35*g33*g35*km40^2*km43*m25^2*m26 + F52^2*K33^2*g33^2*km40^2*km43*km44*m25^2*m42^2 + F52^2*K33^2*g33^2*km40^2*km43*m25^2*m26*m42^2 + F52^2*K33^2*g33^2*km40^2*km44*m25^2*m26*m42^2 + 2*F52*K33^2*g33^2*km40^2*km43*km44*m25^2*m42 + 2*F52*K33^2*g33^2*km40^2*km43*m25^2*m26*m42 + 2*F52*K33^2*g33^2*km40^2*km44*m25^2*m26*m42 + 4*K31*K33*g31*g33*km40*km42*m2*m25*m26^2 + 4*F52^2*K33*K34*g33*g34*km40^2*m25^2*m26^2*m42^2 - 4*F52^2*K33*K35*g33*g35*km40^2*m25^2*m26^2*m42^2 + 8*F52*K31*K34*g31*g34*km42^2*m2^2*m26^2*m42 - 8*F52*K31*K35*g31*g35*km42^2*m2^2*m26^2*m42 + 8*F52*K33*K34*g33*g34*km40^2*m25^2*m26^2*m42 - 8*F52*K33*K35*g33*g35*km40^2*m25^2*m26^2*m42 + 4*K31*K33*g31*g33*km40*km42*km43*km44*m2*m25 + 4*K31*K33*g31*g33*km40*km42*km43*m2*m25*m26 + 4*K31*K33*g31*g33*km40*km42*km44*m2*m25*m26 + 4*F52^2*K33*K34*g33*g34*km40^2*km44*m25^2*m26*m42^2 - 4*F52^2*K33*K35*g33*g35*km40^2*km43*m25^2*m26*m42^2 + 8*F52*K31*K34*g31*g34*km42^2*km44*m2^2*m26*m42 - 8*F52*K31*K35*g31*g35*km42^2*km43*m2^2*m26*m42 + 8*F52*K33*K34*g33*g34*km40^2*km44*m25^2*m26*m42 - 8*F52*K33*K35*g33*g35*km40^2*km43*m25^2*m26*m42 + 4*F52*K31*K33*g31*g33*km40*km42*m2*m25*m26^2*m42 + 4*F52*K31*K33*g31*g33*km40*km42*km43*km44*m2*m25*m42 + 4*F52*K31*K33*g31*g33*km40*km42*km43*m2*m25*m26*m42 + 4*F52*K31*K33*g31*g33*km40*km42*km44*m2*m25*m26*m42)/(12960000*(km43 + m26)*(km44 + m26)))^(1/2) + K33*g33*km40*m25 + F52*K33*g33*km40*m25*m42))/(2*(2*K31*g31*km42^2*m2^2 + K33*g33*km40^2*m25^2 + F52*K33*g33*km40^2*m25^2*m42)))*(2*K31*g31*km42^2*m2^2 + K33*g33*km40^2*m25^2 + F52*K33*g33*km40^2*m25^2*m42))))/(F3*K3*g3*m3*m31*(F4*m32 + 1)) + 1)) + (K21*g21*m13*m31*(F33*m15 + 1))/(3600*(km28 + m13*m31)) - (K23*g23*m15*m16*(F36*m15 + 1)*(F37*m16 + 1))/(3600*(km30 + m15*m16)*(F39*m15 + 1)*(F38*m31 + 1))))/(F34*K22*g22*m16*m31) + 1)*((K15*g15*m11)/(3600*(km21 + m11)) - (K15*g15*m12)/(3600*(km22 + m12)) + (K22*g22*m16*m31*((3600*(km29 + m16*m31)*((3600*F35*(km3 + m3*m31)*(F5*m2 + 1)*(F6*m15 + 1)*((K2*g2*m1*((F1*((K2*g2*m1)/(3600*(km1 + m1)) - (K2*g2*m2)/(3600*(km2 + m2))))/((F1*K2*g2*m1)/(3600*(km1 + m1)) - (F2*K2*g2*m2)/(3600*(km2 + m2))) - 1))/(3600*(km1 + m1)) - (K2*g2*m2*((F2*((K2*g2*m1)/(3600*(km1 + m1)) - (K2*g2*m2)/(3600*(km2 + m2))))/((F1*K2*g2*m1)/(3600*(km1 + m1)) - (F2*K2*g2*m2)/(3600*(km2 + m2))) - 1))/(3600*(km2 + m2)) + (K4*g4*m2*(F7*m2 + 1))/(3600*(km4 + m2)) + (K5*g5*m2)/(3600*(km5 + m2)*(F8*m5 + 1)*(F9*m25 + 1)*(F10*m30 + 1)) - (K5*g5*m4)/(3600*(km6 + m4)*(F11*m5 + 1)*(F12*m25 + 1)*(F13*m30 + 1)) - (K3*g3*m3*m31*(F4*m32 + 1))/(3600*(km3 + m3*m31)*(F5*m2 + 1)*(F6*m15 + 1)) + (K31*g31*km40*km42*m2*(2*K31*g31*km42*m2 - 3600*((4*K31^2*g31^2*km42^2*m2^2*m26^2 + K33^2*g33^2*km40^2*m25^2*m26^2 + 4*K31^2*g31^2*km42^2*km43*km44*m2^2 + K33^2*g33^2*km40^2*km43*km44*m25^2 + 4*K31^2*g31^2*km42^2*km43*m2^2*m26 + 4*K31^2*g31^2*km42^2*km44*m2^2*m26 + K33^2*g33^2*km40^2*km43*m25^2*m26 + K33^2*g33^2*km40^2*km44*m25^2*m26 + 8*K31*K34*g31*g34*km42^2*m2^2*m26^2 - 8*K31*K35*g31*g35*km42^2*m2^2*m26^2 + 4*K33*K34*g33*g34*km40^2*m25^2*m26^2 - 4*K33*K35*g33*g35*km40^2*m25^2*m26^2 + F52^2*K33^2*g33^2*km40^2*m25^2*m26^2*m42^2 + 2*F52*K33^2*g33^2*km40^2*m25^2*m26^2*m42 + 8*K31*K34*g31*g34*km42^2*km44*m2^2*m26 - 8*K31*K35*g31*g35*km42^2*km43*m2^2*m26 + 4*K33*K34*g33*g34*km40^2*km44*m25^2*m26 - 4*K33*K35*g33*g35*km40^2*km43*m25^2*m26 + F52^2*K33^2*g33^2*km40^2*km43*km44*m25^2*m42^2 + F52^2*K33^2*g33^2*km40^2*km43*m25^2*m26*m42^2 + F52^2*K33^2*g33^2*km40^2*km44*m25^2*m26*m42^2 + 2*F52*K33^2*g33^2*km40^2*km43*km44*m25^2*m42 + 2*F52*K33^2*g33^2*km40^2*km43*m25^2*m26*m42 + 2*F52*K33^2*g33^2*km40^2*km44*m25^2*m26*m42 + 4*K31*K33*g31*g33*km40*km42*m2*m25*m26^2 + 4*F52^2*K33*K34*g33*g34*km40^2*m25^2*m26^2*m42^2 - 4*F52^2*K33*K35*g33*g35*km40^2*m25^2*m26^2*m42^2 + 8*F52*K31*K34*g31*g34*km42^2*m2^2*m26^2*m42 - 8*F52*K31*K35*g31*g35*km42^2*m2^2*m26^2*m42 + 8*F52*K33*K34*g33*g34*km40^2*m25^2*m26^2*m42 - 8*F52*K33*K35*g33*g35*km40^2*m25^2*m26^2*m42 + 4*K31*K33*g31*g33*km40*km42*km43*km44*m2*m25 + 4*K31*K33*g31*g33*km40*km42*km43*m2*m25*m26 + 4*K31*K33*g31*g33*km40*km42*km44*m2*m25*m26 + 4*F52^2*K33*K34*g33*g34*km40^2*km44*m25^2*m26*m42^2 - 4*F52^2*K33*K35*g33*g35*km40^2*km43*m25^2*m26*m42^2 + 8*F52*K31*K34*g31*g34*km42^2*km44*m2^2*m26*m42 - 8*F52*K31*K35*g31*g35*km42^2*km43*m2^2*m26*m42 + 8*F52*K33*K34*g33*g34*km40^2*km44*m25^2*m26*m42 - 8*F52*K33*K35*g33*g35*km40^2*km43*m25^2*m26*m42 + 4*F52*K31*K33*g31*g33*km40*km42*m2*m25*m26^2*m42 + 4*F52*K31*K33*g31*g33*km40*km42*km43*km44*m2*m25*m42 + 4*F52*K31*K33*g31*g33*km40*km42*km43*m2*m25*m26*m42 + 4*F52*K31*K33*g31*g33*km40*km42*km44*m2*m25*m26*m42)/(12960000*(km43 + m26)*(km44 + m26)))^(1/2) + K33*g33*km40*m25 + F52*K33*g33*km40*m25*m42))/(7200*(F52*m42 + 1)*(km40 + (km40*km42*m2*(2*K31*g31*km42*m2 - 3600*((4*K31^2*g31^2*km42^2*m2^2*m26^2 + K33^2*g33^2*km40^2*m25^2*m26^2 + 4*K31^2*g31^2*km42^2*km43*km44*m2^2 + K33^2*g33^2*km40^2*km43*km44*m25^2 + 4*K31^2*g31^2*km42^2*km43*m2^2*m26 + 4*K31^2*g31^2*km42^2*km44*m2^2*m26 + K33^2*g33^2*km40^2*km43*m25^2*m26 + K33^2*g33^2*km40^2*km44*m25^2*m26 + 8*K31*K34*g31*g34*km42^2*m2^2*m26^2 - 8*K31*K35*g31*g35*km42^2*m2^2*m26^2 + 4*K33*K34*g33*g34*km40^2*m25^2*m26^2 - 4*K33*K35*g33*g35*km40^2*m25^2*m26^2 + F52^2*K33^2*g33^2*km40^2*m25^2*m26^2*m42^2 + 2*F52*K33^2*g33^2*km40^2*m25^2*m26^2*m42 + 8*K31*K34*g31*g34*km42^2*km44*m2^2*m26 - 8*K31*K35*g31*g35*km42^2*km43*m2^2*m26 + 4*K33*K34*g33*g34*km40^2*km44*m25^2*m26 - 4*K33*K35*g33*g35*km40^2*km43*m25^2*m26 + F52^2*K33^2*g33^2*km40^2*km43*km44*m25^2*m42^2 + F52^2*K33^2*g33^2*km40^2*km43*m25^2*m26*m42^2 + F52^2*K33^2*g33^2*km40^2*km44*m25^2*m26*m42^2 + 2*F52*K33^2*g33^2*km40^2*km43*km44*m25^2*m42 + 2*F52*K33^2*g33^2*km40^2*km43*m25^2*m26*m42 + 2*F52*K33^2*g33^2*km40^2*km44*m25^2*m26*m42 + 4*K31*K33*g31*g33*km40*km42*m2*m25*m26^2 + 4*F52^2*K33*K34*g33*g34*km40^2*m25^2*m26^2*m42^2 - 4*F52^2*K33*K35*g33*g35*km40^2*m25^2*m26^2*m42^2 + 8*F52*K31*K34*g31*g34*km42^2*m2^2*m26^2*m42 - 8*F52*K31*K35*g31*g35*km42^2*m2^2*m26^2*m42 + 8*F52*K33*K34*g33*g34*km40^2*m25^2*m26^2*m42 - 8*F52*K33*K35*g33*g35*km40^2*m25^2*m26^2*m42 + 4*K31*K33*g31*g33*km40*km42*km43*km44*m2*m25 + 4*K31*K33*g31*g33*km40*km42*km43*m2*m25*m26 + 4*K31*K33*g31*g33*km40*km42*km44*m2*m25*m26 + 4*F52^2*K33*K34*g33*g34*km40^2*km44*m25^2*m26*m42^2 - 4*F52^2*K33*K35*g33*g35*km40^2*km43*m25^2*m26*m42^2 + 8*F52*K31*K34*g31*g34*km42^2*km44*m2^2*m26*m42 - 8*F52*K31*K35*g31*g35*km42^2*km43*m2^2*m26*m42 + 8*F52*K33*K34*g33*g34*km40^2*km44*m25^2*m26*m42 - 8*F52*K33*K35*g33*g35*km40^2*km43*m25^2*m26*m42 + 4*F52*K31*K33*g31*g33*km40*km42*m2*m25*m26^2*m42 + 4*F52*K31*K33*g31*g33*km40*km42*km43*km44*m2*m25*m42 + 4*F52*K31*K33*g31*g33*km40*km42*km43*m2*m25*m26*m42 + 4*F52*K31*K33*g31*g33*km40*km42*km44*m2*m25*m26*m42)/(12960000*(km43 + m26)*(km44 + m26)))^(1/2) + K33*g33*km40*m25 + F52*K33*g33*km40*m25*m42))/(2*(2*K31*g31*km42^2*m2^2 + K33*g33*km40^2*m25^2 + F52*K33*g33*km40^2*m25^2*m42)))*(2*K31*g31*km42^2*m2^2 + K33*g33*km40^2*m25^2 + F52*K33*g33*km40^2*m25^2*m42))))/(F3*K3*g3*m3*m31*(F4*m32 + 1)) + 1)*((K30*g30*m23*m34)/(3600*(km39 + m23*m34)*(F51*m16 + 1)) - (K22*g22*m16*m31)/(3600*(km29 + m16*m31)*((3600*F35*(km3 + m3*m31)*(F5*m2 + 1)*(F6*m15 + 1)*((K2*g2*m1*((F1*((K2*g2*m1)/(3600*(km1 + m1)) - (K2*g2*m2)/(3600*(km2 + m2))))/((F1*K2*g2*m1)/(3600*(km1 + m1)) - (F2*K2*g2*m2)/(3600*(km2 + m2))) - 1))/(3600*(km1 + m1)) - (K2*g2*m2*((F2*((K2*g2*m1)/(3600*(km1 + m1)) - (K2*g2*m2)/(3600*(km2 + m2))))/((F1*K2*g2*m1)/(3600*(km1 + m1)) - (F2*K2*g2*m2)/(3600*(km2 + m2))) - 1))/(3600*(km2 + m2)) + (K4*g4*m2*(F7*m2 + 1))/(3600*(km4 + m2)) + (K5*g5*m2)/(3600*(km5 + m2)*(F8*m5 + 1)*(F9*m25 + 1)*(F10*m30 + 1)) - (K5*g5*m4)/(3600*(km6 + m4)*(F11*m5 + 1)*(F12*m25 + 1)*(F13*m30 + 1)) - (K3*g3*m3*m31*(F4*m32 + 1))/(3600*(km3 + m3*m31)*(F5*m2 + 1)*(F6*m15 + 1)) + (K31*g31*km40*km42*m2*(2*K31*g31*km42*m2 - 3600*((4*K31^2*g31^2*km42^2*m2^2*m26^2 + K33^2*g33^2*km40^2*m25^2*m26^2 + 4*K31^2*g31^2*km42^2*km43*km44*m2^2 + K33^2*g33^2*km40^2*km43*km44*m25^2 + 4*K31^2*g31^2*km42^2*km43*m2^2*m26 + 4*K31^2*g31^2*km42^2*km44*m2^2*m26 + K33^2*g33^2*km40^2*km43*m25^2*m26 + K33^2*g33^2*km40^2*km44*m25^2*m26 + 8*K31*K34*g31*g34*km42^2*m2^2*m26^2 - 8*K31*K35*g31*g35*km42^2*m2^2*m26^2 + 4*K33*K34*g33*g34*km40^2*m25^2*m26^2 - 4*K33*K35*g33*g35*km40^2*m25^2*m26^2 + F52^2*K33^2*g33^2*km40^2*m25^2*m26^2*m42^2 + 2*F52*K33^2*g33^2*km40^2*m25^2*m26^2*m42 + 8*K31*K34*g31*g34*km42^2*km44*m2^2*m26 - 8*K31*K35*g31*g35*km42^2*km43*m2^2*m26 + 4*K33*K34*g33*g34*km40^2*km44*m25^2*m26 - 4*K33*K35*g33*g35*km40^2*km43*m25^2*m26 + F52^2*K33^2*g33^2*km40^2*km43*km44*m25^2*m42^2 + F52^2*K33^2*g33^2*km40^2*km43*m25^2*m26*m42^2 + F52^2*K33^2*g33^2*km40^2*km44*m25^2*m26*m42^2 + 2*F52*K33^2*g33^2*km40^2*km43*km44*m25^2*m42 + 2*F52*K33^2*g33^2*km40^2*km43*m25^2*m26*m42 + 2*F52*K33^2*g33^2*km40^2*km44*m25^2*m26*m42 + 4*K31*K33*g31*g33*km40*km42*m2*m25*m26^2 + 4*F52^2*K33*K34*g33*g34*km40^2*m25^2*m26^2*m42^2 - 4*F52^2*K33*K35*g33*g35*km40^2*m25^2*m26^2*m42^2 + 8*F52*K31*K34*g31*g34*km42^2*m2^2*m26^2*m42 - 8*F52*K31*K35*g31*g35*km42^2*m2^2*m26^2*m42 + 8*F52*K33*K34*g33*g34*km40^2*m25^2*m26^2*m42 - 8*F52*K33*K35*g33*g35*km40^2*m25^2*m26^2*m42 + 4*K31*K33*g31*g33*km40*km42*km43*km44*m2*m25 + 4*K31*K33*g31*g33*km40*km42*km43*m2*m25*m26 + 4*K31*K33*g31*g33*km40*km42*km44*m2*m25*m26 + 4*F52^2*K33*K34*g33*g34*km40^2*km44*m25^2*m26*m42^2 - 4*F52^2*K33*K35*g33*g35*km40^2*km43*m25^2*m26*m42^2 + 8*F52*K31*K34*g31*g34*km42^2*km44*m2^2*m26*m42 - 8*F52*K31*K35*g31*g35*km42^2*km43*m2^2*m26*m42 + 8*F52*K33*K34*g33*g34*km40^2*km44*m25^2*m26*m42 - 8*F52*K33*K35*g33*g35*km40^2*km43*m25^2*m26*m42 + 4*F52*K31*K33*g31*g33*km40*km42*m2*m25*m26^2*m42 + 4*F52*K31*K33*g31*g33*km40*km42*km43*km44*m2*m25*m42 + 4*F52*K31*K33*g31*g33*km40*km42*km43*m2*m25*m26*m42 + 4*F52*K31*K33*g31*g33*km40*km42*km44*m2*m25*m26*m42)/(12960000*(km43 + m26)*(km44 + m26)))^(1/2) + K33*g33*km40*m25 + F52*K33*g33*km40*m25*m42))/(7200*(F52*m42 + 1)*(km40 + (km40*km42*m2*(2*K31*g31*km42*m2 - 3600*((4*K31^2*g31^2*km42^2*m2^2*m26^2 + K33^2*g33^2*km40^2*m25^2*m26^2 + 4*K31^2*g31^2*km42^2*km43*km44*m2^2 + K33^2*g33^2*km40^2*km43*km44*m25^2 + 4*K31^2*g31^2*km42^2*km43*m2^2*m26 + 4*K31^2*g31^2*km42^2*km44*m2^2*m26 + K33^2*g33^2*km40^2*km43*m25^2*m26 + K33^2*g33^2*km40^2*km44*m25^2*m26 + 8*K31*K34*g31*g34*km42^2*m2^2*m26^2 - 8*K31*K35*g31*g35*km42^2*m2^2*m26^2 + 4*K33*K34*g33*g34*km40^2*m25^2*m26^2 - 4*K33*K35*g33*g35*km40^2*m25^2*m26^2 + F52^2*K33^2*g33^2*km40^2*m25^2*m26^2*m42^2 + 2*F52*K33^2*g33^2*km40^2*m25^2*m26^2*m42 + 8*K31*K34*g31*g34*km42^2*km44*m2^2*m26 - 8*K31*K35*g31*g35*km42^2*km43*m2^2*m26 + 4*K33*K34*g33*g34*km40^2*km44*m25^2*m26 - 4*K33*K35*g33*g35*km40^2*km43*m25^2*m26 + F52^2*K33^2*g33^2*km40^2*km43*km44*m25^2*m42^2 + F52^2*K33^2*g33^2*km40^2*km43*m25^2*m26*m42^2 + F52^2*K33^2*g33^2*km40^2*km44*m25^2*m26*m42^2 + 2*F52*K33^2*g33^2*km40^2*km43*km44*m25^2*m42 + 2*F52*K33^2*g33^2*km40^2*km43*m25^2*m26*m42 + 2*F52*K33^2*g33^2*km40^2*km44*m25^2*m26*m42 + 4*K31*K33*g31*g33*km40*km42*m2*m25*m26^2 + 4*F52^2*K33*K34*g33*g34*km40^2*m25^2*m26^2*m42^2 - 4*F52^2*K33*K35*g33*g35*km40^2*m25^2*m26^2*m42^2 + 8*F52*K31*K34*g31*g34*km42^2*m2^2*m26^2*m42 - 8*F52*K31*K35*g31*g35*km42^2*m2^2*m26^2*m42 + 8*F52*K33*K34*g33*g34*km40^2*m25^2*m26^2*m42 - 8*F52*K33*K35*g33*g35*km40^2*m25^2*m26^2*m42 + 4*K31*K33*g31*g33*km40*km42*km43*km44*m2*m25 + 4*K31*K33*g31*g33*km40*km42*km43*m2*m25*m26 + 4*K31*K33*g31*g33*km40*km42*km44*m2*m25*m26 + 4*F52^2*K33*K34*g33*g34*km40^2*km44*m25^2*m26*m42^2 - 4*F52^2*K33*K35*g33*g35*km40^2*km43*m25^2*m26*m42^2 + 8*F52*K31*K34*g31*g34*km42^2*km44*m2^2*m26*m42 - 8*F52*K31*K35*g31*g35*km42^2*km43*m2^2*m26*m42 + 8*F52*K33*K34*g33*g34*km40^2*km44*m25^2*m26*m42 - 8*F52*K33*K35*g33*g35*km40^2*km43*m25^2*m26*m42 + 4*F52*K31*K33*g31*g33*km40*km42*m2*m25*m26^2*m42 + 4*F52*K31*K33*g31*g33*km40*km42*km43*km44*m2*m25*m42 + 4*F52*K31*K33*g31*g33*km40*km42*km43*m2*m25*m26*m42 + 4*F52*K31*K33*g31*g33*km40*km42*km44*m2*m25*m26*m42)/(12960000*(km43 + m26)*(km44 + m26)))^(1/2) + K33*g33*km40*m25 + F52*K33*g33*km40*m25*m42))/(2*(2*K31*g31*km42^2*m2^2 + K33*g33*km40^2*m25^2 + F52*K33*g33*km40^2*m25^2*m42)))*(2*K31*g31*km42^2*m2^2 + K33*g33*km40^2*m25^2 + F52*K33*g33*km40^2*m25^2*m42))))/(F3*K3*g3*m3*m31*(F4*m32 + 1)) + 1)) + (K21*g21*m13*m31*(F33*m15 + 1))/(3600*(km28 + m13*m31)) - (K23*g23*m15*m16*(F36*m15 + 1)*(F37*m16 + 1))/(3600*(km30 + m15*m16)*(F39*m15 + 1)*(F38*m31 + 1))))/(K22*g22*m16*m31) + 1))/(3600*(km29 + m16*m31)*((3600*F35*(km3 + m3*m31)*(F5*m2 + 1)*(F6*m15 + 1)*((K2*g2*m1*((F1*((K2*g2*m1)/(3600*(km1 + m1)) - (K2*g2*m2)/(3600*(km2 + m2))))/((F1*K2*g2*m1)/(3600*(km1 + m1)) - (F2*K2*g2*m2)/(3600*(km2 + m2))) - 1))/(3600*(km1 + m1)) - (K2*g2*m2*((F2*((K2*g2*m1)/(3600*(km1 + m1)) - (K2*g2*m2)/(3600*(km2 + m2))))/((F1*K2*g2*m1)/(3600*(km1 + m1)) - (F2*K2*g2*m2)/(3600*(km2 + m2))) - 1))/(3600*(km2 + m2)) + (K4*g4*m2*(F7*m2 + 1))/(3600*(km4 + m2)) + (K5*g5*m2)/(3600*(km5 + m2)*(F8*m5 + 1)*(F9*m25 + 1)*(F10*m30 + 1)) - (K5*g5*m4)/(3600*(km6 + m4)*(F11*m5 + 1)*(F12*m25 + 1)*(F13*m30 + 1)) - (K3*g3*m3*m31*(F4*m32 + 1))/(3600*(km3 + m3*m31)*(F5*m2 + 1)*(F6*m15 + 1)) + (K31*g31*km40*km42*m2*(2*K31*g31*km42*m2 - 3600*((4*K31^2*g31^2*km42^2*m2^2*m26^2 + K33^2*g33^2*km40^2*m25^2*m26^2 + 4*K31^2*g31^2*km42^2*km43*km44*m2^2 + K33^2*g33^2*km40^2*km43*km44*m25^2 + 4*K31^2*g31^2*km42^2*km43*m2^2*m26 + 4*K31^2*g31^2*km42^2*km44*m2^2*m26 + K33^2*g33^2*km40^2*km43*m25^2*m26 + K33^2*g33^2*km40^2*km44*m25^2*m26 + 8*K31*K34*g31*g34*km42^2*m2^2*m26^2 - 8*K31*K35*g31*g35*km42^2*m2^2*m26^2 + 4*K33*K34*g33*g34*km40^2*m25^2*m26^2 - 4*K33*K35*g33*g35*km40^2*m25^2*m26^2 + F52^2*K33^2*g33^2*km40^2*m25^2*m26^2*m42^2 + 2*F52*K33^2*g33^2*km40^2*m25^2*m26^2*m42 + 8*K31*K34*g31*g34*km42^2*km44*m2^2*m26 - 8*K31*K35*g31*g35*km42^2*km43*m2^2*m26 + 4*K33*K34*g33*g34*km40^2*km44*m25^2*m26 - 4*K33*K35*g33*g35*km40^2*km43*m25^2*m26 + F52^2*K33^2*g33^2*km40^2*km43*km44*m25^2*m42^2 + F52^2*K33^2*g33^2*km40^2*km43*m25^2*m26*m42^2 + F52^2*K33^2*g33^2*km40^2*km44*m25^2*m26*m42^2 + 2*F52*K33^2*g33^2*km40^2*km43*km44*m25^2*m42 + 2*F52*K33^2*g33^2*km40^2*km43*m25^2*m26*m42 + 2*F52*K33^2*g33^2*km40^2*km44*m25^2*m26*m42 + 4*K31*K33*g31*g33*km40*km42*m2*m25*m26^2 + 4*F52^2*K33*K34*g33*g34*km40^2*m25^2*m26^2*m42^2 - 4*F52^2*K33*K35*g33*g35*km40^2*m25^2*m26^2*m42^2 + 8*F52*K31*K34*g31*g34*km42^2*m2^2*m26^2*m42 - 8*F52*K31*K35*g31*g35*km42^2*m2^2*m26^2*m42 + 8*F52*K33*K34*g33*g34*km40^2*m25^2*m26^2*m42 - 8*F52*K33*K35*g33*g35*km40^2*m25^2*m26^2*m42 + 4*K31*K33*g31*g33*km40*km42*km43*km44*m2*m25 + 4*K31*K33*g31*g33*km40*km42*km43*m2*m25*m26 + 4*K31*K33*g31*g33*km40*km42*km44*m2*m25*m26 + 4*F52^2*K33*K34*g33*g34*km40^2*km44*m25^2*m26*m42^2 - 4*F52^2*K33*K35*g33*g35*km40^2*km43*m25^2*m26*m42^2 + 8*F52*K31*K34*g31*g34*km42^2*km44*m2^2*m26*m42 - 8*F52*K31*K35*g31*g35*km42^2*km43*m2^2*m26*m42 + 8*F52*K33*K34*g33*g34*km40^2*km44*m25^2*m26*m42 - 8*F52*K33*K35*g33*g35*km40^2*km43*m25^2*m26*m42 + 4*F52*K31*K33*g31*g33*km40*km42*m2*m25*m26^2*m42 + 4*F52*K31*K33*g31*g33*km40*km42*km43*km44*m2*m25*m42 + 4*F52*K31*K33*g31*g33*km40*km42*km43*m2*m25*m26*m42 + 4*F52*K31*K33*g31*g33*km40*km42*km44*m2*m25*m26*m42)/(12960000*(km43 + m26)*(km44 + m26)))^(1/2) + K33*g33*km40*m25 + F52*K33*g33*km40*m25*m42))/(7200*(F52*m42 + 1)*(km40 + (km40*km42*m2*(2*K31*g31*km42*m2 - 3600*((4*K31^2*g31^2*km42^2*m2^2*m26^2 + K33^2*g33^2*km40^2*m25^2*m26^2 + 4*K31^2*g31^2*km42^2*km43*km44*m2^2 + K33^2*g33^2*km40^2*km43*km44*m25^2 + 4*K31^2*g31^2*km42^2*km43*m2^2*m26 + 4*K31^2*g31^2*km42^2*km44*m2^2*m26 + K33^2*g33^2*km40^2*km43*m25^2*m26 + K33^2*g33^2*km40^2*km44*m25^2*m26 + 8*K31*K34*g31*g34*km42^2*m2^2*m26^2 - 8*K31*K35*g31*g35*km42^2*m2^2*m26^2 + 4*K33*K34*g33*g34*km40^2*m25^2*m26^2 - 4*K33*K35*g33*g35*km40^2*m25^2*m26^2 + F52^2*K33^2*g33^2*km40^2*m25^2*m26^2*m42^2 + 2*F52*K33^2*g33^2*km40^2*m25^2*m26^2*m42 + 8*K31*K34*g31*g34*km42^2*km44*m2^2*m26 - 8*K31*K35*g31*g35*km42^2*km43*m2^2*m26 + 4*K33*K34*g33*g34*km40^2*km44*m25^2*m26 - 4*K33*K35*g33*g35*km40^2*km43*m25^2*m26 + F52^2*K33^2*g33^2*km40^2*km43*km44*m25^2*m42^2 + F52^2*K33^2*g33^2*km40^2*km43*m25^2*m26*m42^2 + F52^2*K33^2*g33^2*km40^2*km44*m25^2*m26*m42^2 + 2*F52*K33^2*g33^2*km40^2*km43*km44*m25^2*m42 + 2*F52*K33^2*g33^2*km40^2*km43*m25^2*m26*m42 + 2*F52*K33^2*g33^2*km40^2*km44*m25^2*m26*m42 + 4*K31*K33*g31*g33*km40*km42*m2*m25*m26^2 + 4*F52^2*K33*K34*g33*g34*km40^2*m25^2*m26^2*m42^2 - 4*F52^2*K33*K35*g33*g35*km40^2*m25^2*m26^2*m42^2 + 8*F52*K31*K34*g31*g34*km42^2*m2^2*m26^2*m42 - 8*F52*K31*K35*g31*g35*km42^2*m2^2*m26^2*m42 + 8*F52*K33*K34*g33*g34*km40^2*m25^2*m26^2*m42 - 8*F52*K33*K35*g33*g35*km40^2*m25^2*m26^2*m42 + 4*K31*K33*g31*g33*km40*km42*km43*km44*m2*m25 + 4*K31*K33*g31*g33*km40*km42*km43*m2*m25*m26 + 4*K31*K33*g31*g33*km40*km42*km44*m2*m25*m26 + 4*F52^2*K33*K34*g33*g34*km40^2*km44*m25^2*m26*m42^2 - 4*F52^2*K33*K35*g33*g35*km40^2*km43*m25^2*m26*m42^2 + 8*F52*K31*K34*g31*g34*km42^2*km44*m2^2*m26*m42 - 8*F52*K31*K35*g31*g35*km42^2*km43*m2^2*m26*m42 + 8*F52*K33*K34*g33*g34*km40^2*km44*m25^2*m26*m42 - 8*F52*K33*K35*g33*g35*km40^2*km43*m25^2*m26*m42 + 4*F52*K31*K33*g31*g33*km40*km42*m2*m25*m26^2*m42 + 4*F52*K31*K33*g31*g33*km40*km42*km43*km44*m2*m25*m42 + 4*F52*K31*K33*g31*g33*km40*km42*km43*m2*m25*m26*m42 + 4*F52*K31*K33*g31*g33*km40*km42*km44*m2*m25*m26*m42)/(12960000*(km43 + m26)*(km44 + m26)))^(1/2) + K33*g33*km40*m25 + F52*K33*g33*km40*m25*m42))/(2*(2*K31*g31*km42^2*m2^2 + K33*g33*km40^2*m25^2 + F52*K33*g33*km40^2*m25^2*m42)))*(2*K31*g31*km42^2*m2^2 + K33*g33*km40^2*m25^2 + F52*K33*g33*km40^2*m25^2*m42))))/(F3*K3*g3*m3*m31*(F4*m32 + 1)) + 1)))) - 1)/F30;
u(26) = -((K2*g2*m1)/(3600*(km1 + m1)) - (K2*g2*m2)/(3600*(km2 + m2)))/((F1*K2*g2*m1)/(3600*(km1 + m1)) - (F2*K2*g2*m2)/(3600*(km2 + m2)));
u(27) = -((K13*g13*m10)/(3600*(km18 + m10)) + (K14*g14*m10)/(1800*(km19 + m10)) - (K14*g14*m11)/(1800*(km20 + m11)) - (K15*g15*m11)/(3600*(km21 + m11)) + (K15*g15*m12)/(3600*(km22 + m12)) - (K13*g13*m9*m32)/(3600*(km17 + m9*m32)))/((F24*K14*g14*m10)/(1800*(km19 + m10)) - (F25*K14*g14*m11)/(1800*(km20 + m11)));

u = real(u);
u = mat2gray(u);
%%u(:)=abs(u(:));
%%%%% METABOLIC ODE PART

xdot(1) = (K2*g2*m2*(F2*u(26) + 1))/(km2 + m2) - (K2*g2*m1*(F1*u(26) + 1))/(km1 + m1);
xdot(2) = (K2*g2*m1*(F1*u(26) + 1))/(km1 + m1) - (K4*g4*m2*(F7*m2 + 1))/(km4 + m2) - (K2*g2*m2*(F2*u(26) + 1))/(km2 + m2) - (K31*g31*m2*u(18))/((km40 + m2*u(18))*(F52*m42 + 1)) - (K5*g5*m2)/((km5 + m2)*(F8*m5 + 1)*(F9*m25 + 1)*(F10*m30 + 1)) + (K5*g5*m4)/((km6 + m4)*(F11*m5 + 1)*(F12*m25 + 1)*(F13*m30 + 1)) + (K3*g3*m3*m31*(F4*m32 + 1)*(F3*u(19) + 1))/((km3 + m3*m31)*(F5*m2 + 1)*(F6*m15 + 1));
xdot(3) = (K1*g1*u(17))/(km0 + u(17)) + (K4*g4*m2*(F7*m2 + 1))/(km4 + m2) - (K3*g3*m3*m31*(F4*m32 + 1)*(F3*u(19) + 1))/((km3 + m3*m31)*(F5*m2 + 1)*(F6*m15 + 1));
xdot(4) = (K37*g37*m8*m29)/(km46 + m8*m29) - (K36*g36*m4*m8)/(km48 + m4*m8) + (K36*g36*m28*m30)/(km47 + m28*m30) + (K9*g9*m6*(F22*u(8) + 1))/((km10 + m6)*(F23*m4 + 1)) + (K7*g7*m5)/((km8 + m5)*(F19*m6 + 1)*(F18*u(24) + 1)) + (K5*g5*m2)/((km5 + m2)*(F8*m5 + 1)*(F9*m25 + 1)*(F10*m30 + 1)) - (K5*g5*m4)/((km6 + m4)*(F11*m5 + 1)*(F12*m25 + 1)*(F13*m30 + 1)) - (K8*g8*m4*m31*(F20*m4 + 1))/((km9 + m4*m31)*(F21*u(8) + 1)) - (K6*g6*m4*m31*(F15*m6 + 1)*(F14_modified*u(24) + 1))/((km7 + m4*m31)*(F17*m17 + 1)*(F16*m31 + 1));
xdot(5) = (K10*g10*m7*m8)/(km12 + m7*m8) - (K10*g10*m5)/(km11 + m5) - (K7*g7*m5)/((km8 + m5)*(F19*m6 + 1)*(F18*u(24) + 1)) + (K6*g6*m4*m31*(F15*m6 + 1)*(F14_modified*u(24) + 1))/((km7 + m4*m31)*(F17*m17 + 1)*(F16*m31 + 1));
xdot(6) = (K8*g8*m4*m31*(F20*m4 + 1))/((km9 + m4*m31)*(F21*u(8) + 1)) - (K9*g9*m6*(F22*u(8) + 1))/((km10 + m6)*(F23*m4 + 1));
xdot(7) = (K10*g10*m5)/(km11 + m5) - (K11*g11*m7)/(km13 + m7) + (K11*g11*m8)/(km14 + m8) - (K10*g10*m7*m8)/(km12 + m7*m8);
xdot(8) = (K10*g10*m5)/(km11 + m5) + (K11*g11*m7)/(km13 + m7) - (K11*g11*m8)/(km14 + m8) - (K10*g10*m7*m8)/(km12 + m7*m8) - (K12*g12*m8*m34)/(km15 + m8*m34) + (K12*g12*m9*m33)/(km16 + m9*m33) - (K36*g36*m4*m8)/(km48 + m4*m8) - (K37*g37*m8*m29)/(km46 + m8*m29) + (K36*g36*m27*m28)/(km45 + m27*m28) - (K36*g36*m28*m30)/(km47 + m28*m30);
xdot(9) = (K13*g13*m10)/(km18 + m10) + (K12*g12*m8*m34)/(km15 + m8*m34) - (K12*g12*m9*m33)/(km16 + m9*m33) - (K13*g13*m9*m32)/(km17 + m9*m32);
xdot(10) = (K13*g13*m9*m32)/(km17 + m9*m32) - (K13*g13*m10)/(km18 + m10) - (K14*g14*m10*(F24*u(27) + 1))/(km19 + m10) + (K14*g14*m11*(F25*u(27) + 1))/(km20 + m11);
xdot(11) = (K15*g15*m12)/(km22 + m12) - (K15*g15*m11)/(km21 + m11) + (K14*g14*m10*(F24*u(27) + 1))/(km19 + m10) - (K14*g14*m11*(F25*u(27) + 1))/(km20 + m11);
xdot(12) = (K15*g15*m11)/(km21 + m11) - (K15*g15*m12)/(km22 + m12) + (K22*g22*m16*m31*(F34*u(20) + 1))/((km29 + m16*m31)*(F35*u(19) + 1)) - (K16*g16*m12*m32*(F27*m5 + 1)*(F26*u(19) + 1))/((km23 + m12*m32)*(F29*m15 + 1)*(F28*m31 + 1)*(F32*m37 + 1)*(F31*u(20) + 1)*(F30*u(25) + 1));
xdot(13) = (K16*g16*m12*m32*(F27*m5 + 1)*(F26*u(19) + 1))/((km23 + m12*m32)*(F29*m15 + 1)*(F28*m31 + 1)*(F32*m37 + 1)*(F31*u(20) + 1)*(F30*u(25) + 1)) - (K21*g21*m13*m31*(F33*m15 + 1))/(km28 + m13*m31) - (K18*g18*m13*m34*m35*(F53*u(21) + 1))/(km25 + m13*m34*m35) - (K17*g17*m13*m33)/(km24 + m13*m33);
xdot(14) = (K17*g17*m13*m33)/(km24 + m13*m33);
xdot(15) = (K19*g19*m31*m34*m37*m38)/(km26 + m31*m34*m37*m38) - (K20*g20*m15*m32*m33*m40)/(km27 + m15*m32*m33*m40) + (K18*g18*m13*m34*m35*(F53*u(21) + 1))/(km25 + m13*m34*m35) - (K23*g23*m15*m16*(F36*m15 + 1)*(F37*m16 + 1))/((km30 + m15*m16)*(F39*m15 + 1)*(F38*m31 + 1));
xdot(16) = (K30*g30*m23*m34)/((km39 + m23*m34)*(F51*m16 + 1)) + (K21*g21*m13*m31*(F33*m15 + 1))/(km28 + m13*m31) - (K22*g22*m16*m31*(F34*u(20) + 1))/((km29 + m16*m31)*(F35*u(19) + 1)) - (K23*g23*m15*m16*(F36*m15 + 1)*(F37*m16 + 1))/((km30 + m15*m16)*(F39*m15 + 1)*(F38*m31 + 1));
xdot(17) = (K24*g24*m18)/((km32 + m18)*(F41*u(22) + 1)) - (K24*g24*m17)/((km31 + m17)*(F40*u(22) + 1)) + (K23*g23*m15*m16*(F36*m15 + 1)*(F37*m16 + 1))/((km30 + m15*m16)*(F39*m15 + 1)*(F38*m31 + 1));
xdot(18) = (K24*g24*m17)/((km31 + m17)*(F40*u(22) + 1)) - (K24*g24*m18)/((km32 + m18)*(F41*u(22) + 1)) - (K25*g25*m18*m34*(F42*m18 + 1)*(F43*m32 + 1)*(F54*u(21) + 1))/((km33 + m18*m34)*(F44*m31 + 1)*(F45*m33 + 1));
xdot(19) = (K25*g25*m18*m34*(F42*m18 + 1)*(F43*m32 + 1)*(F54*u(21) + 1))/((km33 + m18*m34)*(F44*m31 + 1)*(F45*m33 + 1)) - (K26*g26*m19*m34*m41*(F55*u(21) + 1))/((km34 + m19*m34*m41)*(F47*m20 + 1)*(F46*m31 + 1));
xdot(20) = (K27*g27*m21)/(km36 + m21) - 3*((K27*g27*m20*m32)/(km35 + m20*m32)) + (K26*g26*m19*m34*m41*(F55*u(21) + 1))/((km34 + m19*m34*m41)*(F47*m20 + 1)*(F46*m31 + 1));
xdot(21) = 3*((K27*g27*m20*m32)/(km35 + m20*m32)) - (K27*g27*m21)/(km36 + m21) - (K28*g28*m21*m38)/((km37 + m21*m38)*(F48*u(23) + 1));
xdot(22) = (K28*g28*m21*m38)/((km37 + m21*m38)*(F48*u(23) + 1)) - (K29*g29*m22)/(km38 + m22);
xdot(23) = (K29*g29*m22)/(km38 + m22) - (K30*g30*m23*m34)/((km39 + m23*m34)*(F51*m16 + 1));
xdot(24) = (K31*g31*m2*u(18))/((km40 + m2*u(18))*(F52*m42 + 1)) - (K32*g32*m24)/(km41 + m24);
xdot(25) = (K32*g32*m24)/(km41 + m24) - (K33*g33*m25*u(18))/(km42 + m25*u(18));
xdot(26) = (K33*g33*m25*u(18))/(km42 + m25*u(18)) - (K35*g35*m26)/(km44 + m26) - (K34*g34*m26)/(km43 + m26);
xdot(27) = (K34*g34*m26)/(km43 + m26) - (K36*g36*m27*m28)/(km45 + m27*m28);
xdot(28) = (K35*g35*m26)/(km44 + m26) + (K36*g36*m4*m8)/(km48 + m4*m8) - (K36*g36*m27*m28)/(km45 + m27*m28) - (K36*g36*m28*m30)/(km47 + m28*m30);
xdot(29) = (K36*g36*m27*m28)/(km45 + m27*m28) - (K37*g37*m8*m29)/(km46 + m8*m29);
xdot(30) = (K36*g36*m4*m8)/(km48 + m4*m8) + (K37*g37*m8*m29)/(km46 + m8*m29) - (K36*g36*m28*m30)/(km47 + m28*m30);
xdot(31) = (K13*g13*m9*m32)/(km17 + m9*m32) + 3*((K27*g27*m20*m32)/(km35 + m20*m32)) - (K21*g21*m13*m31*(F33*m15 + 1))/(km28 + m13*m31) + (K20*g20*m15*m32*m33*m40)/(km27 + m15*m32*m33*m40) - (K19*g19*m31*m34*m37*m38)/(km26 + m31*m34*m37*m38) - (K8*g8*m4*m31*(F20*m4 + 1))/((km9 + m4*m31)*(F21*u(8) + 1)) - (K22*g22*m16*m31*(F34*u(20) + 1))/((km29 + m16*m31)*(F35*u(19) + 1)) - (K3*g3*m3*m31*(F4*m32 + 1)*(F3*u(19) + 1))/((km3 + m3*m31)*(F5*m2 + 1)*(F6*m15 + 1)) - (K6*g6*m4*m31*(F15*m6 + 1)*(F14_modified*u(24) + 1))/((km7 + m4*m31)*(F17*m17 + 1)*(F16*m31 + 1)) + (K16*g16*m12*m32*(F27*m5 + 1)*(F26*u(19) + 1))/((km23 + m12*m32)*(F29*m15 + 1)*(F28*m31 + 1)*(F32*m37 + 1)*(F31*u(20) + 1)*(F30*u(25) + 1));
xdot(32) = (K21*g21*m13*m31*(F33*m15 + 1))/(km28 + m13*m31) - 3*((K27*g27*m20*m32)/(km35 + m20*m32)) - (K13*g13*m9*m32)/(km17 + m9*m32) - (K20*g20*m15*m32*m33*m40)/(km27 + m15*m32*m33*m40) + (K19*g19*m31*m34*m37*m38)/(km26 + m31*m34*m37*m38) + (K8*g8*m4*m31*(F20*m4 + 1))/((km9 + m4*m31)*(F21*u(8) + 1)) + (K22*g22*m16*m31*(F34*u(20) + 1))/((km29 + m16*m31)*(F35*u(19) + 1)) + (K3*g3*m3*m31*(F4*m32 + 1)*(F3*u(19) + 1))/((km3 + m3*m31)*(F5*m2 + 1)*(F6*m15 + 1)) + (K6*g6*m4*m31*(F15*m6 + 1)*(F14_modified*u(24) + 1))/((km7 + m4*m31)*(F17*m17 + 1)*(F16*m31 + 1)) - (K16*g16*m12*m32*(F27*m5 + 1)*(F26*u(19) + 1))/((km23 + m12*m32)*(F29*m15 + 1)*(F28*m31 + 1)*(F32*m37 + 1)*(F31*u(20) + 1)*(F30*u(25) + 1));
xdot(33) = (K12*g12*m8*m34)/(km15 + m8*m34) - (K12*g12*m9*m33)/(km16 + m9*m33) - (K17*g17*m13*m33)/(km24 + m13*m33) + (K30*g30*m23*m34)/((km39 + m23*m34)*(F51*m16 + 1)) - (K20*g20*m15*m32*m33*m40)/(km27 + m15*m32*m33*m40) + (K19*g19*m31*m34*m37*m38)/(km26 + m31*m34*m37*m38) + (K18*g18*m13*m34*m35*(F53*u(21) + 1))/(km25 + m13*m34*m35) + (K26*g26*m19*m34*m41*(F55*u(21) + 1))/((km34 + m19*m34*m41)*(F47*m20 + 1)*(F46*m31 + 1)) + (K25*g25*m18*m34*(F42*m18 + 1)*(F43*m32 + 1)*(F54*u(21) + 1))/((km33 + m18*m34)*(F44*m31 + 1)*(F45*m33 + 1));
xdot(34) = (K12*g12*m9*m33)/(km16 + m9*m33) - (K12*g12*m8*m34)/(km15 + m8*m34) + (K17*g17*m13*m33)/(km24 + m13*m33) - (K30*g30*m23*m34)/((km39 + m23*m34)*(F51*m16 + 1)) + (K20*g20*m15*m32*m33*m40)/(km27 + m15*m32*m33*m40) - (K19*g19*m31*m34*m37*m38)/(km26 + m31*m34*m37*m38) - (K18*g18*m13*m34*m35*(F53*u(21) + 1))/(km25 + m13*m34*m35) - (K26*g26*m19*m34*m41*(F55*u(21) + 1))/((km34 + m19*m34*m41)*(F47*m20 + 1)*(F46*m31 + 1)) - (K25*g25*m18*m34*(F42*m18 + 1)*(F43*m32 + 1)*(F54*u(21) + 1))/((km33 + m18*m34)*(F44*m31 + 1)*(F45*m33 + 1));
xdot(35) = -(K18*g18*m13*m34*m35*(F53*u(21) + 1))/(km25 + m13*m34*m35);
xdot(36) = (K18*g18*m13*m34*m35*(F53*u(21) + 1))/(km25 + m13*m34*m35);
xdot(37) = (K20*g20*m15*m32*m33*m40)/(km27 + m15*m32*m33*m40) - (K19*g19*m31*m34*m37*m38)/(km26 + m31*m34*m37*m38);
xdot(38) = (K20*g20*m15*m32*m33*m40)/(km27 + m15*m32*m33*m40) - (K28*g28*m21*m38)/((km37 + m21*m38)*(F48*u(23) + 1)) - (K19*g19*m31*m34*m37*m38)/(km26 + m31*m34*m37*m38);
xdot(39) = (K28*g28*m21*m38)/((km37 + m21*m38)*(F48*u(23) + 1)) - (K20*g20*m15*m32*m33*m40)/(km27 + m15*m32*m33*m40) + (K19*g19*m31*m34*m37*m38)/(km26 + m31*m34*m37*m38);
xdot(40) = (K23*g23*m15*m16*(F36*m15 + 1)*(F37*m16 + 1))/((km30 + m15*m16)*(F39*m15 + 1)*(F38*m31 + 1)) - (K26*g26*m19*m34*m41*(F55*u(21) + 1))/((km34 + m19*m34*m41)*(F47*m20 + 1)*(F46*m31 + 1));
xdot(41) = (K33*g33*m25*u(18))/(km42 + m25*u(18)) + (K31*g31*m2*u(18))/((km40 + m2*u(18))*(F52*m42 + 1));

%%%%% SIGNALLING ODE PART

xdot(42) = KS1*u(1) + KS31*u(6) - (KS8*s1*s2*s3*s6*s9)/((F_S6*s16 + 1)*(F_S7*s17 + 1));
xdot(43) = KS48*u(4) + KS49*u(5) - KS2*s4*s11;
xdot(44) = KS37*u(6) + KS38*u(7) + KS52*u(6) - KS2*s4*s11 + (KS4*s10)/(F_S3*s15 + 1) - (KS18*s4)/(F_S19*s5 + 1) - (KS5*s4*s28)/(F_S4*s15 + 1) - (KS14*s4*s8)/(F_S16*s15 + 1) - (KS17*s2)/((F_S17*s4 + 1)*(F_S18*s5 + 1)) - (KS6*s4*s5*s13*s14)/(F_S5*u(9) + 1);
xdot(45) = (KS11*s10)/(F_S13*s5 + 1) - (KS3*s10*s12)/((F_S1*u(13) + 1)*(F_S2*u(14) + 1));
xdot(46) = KS39*u(6) + KS43*u(8) + KS10/(F_S10*u(15) + 1) - (KS4*s10)/(F_S3*s15 + 1) - (KS11*s10)/(F_S13*s5 + 1) - (KS3*s10*s12)/((F_S1*u(13) + 1)*(F_S2*u(14) + 1)) - (KS19*s5*s24*s25)/((F_S21*s10 + 1)*(F_S20*s26 + 1)*(F_S22*u(16) + 1));
xdot(47) = KS16*u(24) + (KS26*m31)/(F_S29*s29 + 1) - (KS27*s5)/(F_S30*s28 + 1) - (KS5*s4*s28)/(F_S4*s15 + 1);
xdot(48) = (KS5*s4*s28)/(F_S4*s15 + 1) - (KS11*s10)/(F_S13*s5 + 1) - (KS18*s4)/(F_S19*s5 + 1) - (KS27*s5)/(F_S30*s28 + 1) - KS2*s4*s11 - (KS17*s2)/((F_S17*s4 + 1)*(F_S18*s5 + 1)) - (KS6*s4*s5*s13*s14)/(F_S5*u(9) + 1) - (KS19*s5*s24*s25)/((F_S21*s10 + 1)*(F_S20*s26 + 1)*(F_S22*u(16) + 1));
xdot(49) = KS28*m31*s29 - KS24/(F_S28*s13 + 1) - KS23/((F_S25*m21 + 1)*(F_S26*m22 + 1)*(F_S27*s13 + 1)) - (KS6*s4*s5*s13*s14)/(F_S5*u(9) + 1);
xdot(50) = KS47*u(4) + KS46*u(11) + KS13/((F_S15*s3 + 1)*(F_S14*s18 + 1)) - (KS6*s4*s5*s13*s14)/(F_S5*u(9) + 1);
xdot(51) = KS36*u(6) - KS29*s3 - KS7*s3 - KS13/((F_S15*s3 + 1)*(F_S14*s18 + 1)) - (KS9*s7)/((F_S8*s3 + 1)*(F_S9*s6 + 1)) + (KS3*s10*s12)/((F_S1*u(13) + 1)*(F_S2*u(14) + 1)) - (KS8*s1*s2*s3*s6*s9)/((F_S6*s16 + 1)*(F_S7*s17 + 1));
xdot(52) = KS53*u(7) - (KS9*s7)/((F_S8*s3 + 1)*(F_S9*s6 + 1)) + (KS6*s4*s5*s13*s14)/(F_S5*u(9) + 1) - (KS8*s1*s2*s3*s6*s9)/((F_S6*s16 + 1)*(F_S7*s17 + 1));
xdot(53) = (KS9*s7)/((F_S8*s3 + 1)*(F_S9*s6 + 1)) - (KS8*s1*s2*s3*s6*s9)/((F_S6*s16 + 1)*(F_S7*s17 + 1));
xdot(54) = KS32*u(2) + KS33*u(3) + KS34*u(4) + KS35*u(5) + KS2*s4*s11 - (KS17*s2)/((F_S17*s4 + 1)*(F_S18*s5 + 1)) - (KS8*s1*s2*s3*s6*s9)/((F_S6*s16 + 1)*(F_S7*s17 + 1));
xdot(55) = KS7*s3 - (KS9*s7)/((F_S8*s3 + 1)*(F_S9*s6 + 1));
xdot(56) = (KS14*s4*s8)/(F_S16*s15 + 1) - KS15*s19;
xdot(57) = KS50*u(12) - KS25*s24 - KS22*s24 + (KS20*s8*s18)/((F_S23*s23 + 1)*(F_S11*u(7) + 1)) - (KS19*s5*s24*s25)/((F_S21*s10 + 1)*(F_S20*s26 + 1)*(F_S22*u(16) + 1));
xdot(58) = KS21/(F_S24*s23 + 1) - (KS19*s5*s24*s25)/((F_S21*s10 + 1)*(F_S20*s26 + 1)*(F_S22*u(16) + 1));
xdot(59) = KS12*m31 + KS45*u(9) + KS44*u(24) - KS13/((F_S15*s3 + 1)*(F_S14*s18 + 1)) - (KS20*s8*s18)/((F_S23*s23 + 1)*(F_S11*u(7) + 1));
xdot(60) = (KS8*s1*s2*s3*s6*s9)/((F_S6*s16 + 1)*(F_S7*s17 + 1)) - (KS14*s4*s8)/(F_S16*s15 + 1);
xdot(61) = KS15*s19;
xdot(62) = (KS18*s4)/(F_S19*s5 + 1);
xdot(63) = (KS19*s5*s24*s25)/((F_S21*s10 + 1)*(F_S20*s26 + 1)*(F_S22*u(16) + 1)) - (KS20*s8*s18)/((F_S23*s23 + 1)*(F_S11*u(7) + 1)) - KS21/(F_S24*s23 + 1);
xdot(64) = KS22*s24 - (KS4*s10)/(F_S3*s15 + 1) - (KS5*s4*s28)/(F_S4*s15 + 1) - (KS14*s4*s8)/(F_S16*s15 + 1);
xdot(65) = (KS27*s5)/(F_S30*s28 + 1) - (KS26*m31)/(F_S29*s29 + 1) - KS28*m31*s29;
xdot(66) = KS25*s24;
xdot(67) = KS23/((F_S25*m21 + 1)*(F_S26*m22 + 1)*(F_S27*s13 + 1));
xdot(68) = KS24/(F_S28*s13 + 1) + KS30/((F_S31*m21 + 1)*(F_S32*m22 + 1));
xdot(69) = (KS17*s2)/((F_S17*s4 + 1)*(F_S18*s5 + 1));
xdot(70) = KS29*s3 + KS54*u(3) - (KS19*s5*s24*s25)/((F_S21*s10 + 1)*(F_S20*s26 + 1)*(F_S22*u(16) + 1));

%%%% GENE REGULATORY DOT PART

xdot(71) = kbg1 - lg1 + (kg1*s4*s5*s6*s8*s26)/(FG1*s24 + 1);
xdot(72) = kbg2 - lg2 + kg2*s8;
xdot(73) = kbg3 - lg3 + kg3*s4*s5*s6*s8*s26;
xdot(74) = kbg4 - lg4 + (kg4*s8)/((FG2*s5 + 1)*(FG3*s18 + 1));
xdot(75) = kbg5 - lg5 + kg5*s5*s6*s8*s26;
xdot(76) = kbg6 - lg6 + (kg6*s4*s5*s6*s8*s18*s26)/(FG4*s24 + 1);
xdot(77) = kbg7 - lg7 + kg7*s3;
xdot(78) = kbg8 - lg8 + (kg8*s4*s5*s8*s18*s26)/(FG5*s24 + 1);
xdot(79) = kbg9 - lg9 + kg9*s3*s26*u(16);
xdot(80) = kbg10 - lg10 + kg10*s5*s6*s8*s26;
xdot(81) = kbg11 - lg11 + kg11*s6;
xdot(82) = kbg12 - lg12 + kg12*s5*s8*s26;
xdot(83) = kbg13 - lg13 + kg13*s5*s6*s8*s26;
xdot(84) = kbg14 - lg14 + kg14*s6*s8;
xdot(85) = kbg15 - lg15 + kg15*s8*s26;
xdot(86) = kbg16 - lg16 + (kg16*s8)/(FG6*u(6) + 1);
xdot(87) = kbg17 - lg17 + kg17*s6*s8*s26;
xdot(88) = kbg18 - lg18 + (kg18*s5*s26)/((FG7*s8 + 1)*(FG8*s24 + 1));
xdot(89) = kbg19 - lg19 + kg19*s1*s3*s26;
xdot(90) = kbg20 - lg20 + (kg20*s3*s5*s26*u(16))/(FG9*s18 + 1);
xdot(91) = kbg21 - lg21 + kg21*s1*s2*s24*s26*u(16);
xdot(92) = kbg22 - lg22 + (kg22*s2*s3)/((FG10*s5 + 1)*(FG11*s18 + 1));
xdot(93) = kbg23 - lg23 + kg23*u(10);
xdot(94) = kbg24 - lg24 + kg24*s8;
xdot(95) = kbg25 - lg25 + kg25*s26;
xdot(96) = kbg26 - lg26 + kg26*s1*s26;
xdot(97) = kbg27 - lg27 + kg27*s3*s26*u(16);
xdot(98) = kbg28 - lg28 + (kg28*s5*s24*s26*u(16))/(FG13*u(2) + 1);
xdot(99) = kbg29 - lg29 + kg29*s2*s24*s26*u(16);
xdot(100) = kbg30 - lg30 + kg30*s3*u(16);
xdot(101) = kbg31 - lg31 + (kg31*s6)/(FG12*s24 + 1);
xdot(102) = kbg32 - lg32 + kg32*s24;
xdot(103) = kbg33 - lg33 + kg33*s6*s24;
xdot(104) = kbg34 - lg34 + kg34*s3*s5;
xdot(105) = kbg35 - lg35 + kg35*s6;
xdot(106) = kbg36 - lg36 + kg36*s6;
xdot(107) = kbg37 - lg37 + kg37*s1*s26*u(16);
