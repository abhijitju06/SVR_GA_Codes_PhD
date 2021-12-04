import numpy as np
from math import sin
from scipy.integrate import odeint
from numpy import linspace
class plant:
    def __init__(self):
        return
    
    def integrated_network(self, y, t, km,K,KS,kbg,kg,lg,F,F_S,FG,u1,u2,u3,u4,u5,u6,u7,u8,u9,u10,u11,u12,u13,u14,u15,u16,u17,u18,u19,u20,u21,u22,u23,u24,u25,u26,u27):
        
        
        km0,km1,km2,km3,km4,km5,km6,km7,km8,km9,km10,km11,km12,km13,km14,km15,km16,km17,km18,km19,km20,km21,km22,km23,km24,km25,km26,km27,km28,km29,km30,km31,km32,km33,km34,km35,km36,km37,km38,km39,km40,km41,km42,km43,km44,km45,km46,km47,km48 = km[0:]        
        K1,K2,K3,K4,K5,K6,K7,K8,K9,K10,K11,K12,K13,K14,K15,K16,K17,K18,K19,K20,K21,K22,K23,K24,K25,K26,K27,K28,K29,K30,K31,K32,K33,K34,K35,K36,K37 = K[0:]
        KS1,KS2,KS3,KS4,KS5,KS6,KS7,KS8,KS9,KS10,KS11,KS12,KS13,KS14,KS15,KS16,KS17,KS18,KS19,KS20,KS21,KS22,KS23,KS24,KS25,KS26,KS27,KS28,KS29,KS30,KS31,KS32,KS33,KS34,KS35,KS36,KS37,KS38,KS39,KS43,KS44,KS45,KS46,KS47,KS48,KS49,KS50,KS52,KS53,KS54 = KS[0:]
        kbg1,kbg2,kbg3,kbg4,kbg5,kbg6,kbg7,kbg8,kbg9,kbg10,kbg11,kbg12,kbg13,kbg14,kbg15,kbg16,kbg17,kbg18,kbg19,kbg20,kbg21,kbg22,kbg23,kbg24,kbg25,kbg26,kbg27,kbg28,kbg29,kbg30,kbg31,kbg32,kbg33,kbg34,kbg35,kbg36,kbg37 = kbg[0:]       
        kg1,kg2,kg3,kg4,kg5,kg6,kg7,kg8,kg9,kg10,kg11,kg12,kg13,kg14,kg15,kg16,kg17,kg18,kg19,kg20,kg21,kg22,kg23,kg24,kg25,kg26,kg27,kg28,kg29,kg30,kg31,kg32,kg33,kg34,kg35,kg36,kg37 = kg[0:]
        lg1,lg2,lg3,lg4,lg5,lg6,lg7,lg8,lg9,lg10,lg11,lg12,lg13,lg14,lg15,lg16,lg17,lg18,lg19,lg20,lg21,lg22,lg23,lg24,lg25,lg26,lg27,lg28,lg29,lg30,lg31,lg32,lg33,lg34,lg35,lg36,lg37 = lg[0:]
        F1,F2,F3,F4,F5,F6,F7,F8,F9,F10,F11,F12,F13,F14_modified,F15,F16,F17,F18,F19,F20,F21,F22,F23,F24,F25,F26,F27,F28,F29,F30,F31,F32,F33,F34,F35,F36,F37,F38,F39,F40,F41,F42,F43,F44,F45,F46,F47,F48,F51,F52,F53,F54,F55 = F[0:]
        F_S1,F_S2,F_S3,F_S4,F_S5,F_S6,F_S7,F_S8,F_S9,F_S10,F_S11,F_S13,F_S14,F_S15,F_S16,F_S17,F_S18,F_S19,F_S20,F_S21,F_S22,F_S23,F_S24,F_S25,F_S26,F_S27,F_S28,F_S29,F_S30,F_S31,F_S32 = F_S[0:]
        FG1,FG2,FG3,FG4,FG5,FG6,FG7,FG8,FG9,FG10,FG11,FG12,FG13 = FG[0:]        
        
        m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15,m16,m17,m18,m19,m20,m21,m22,m23,m24,m25,m26,m27,m28,m29,m30,m31,m32,m33,m34,m35,m36,m37,m38,m40,m41,m42,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,s16,s17,s18,s19,s20,s21,s22,s23,s24,s25,s26,s27,s28,s29,g1,g2,g3,g4,g5,g6,g7,g8,g9,g10,g11,g12,g13,g14,g15,g16,g17,g18,g19,g20,g21,g22,g23,g24,g25,g26,g27,g28,g29,g30,g31,g32,g33,g34,g35,g36,g37 = y
        
        dydt = [(K2*g2*m2*(F2*u26 + 1))/(km2 + m2) - (K2*g2*m1*(F1*u26 + 1))/(km1 + m1),
                (K2*g2*m1*(F1*u26 + 1))/(km1 + m1) - (K4*g4*m2*(F7*m2 + 1))/(km4 + m2) - (K2*g2*m2*(F2*u26 + 1))/(km2 + m2) - (K31*g31*m2*u18)/((km40 + m2*u18)*(F52*m42 + 1)) - (K5*g5*m2)/((km5 + m2)*(F8*m5 + 1)*(F9*m25 + 1)*(F10*m30 + 1)) + (K5*g5*m4)/((km6 + m4)*(F11*m5 + 1)*(F12*m25 + 1)*(F13*m30 + 1)) + (K3*g3*m3*m31*(F4*m32 + 1)*(F3*u19 + 1))/((km3 + m3*m31)*(F5*m2 + 1)*(F6*m15 + 1)),
                (K1*g1*u17)/(km0 + u17) + (K4*g4*m2*(F7*m2 + 1))/(km4 + m2) - (K3*g3*m3*m31*(F4*m32 + 1)*(F3*u19 + 1))/((km3 + m3*m31)*(F5*m2 + 1)*(F6*m15 + 1)),
                (K37*g37*m8*m29)/(km46 + m8*m29) - (K36*g36*m4*m8)/(km48 + m4*m8) + (K36*g36*m28*m30)/(km47 + m28*m30) + (K9*g9*m6*(F22*u8 + 1))/((km10 + m6)*(F23*m4 + 1)) + (K7*g7*m5)/((km8 + m5)*(F19*m6 + 1)*(F18*u24 + 1)) + (K5*g5*m2)/((km5 + m2)*(F8*m5 + 1)*(F9*m25 + 1)*(F10*m30 + 1)) - (K5*g5*m4)/((km6 + m4)*(F11*m5 + 1)*(F12*m25 + 1)*(F13*m30 + 1)) - (K8*g8*m4*m31*(F20*m4 + 1))/((km9 + m4*m31)*(F21*u8 + 1)) - (K6*g6*m4*m31*(F15*m6 + 1)*(F14_modified*u24 + 1))/((km7 + m4*m31)*(F17*m17 + 1)*(F16*m31 + 1)),
                (K10*g10*m7*m8)/(km12 + m7*m8) - (K10*g10*m5)/(km11 + m5) - (K7*g7*m5)/((km8 + m5)*(F19*m6 + 1)*(F18*u24 + 1)) + (K6*g6*m4*m31*(F15*m6 + 1)*(F14_modified*u24 + 1))/((km7 + m4*m31)*(F17*m17 + 1)*(F16*m31 + 1)),
                (K8*g8*m4*m31*(F20*m4 + 1))/((km9 + m4*m31)*(F21*u8 + 1)) - (K9*g9*m6*(F22*u8 + 1))/((km10 + m6)*(F23*m4 + 1)),
                (K10*g10*m5)/(km11 + m5) - (K11*g11*m7)/(km13 + m7) + (K11*g11*m8)/(km14 + m8) - (K10*g10*m7*m8)/(km12 + m7*m8),
                (K10*g10*m5)/(km11 + m5) + (K11*g11*m7)/(km13 + m7) - (K11*g11*m8)/(km14 + m8) - (K10*g10*m7*m8)/(km12 + m7*m8) - (K12*g12*m8*m34)/(km15 + m8*m34) + (K12*g12*m9*m33)/(km16 + m9*m33) - (K36*g36*m4*m8)/(km48 + m4*m8) - (K37*g37*m8*m29)/(km46 + m8*m29) + (K36*g36*m27*m28)/(km45 + m27*m28) - (K36*g36*m28*m30)/(km47 + m28*m30),
                (K13*g13*m10)/(km18 + m10) + (K12*g12*m8*m34)/(km15 + m8*m34) - (K12*g12*m9*m33)/(km16 + m9*m33) - (K13*g13*m9*m32)/(km17 + m9*m32),
                (K13*g13*m9*m32)/(km17 + m9*m32) - (K13*g13*m10)/(km18 + m10) - (K14*g14*m10*(F24*u27 + 1))/(km19 + m10) + (K14*g14*m11*(F25*u27 + 1))/(km20 + m11),
                (K15*g15*m12)/(km22 + m12) - (K15*g15*m11)/(km21 + m11) + (K14*g14*m10*(F24*u27 + 1))/(km19 + m10) - (K14*g14*m11*(F25*u27 + 1))/(km20 + m11),
                (K15*g15*m11)/(km21 + m11) - (K15*g15*m12)/(km22 + m12) + (K22*g22*m16*m31*(F34*u20 + 1))/((km29 + m16*m31)*(F35*u19 + 1)) - (K16*g16*m12*m32*(F27*m5 + 1)*(F26*u19 + 1))/((km23 + m12*m32)*(F29*m15 + 1)*(F28*m31 + 1)*(F32*m37 + 1)*(F31*u20 + 1)*(F30*u25 + 1)),
                (K16*g16*m12*m32*(F27*m5 + 1)*(F26*u19 + 1))/((km23 + m12*m32)*(F29*m15 + 1)*(F28*m31 + 1)*(F32*m37 + 1)*(F31*u20 + 1)*(F30*u25 + 1)) - (K21*g21*m13*m31*(F33*m15 + 1))/(km28 + m13*m31) - (K18*g18*m13*m34*m35*(F53*u21 + 1))/(km25 + m13*m34*m35) - (K17*g17*m13*m33)/(km24 + m13*m33),
                (K17*g17*m13*m33)/(km24 + m13*m33),
                (K19*g19*m31*m34*m37*m38)/(km26 + m31*m34*m37*m38) - (K20*g20*m15*m32*m33*m40)/(km27 + m15*m32*m33*m40) + (K18*g18*m13*m34*m35*(F53*u21 + 1))/(km25 + m13*m34*m35) - (K23*g23*m15*m16*(F36*m15 + 1)*(F37*m16 + 1))/((km30 + m15*m16)*(F39*m15 + 1)*(F38*m31 + 1)),
                (K30*g30*m23*m34)/((km39 + m23*m34)*(F51*m16 + 1)) + (K21*g21*m13*m31*(F33*m15 + 1))/(km28 + m13*m31) - (K22*g22*m16*m31*(F34*u20 + 1))/((km29 + m16*m31)*(F35*u19 + 1)) - (K23*g23*m15*m16*(F36*m15 + 1)*(F37*m16 + 1))/((km30 + m15*m16)*(F39*m15 + 1)*(F38*m31 + 1)),
                (K24*g24*m18)/((km32 + m18)*(F41*u22 + 1)) - (K24*g24*m17)/((km31 + m17)*(F40*u22 + 1)) + (K23*g23*m15*m16*(F36*m15 + 1)*(F37*m16 + 1))/((km30 + m15*m16)*(F39*m15 + 1)*(F38*m31 + 1)),
                (K24*g24*m17)/((km31 + m17)*(F40*u22 + 1)) - (K24*g24*m18)/((km32 + m18)*(F41*u22 + 1)) - (K25*g25*m18*m34*(F42*m18 + 1)*(F43*m32 + 1)*(F54*u21 + 1))/((km33 + m18*m34)*(F44*m31 + 1)*(F45*m33 + 1)),
                (K25*g25*m18*m34*(F42*m18 + 1)*(F43*m32 + 1)*(F54*u21 + 1))/((km33 + m18*m34)*(F44*m31 + 1)*(F45*m33 + 1)) - (K26*g26*m19*m34*m41*(F55*u21 + 1))/((km34 + m19*m34*m41)*(F47*m20 + 1)*(F46*m31 + 1)),
                (K27*g27*m21)/(km36 + m21) - 3*((K27*g27*m20*m32)/(km35 + m20*m32)) + (K26*g26*m19*m34*m41*(F55*u21 + 1))/((km34 + m19*m34*m41)*(F47*m20 + 1)*(F46*m31 + 1)),
                3*((K27*g27*m20*m32)/(km35 + m20*m32)) - (K27*g27*m21)/(km36 + m21) - (K28*g28*m21*m38)/((km37 + m21*m38)*(F48*u23 + 1)),
                (K28*g28*m21*m38)/((km37 + m21*m38)*(F48*u23 + 1)) - (K29*g29*m22)/(km38 + m22),
                (K29*g29*m22)/(km38 + m22) - (K30*g30*m23*m34)/((km39 + m23*m34)*(F51*m16 + 1)),
                (K31*g31*m2*u18)/((km40 + m2*u18)*(F52*m42 + 1)) - (K32*g32*m24)/(km41 + m24),
                (K32*g32*m24)/(km41 + m24) - (K33*g33*m25*u18)/(km42 + m25*u18),
                (K33*g33*m25*u18)/(km42 + m25*u18) - (K35*g35*m26)/(km44 + m26) - (K34*g34*m26)/(km43 + m26),
                (K34*g34*m26)/(km43 + m26) - (K36*g36*m27*m28)/(km45 + m27*m28),
                (K35*g35*m26)/(km44 + m26) + (K36*g36*m4*m8)/(km48 + m4*m8) - (K36*g36*m27*m28)/(km45 + m27*m28) - (K36*g36*m28*m30)/(km47 + m28*m30),
                (K36*g36*m27*m28)/(km45 + m27*m28) - (K37*g37*m8*m29)/(km46 + m8*m29),
                (K36*g36*m4*m8)/(km48 + m4*m8) + (K37*g37*m8*m29)/(km46 + m8*m29) - (K36*g36*m28*m30)/(km47 + m28*m30),
                (K13*g13*m9*m32)/(km17 + m9*m32) + 3*((K27*g27*m20*m32)/(km35 + m20*m32)) - (K21*g21*m13*m31*(F33*m15 + 1))/(km28 + m13*m31) + (K20*g20*m15*m32*m33*m40)/(km27 + m15*m32*m33*m40) - (K19*g19*m31*m34*m37*m38)/(km26 + m31*m34*m37*m38) - (K8*g8*m4*m31*(F20*m4 + 1))/((km9 + m4*m31)*(F21*u8 + 1)) - (K22*g22*m16*m31*(F34*u20 + 1))/((km29 + m16*m31)*(F35*u19 + 1)) - (K3*g3*m3*m31*(F4*m32 + 1)*(F3*u19 + 1))/((km3 + m3*m31)*(F5*m2 + 1)*(F6*m15 + 1)) - (K6*g6*m4*m31*(F15*m6 + 1)*(F14_modified*u24 + 1))/((km7 + m4*m31)*(F17*m17 + 1)*(F16*m31 + 1)) + (K16*g16*m12*m32*(F27*m5 + 1)*(F26*u19 + 1))/((km23 + m12*m32)*(F29*m15 + 1)*(F28*m31 + 1)*(F32*m37 + 1)*(F31*u20 + 1)*(F30*u25 + 1)),
                (K21*g21*m13*m31*(F33*m15 + 1))/(km28 + m13*m31) - 3*((K27*g27*m20*m32)/(km35 + m20*m32)) - (K13*g13*m9*m32)/(km17 + m9*m32) - (K20*g20*m15*m32*m33*m40)/(km27 + m15*m32*m33*m40) + (K19*g19*m31*m34*m37*m38)/(km26 + m31*m34*m37*m38) + (K8*g8*m4*m31*(F20*m4 + 1))/((km9 + m4*m31)*(F21*u8 + 1)) + (K22*g22*m16*m31*(F34*u20 + 1))/((km29 + m16*m31)*(F35*u19 + 1)) + (K3*g3*m3*m31*(F4*m32 + 1)*(F3*u19 + 1))/((km3 + m3*m31)*(F5*m2 + 1)*(F6*m15 + 1)) + (K6*g6*m4*m31*(F15*m6 + 1)*(F14_modified*u24 + 1))/((km7 + m4*m31)*(F17*m17 + 1)*(F16*m31 + 1)) - (K16*g16*m12*m32*(F27*m5 + 1)*(F26*u19 + 1))/((km23 + m12*m32)*(F29*m15 + 1)*(F28*m31 + 1)*(F32*m37 + 1)*(F31*u20 + 1)*(F30*u25 + 1)),
                (K12*g12*m8*m34)/(km15 + m8*m34) - (K12*g12*m9*m33)/(km16 + m9*m33) - (K17*g17*m13*m33)/(km24 + m13*m33) + (K30*g30*m23*m34)/((km39 + m23*m34)*(F51*m16 + 1)) - (K20*g20*m15*m32*m33*m40)/(km27 + m15*m32*m33*m40) + (K19*g19*m31*m34*m37*m38)/(km26 + m31*m34*m37*m38) + (K18*g18*m13*m34*m35*(F53*u21 + 1))/(km25 + m13*m34*m35) + (K26*g26*m19*m34*m41*(F55*u21 + 1))/((km34 + m19*m34*m41)*(F47*m20 + 1)*(F46*m31 + 1)) + (K25*g25*m18*m34*(F42*m18 + 1)*(F43*m32 + 1)*(F54*u21 + 1))/((km33 + m18*m34)*(F44*m31 + 1)*(F45*m33 + 1)),
                (K12*g12*m9*m33)/(km16 + m9*m33) - (K12*g12*m8*m34)/(km15 + m8*m34) + (K17*g17*m13*m33)/(km24 + m13*m33) - (K30*g30*m23*m34)/((km39 + m23*m34)*(F51*m16 + 1)) + (K20*g20*m15*m32*m33*m40)/(km27 + m15*m32*m33*m40) - (K19*g19*m31*m34*m37*m38)/(km26 + m31*m34*m37*m38) - (K18*g18*m13*m34*m35*(F53*u21 + 1))/(km25 + m13*m34*m35) - (K26*g26*m19*m34*m41*(F55*u21 + 1))/((km34 + m19*m34*m41)*(F47*m20 + 1)*(F46*m31 + 1)) - (K25*g25*m18*m34*(F42*m18 + 1)*(F43*m32 + 1)*(F54*u21 + 1))/((km33 + m18*m34)*(F44*m31 + 1)*(F45*m33 + 1)),
                -(K18*g18*m13*m34*m35*(F53*u21 + 1))/(km25 + m13*m34*m35),
                (K18*g18*m13*m34*m35*(F53*u21 + 1))/(km25 + m13*m34*m35),
                (K20*g20*m15*m32*m33*m40)/(km27 + m15*m32*m33*m40) - (K19*g19*m31*m34*m37*m38)/(km26 + m31*m34*m37*m38),
                (K20*g20*m15*m32*m33*m40)/(km27 + m15*m32*m33*m40) - (K28*g28*m21*m38)/((km37 + m21*m38)*(F48*u23 + 1)) - (K19*g19*m31*m34*m37*m38)/(km26 + m31*m34*m37*m38),
                (K28*g28*m21*m38)/((km37 + m21*m38)*(F48*u23 + 1)) - (K20*g20*m15*m32*m33*m40)/(km27 + m15*m32*m33*m40) + (K19*g19*m31*m34*m37*m38)/(km26 + m31*m34*m37*m38),
                (K23*g23*m15*m16*(F36*m15 + 1)*(F37*m16 + 1))/((km30 + m15*m16)*(F39*m15 + 1)*(F38*m31 + 1)) - (K26*g26*m19*m34*m41*(F55*u21 + 1))/((km34 + m19*m34*m41)*(F47*m20 + 1)*(F46*m31 + 1)),
                (K33*g33*m25*u18)/(km42 + m25*u18) + (K31*g31*m2*u18)/((km40 + m2*u18)*(F52*m42 + 1)),
                KS1*u1 + KS31*u6 - (KS8*s1*s2*s3*s6*s9)/((F_S6*s16 + 1)*(F_S7*s17 + 1)),
                KS48*u4 + KS49*u5 - KS2*s4*s11,
                KS37*u6 + KS38*u7 + KS52*u6 - KS2*s4*s11 + (KS4*s10)/(F_S3*s15 + 1) - (KS18*s4)/(F_S19*s5 + 1) - (KS5*s4*s28)/(F_S4*s15 + 1) - (KS14*s4*s8)/(F_S16*s15 + 1) - (KS17*s2)/((F_S17*s4 + 1)*(F_S18*s5 + 1)) - (KS6*s4*s5*s13*s14)/(F_S5*u9 + 1),
                (KS11*s10)/(F_S13*s5 + 1) - (KS3*s10*s12)/((F_S1*u13 + 1)*(F_S2*u14 + 1)),
                KS39*u6 + KS43*u8 + KS10/(F_S10*u15 + 1) - (KS4*s10)/(F_S3*s15 + 1) - (KS11*s10)/(F_S13*s5 + 1) - (KS3*s10*s12)/((F_S1*u13 + 1)*(F_S2*u14 + 1)) - (KS19*s5*s24*s25)/((F_S21*s10 + 1)*(F_S20*s26 + 1)*(F_S22*u16 + 1)),
                KS16*u24 + (KS26*m31)/(F_S29*s29 + 1) - (KS27*s5)/(F_S30*s28 + 1) - (KS5*s4*s28)/(F_S4*s15 + 1),
                (KS5*s4*s28)/(F_S4*s15 + 1) - (KS11*s10)/(F_S13*s5 + 1) - (KS18*s4)/(F_S19*s5 + 1) - (KS27*s5)/(F_S30*s28 + 1) - KS2*s4*s11 - (KS17*s2)/((F_S17*s4 + 1)*(F_S18*s5 + 1)) - (KS6*s4*s5*s13*s14)/(F_S5*u9 + 1) - (KS19*s5*s24*s25)/((F_S21*s10 + 1)*(F_S20*s26 + 1)*(F_S22*u16 + 1)),
                KS28*m31*s29 - KS24/(F_S28*s13 + 1) - KS23/((F_S25*m21 + 1)*(F_S26*m22 + 1)*(F_S27*s13 + 1)) - (KS6*s4*s5*s13*s14)/(F_S5*u9 + 1),
                KS47*u4 + KS46*u11 + KS13/((F_S15*s3 + 1)*(F_S14*s18 + 1)) - (KS6*s4*s5*s13*s14)/(F_S5*u9 + 1),
                KS36*u6 - KS29*s3 - KS7*s3 - KS13/((F_S15*s3 + 1)*(F_S14*s18 + 1)) - (KS9*s7)/((F_S8*s3 + 1)*(F_S9*s6 + 1)) + (KS3*s10*s12)/((F_S1*u13 + 1)*(F_S2*u14 + 1)) - (KS8*s1*s2*s3*s6*s9)/((F_S6*s16 + 1)*(F_S7*s17 + 1)),
                KS53*u7 - (KS9*s7)/((F_S8*s3 + 1)*(F_S9*s6 + 1)) + (KS6*s4*s5*s13*s14)/(F_S5*u9 + 1) - (KS8*s1*s2*s3*s6*s9)/((F_S6*s16 + 1)*(F_S7*s17 + 1)),
                (KS9*s7)/((F_S8*s3 + 1)*(F_S9*s6 + 1)) - (KS8*s1*s2*s3*s6*s9)/((F_S6*s16 + 1)*(F_S7*s17 + 1)),
                KS32*u2 + KS33*u3 + KS34*u4 + KS35*u5 + KS2*s4*s11 - (KS17*s2)/((F_S17*s4 + 1)*(F_S18*s5 + 1)) - (KS8*s1*s2*s3*s6*s9)/((F_S6*s16 + 1)*(F_S7*s17 + 1)),
                KS7*s3 - (KS9*s7)/((F_S8*s3 + 1)*(F_S9*s6 + 1)),
                (KS14*s4*s8)/(F_S16*s15 + 1) - KS15*s19,
                KS50*u12 - KS25*s24 - KS22*s24 + (KS20*s8*s18)/((F_S23*s23 + 1)*(F_S11*u7 + 1)) - (KS19*s5*s24*s25)/((F_S21*s10 + 1)*(F_S20*s26 + 1)*(F_S22*u16 + 1)),
                KS21/(F_S24*s23 + 1) - (KS19*s5*s24*s25)/((F_S21*s10 + 1)*(F_S20*s26 + 1)*(F_S22*u16 + 1)),
                KS12*m31 + KS45*u9 + KS44*u24 - KS13/((F_S15*s3 + 1)*(F_S14*s18 + 1)) - (KS20*s8*s18)/((F_S23*s23 + 1)*(F_S11*u7 + 1)),
                (KS8*s1*s2*s3*s6*s9)/((F_S6*s16 + 1)*(F_S7*s17 + 1)) - (KS14*s4*s8)/(F_S16*s15 + 1),
                KS15*s19,
                (KS18*s4)/(F_S19*s5 + 1),
                (KS19*s5*s24*s25)/((F_S21*s10 + 1)*(F_S20*s26 + 1)*(F_S22*u16 + 1)) - (KS20*s8*s18)/((F_S23*s23 + 1)*(F_S11*u7 + 1)) - KS21/(F_S24*s23 + 1),
                KS22*s24 - (KS4*s10)/(F_S3*s15 + 1) - (KS5*s4*s28)/(F_S4*s15 + 1) - (KS14*s4*s8)/(F_S16*s15 + 1),
                (KS27*s5)/(F_S30*s28 + 1) - (KS26*m31)/(F_S29*s29 + 1) - KS28*m31*s29,
                KS25*s24,
                KS23/((F_S25*m21 + 1)*(F_S26*m22 + 1)*(F_S27*s13 + 1)),
                KS24/(F_S28*s13 + 1) + KS30/((F_S31*m21 + 1)*(F_S32*m22 + 1)),
                (KS17*s2)/((F_S17*s4 + 1)*(F_S18*s5 + 1)),
                KS29*s3 + KS54*u3 - (KS19*s5*s24*s25)/((F_S21*s10 + 1)*(F_S20*s26 + 1)*(F_S22*u16 + 1)),
                kbg1 - lg1 + (kg1*s4*s5*s6*s8*s26)/(FG1*s24 + 1),
                kbg2 - lg2 + kg2*s8,
                kbg3 - lg3 + kg3*s4*s5*s6*s8*s26,
                kbg4 - lg4 + (kg4*s8)/((FG2*s5 + 1)*(FG3*s18 + 1)),
                kbg5 - lg5 + kg5*s5*s6*s8*s26,
                kbg6 - lg6 + (kg6*s4*s5*s6*s8*s18*s26)/(FG4*s24 + 1),
                kbg7 - lg7 + kg7*s3,
                kbg8 - lg8 + (kg8*s4*s5*s8*s18*s26)/(FG5*s24 + 1),
                kbg9 - lg9 + kg9*s3*s26*u16,
                kbg10 - lg10 + kg10*s5*s6*s8*s26,
                kbg11 - lg11 + kg11*s6,
                kbg12 - lg12 + kg12*s5*s8*s26,
                kbg13 - lg13 + kg13*s5*s6*s8*s26,
                kbg14 - lg14 + kg14*s6*s8,
                kbg15 - lg15 + kg15*s8*s26,
                kbg16 - lg16 + (kg16*s8)/(FG6*u6 + 1),
                kbg17 - lg17 + kg17*s6*s8*s26,
                kbg18 - lg18 + (kg18*s5*s26)/((FG7*s8 + 1)*(FG8*s24 + 1)),
                kbg19 - lg19 + kg19*s1*s3*s26,
                kbg20 - lg20 + (kg20*s3*s5*s26*u16)/(FG9*s18 + 1),
                kbg21 - lg21 + kg21*s1*s2*s24*s26*u16,
                kbg22 - lg22 + (kg22*s2*s3)/((FG10*s5 + 1)*(FG11*s18 + 1)),
                kbg23 - lg23 + kg23*u10,
                kbg24 - lg24 + kg24*s8,
                kbg25 - lg25 + kg25*s26,
                kbg26 - lg26 + kg26*s1*s26,
                kbg27 - lg27 + kg27*s3*s26*u16,
                kbg28 - lg28 + (kg28*s5*s24*s26*u16)/(FG13*u2 + 1),
                kbg29 - lg29 + kg29*s2*s24*s26*u16,
                kbg30 - lg30 + kg30*s3*u16,
                kbg31 - lg31 + (kg31*s6)/(FG12*s24 + 1),
                kbg32 - lg32 + kg32*s24,
                kbg33 - lg33 + kg33*s6*s24,
                kbg34 - lg34 + kg34*s3*s5,
                kbg35 - lg35 + kg35*s6,
                kbg36 - lg36 + kg36*s6,
                kbg37 - lg37 + kg37*s1*s26*u16]
        
        
        return dydt






    def Predict(self, x):
        t = linspace(0, 0.0002, 5000)
        
            
        ################  CONSTANT PARTS  #####################

        ##### METABOLIC REACTION SUBSTRATE CONSTANT km in nM #####

        km0	 =	0.05564	
        km1	 =	0.4479	
        km2	 =	0.05966	
        km3	 =	0.052076	
        km4	 =	0.4479	
        km5	 =	0.052373	
        km6	 =	0.4289	
        km7	 =	0.054248	
        km8	 =	0.4654	
        km9	 =	0.053261	
        km10	 =	0.07412	
        km11	 =	0.052933	
        km12	 =	0.4415	
        km13	 =	0.09692	
        km14	 =	0.41931	
        km15	 =	0.051036	
        km16	 =	0.4729	
        km17	 =	0.13405	
        km18	 =	0.4658	
        km19	 =	0.053377	
        km20	 =	0.3951	
        km21	 =	0.073413	
        km22	 =	0.42947	
        km23	 =	0.13745	
        km24	 =	0.49851	
        km25	 =	0.054972	
        km26	 =	0.45491484	
        km27	 =	0.1576	
        km28	 =	0.074994	
        km29	 =	0.4786	
        km30	 =	0.052321	
        km31	 =	0.052518	
        km32	 =	0.42146	
        km33	 =	0.053936	
        km34	 =	0.063326	
        km35	 =	0.050974	
        km36	 =	0.4698	
        km37	 =	0.054877	
        km38	 =	0.051364	
        km39	 =	0.051125	
        km40	 =	0.093633	
        km41	 =	0.4922	
        km42	 =	0.062864	
        km43	 =	0.052887	
        km44	 =	0.084375	
        km45	 =	0.092682	
        km46	 =	0.052271	
        km47	 =	0.073521	
        km48	 =	0.063836	

        ##### METABOLIC REACTION ENZYME RATE CONATANT K in S^-1  #####

        K1	 =	0.09		   # glut1
        K2	 =	0.01		   # pgm_1
        K3	 =	0.09		   # hk
        K4	 =	0.01		   # g6Pase
        K5	 =	0.0812		# pgi
        K6	 =	0.09		   # pfk1
        K7	 =	0.017		   # f16Bpase
        K8	 =	0.09		   # pfk2
        K9	 =	0.0766		# f26Bpase
        K10	 =	0.0826		# ald
        K11	 =	0.0899		# tpi
        K12	 =	0.076		   # gcld3PD
        K13	 =	0.0532		# pglc_kn
        K14	 =	0.0755		# pglc_m
        K15	 =	0.0727		# enl
        K16	 =	0.05		   # pyrk
        K17	 =	0.01		   # lacd
        K18	 =	0 #0.0809	# pyrd                      ### Perturbation point for warburg effect ####### I have just stop the metabolic reaction. I do not change gene production and signaling molecules interaction.
        K19	 =	0 #0.0643	# acyl_cos_synthase         ### Perturbation point for warburg effect ####### I have just stop the metabolic reaction. I do not change gene production and signaling molecules interaction.
        K20	 =	0 #0.09		# fa_synthase               ### Perturbation point for warburg effect ####### I have just stop the metabolic reaction. I do not change gene production and signaling molecules interaction.
        K21	 =	0 #0.0852	# pyr_crbxylase             ### Perturbation point for warburg effect ####### I have just stop the metabolic reaction. I do not change gene production and signaling molecules interaction.
        K22	 =	0 #0.0296	# pep_crbxykinase1          ### Perturbation point for warburg effect ####### I have just stop the metabolic reaction. I do not change gene production and signaling molecules interaction.
        K23	 =	0.0818		# cit_synthase
        K24	 =	0.0844		# actnase
        K25	 =	0.0881		# isocit_deh
        K26	 =	0.0848		# KG_deh_cmp
        K27	 =	0 #0.09		# succ_coa_synthase             ### Perturbation point for warburg effect ####### I have just stop the metabolic reaction. I do not change gene production and signaling molecules interaction.
        K28	 =	0.0757		# succ_deh
        K29	 =	0.0766		# frmase
        K30	 =	0.0752		# mal_deh
        K31	 =	0.0603		# g6P_deh
        K32	 =	0.0191		# pglc6
        K33	 =	0.0658		# phglc_deh
        K34	 =	0.0792		# rbs_5Pis_A
        K35	 =	0.069		   # rbls_5P_3ep
        K36	 =	0.0892		# trkl
        K37	 =	0.0722		# trsdl_1

        ##### SIGNALLING MOLECULE BINDING RATE CONSTANT KS in per second #####
        ### Randomly generated between 0.001 to 0.009 
        KS1	 =	0.0013	
        KS2	 =	0.0059	
        KS3	 =	0.0039	
        KS4	 =	0.0014	
        KS5	 =	0.0049	
        KS6	 =	0.0025	
        KS7	 =	0.002	
        KS8	 =	0.0016	
        KS9	 =	0.0022	
        KS10	 =	0.0025	
        KS11	 =	0.0013	
        KS12	 =	0.0061	
        KS13	 =	0.0033	
        KS14	 =	0.0053	
        KS15	 =	0.0066	
        KS16	 =	0.005	
        KS17	 =	0.0053	
        KS18	 =	0.0046	
        KS19	 =	0.002	
        KS20	 =	0.0049	
        KS21	 =	0.0078	
        KS22	 =	0.008	
        KS23	 =	0.0032	
        KS24	 =	0.0027	
        KS25	 =	0.0055	
        KS26	 =	0.0061	
        KS27	 =	0.0043	
        KS28	 =	0.0026	
        KS29	 =	0.0086	
        KS30	 =	0.0017	
        KS31	 =	0.0018	
        KS32	 =	0.0021	
        KS33	 =	0.0023	
        KS34	 =	0.006	
        KS35	 =	0.0056	
        KS36	 =	0.0014	
        KS37	 =	0.0084	
        KS38	 =	0.0068	
        KS39	 =	0.0069	
        KS43	 =	0.0015	
        KS44	 =	0.0079	
        KS45	 =	0.0085	
        KS46	 =	0.0089	
        KS47	 =	0.0079	
        KS48	 =	0.0073	
        KS49	 =	0.0051	
        KS50	 =	0.0024	
        KS52	 =	0.0042	
        KS53	 =	0.0021	
        KS54	 =	0.0012	

        ##### GENE BASAL PRODUCTION RATE CONSTANT kbg in per second #####

        kbg1	 =	0.0004939		#	glut1
        kbg2	 =	0.0004301		#	pgm_1
        kbg3	 =	0.0004296		#	hk
        kbg4	 =	0.0004333		#	g6Pase
        kbg5	 =	0.0004467		#	pgi
        kbg6	 =	0.0004648		#	pfk1
        kbg7	 =	0.0004025		#	f16Bpase
        kbg8	 =	0.0004842		#	pfk2
        kbg9	 =	0.0004559		#	f26Bpase
        kbg10	 =	0.0004854		#	ald
        kbg11	 =	0.0004348		#	tpi
        kbg12	 =	0.0004446		#	gcld3PD
        kbg13	 =	0.0004054		#	pglc_kn
        kbg14	 =	0.0004177		#	pglc_m
        kbg15	 =	0.0004663		#	enl
        kbg16	 =	0.0004331		#	pyrk
        kbg17	 =	0.0004898		#	lacd
        kbg18	 =	0.0004118		#	pyrd                          ### Perturbation point for warburg effect... but basal production should not be changed I think
        kbg19	 =	0.0004988		#	acyl_cos_synthase             ### Perturbation point for warburg effect... but basal production should not be changed I think
        kbg20	 =	0.000454		   #	fa_synthase                   ### Perturbation point for warburg effect... but basal production should not be changed I think
        kbg21	 =	0.0004707		#	pyr_crbxylase                 ### Perturbation point for warburg effect... but basal production should not be changed I think
        kbg22	 =	0.0004999		#	pep_crbxykinase1              ### Perturbation point for warburg effect... but basal production should not be changed I think
        kbg23	 =	0.0004288		#	cit_synthase
        kbg24	 =	0.0004415		#	actnase
        kbg25	 =	0.0004465		#	isocit_deh
        kbg26	 =	0.0004764		#	KG_deh_cmp
        kbg27	 =	0.0004818		#	succ_coa_synthase             ### Perturbation point for warburg effect... but basal production should not be changed I think
        kbg28	 =	0.00041		   #	succ_deh
        kbg29	 =	0.0004178		#	frmase
        kbg30	 =	0.000436		   #	mal_deh
        kbg31	 =	0.0004057		#	g6P_deh
        kbg32	 =	0.0004522		#	pglc6
        kbg33	 =	0.0004336		#	phglc_deh
        kbg34	 =	0.0004176		#	rbs_5Pis_A
        kbg35	 =	0.0004209		#	rbls_5P_3ep
        kbg36	 =	0.0004905		#	trkl
        kbg37	 =	0.0004675		#	trsdl_1
        
        
        ##### GENE TRANSCRIPTION FACTORS BINDING RATE CONSTANT kg in per second #####
        
        kg1	=	0.0007405	
        kg2	=	0.0008736	
        kg3	=	0.0006312	
        kg4	=	0.0008237	
        kg5	=	0.0008209	
        kg6	=	0.0007686	
        kg7	=	0.0006553	
        kg8	=	0.0007792	
        kg9	=	0.00069	
        kg10	=	0.0006402	
        kg11	=	0.0006638	
        kg12	=	0.0008685	
        kg13	=	0.0006214	
        kg14	=	0.0006727	
        kg15	=	0.0006161	
        kg16	=	0.0007325	
        kg17	=	0.000604	
        kg18	=	0.0008692	   ### Perturbation point for warburg effect... but production should not be changed I think
        kg19	=	0.000659	      ### Perturbation point for warburg effect... but production should not be changed I think                         
        kg20	=	0.000628	      ### Perturbation point for warburg effect... but production should not be changed I think
        kg21	=	0.0006922	   ### Perturbation point for warburg effect... but production should not be changed I think
        kg22	=	0.0007368	   ### Perturbation point for warburg effect... but production should not be changed I think
        kg23	=	0.0006305	
        kg24	=	0.0008986	
        kg25	=	0.0006996	
        kg26	=	0.0006892	
        kg27	=	0.0006186	   ### Perturbation point for warburg effect... but production should not be changed I think
        kg28	=	0.0006895	
        kg29	=	0.0006139	
        kg30	=	0.0007516	
        kg31	=	0.0008284	
        kg32	=	0.0007893	
        kg33	=	0.000627	
        kg34	=	0.0006243	
        kg35	=	0.0008332	
        kg36	=	0.0008715	
        kg37	=	0.0007601	
        
        ##### GENE DECAY RATE CONSTANT lg in per second #####
        
        lg1=	0.0001109		#	glut1
        lg2=	0.0001826		#	pgm_1
        lg3=	0.0001338		#	hk
        lg4=	0.0001294		#	g6Pase
        lg5=	0.0001746		#	pgi
        lg6=	0.000101		   #	pfk1
        lg7=	0.0001048		#	f16Bpase
        lg8=	0.0001668		#	pfk2
        lg9=	0.0001603		#	f26Bpase
        lg10=	0.0001526		#	ald
        lg11=	0.000173		   #	tpi
        lg12=	0.0001707		#	gcld3PD
        lg13=	0.0001781		#	pglc_kn
        lg14=	0.0001288		#	pglc_m
        lg15=	0.0001693		#	enl
        lg16=	0.0001557		#	pyrk
        lg17=	0.0001397		#	lacd
        lg18=	0.0001062		#	pyrd
        lg19=	0.000178		   #	acyl_cos_synthase
        lg20=	0.0001338		#	fa_synthase
        lg21=	0.0001608		#	pyr_crbxylase
        lg22=	0.0001741		#	pep_crbxykinase1
        lg23=	0.0001105		#	cit_synthase
        lg24=	0.0001128		#	actnase
        lg25=	0.000155		   #	isocit_deh
        lg26=	0.0001485		#	KG_deh_cmp
        lg27=	0.000189		   #	succ_coa_synthase
        lg28=	0.0001799		#	succ_deh
        lg29=	0.0001734		#	frmase
        lg30=	0.0001051		#	mal_deh
        lg31=	0.0001073		#	g6P_deh
        lg32=	0.0001089		#	pglc6
        lg33=	0.0001798		#	phglc_deh
        lg34=	0.0001943		#	rbs_5Pis_A
        lg35=	0.0001684		#	rbls_5P_3ep
        lg36=	0.0001132		#	trkl
        lg37=	0.0001723		#	trsdl_1
        
        ##### Feedback parameters for Metabolic Network ######
        #### F49, F50 is not included intentionally
        F1	=	0.9            # 0.2435
        F2	=	0.9            #0.9293
        F3	=	0.9            #0.3500
        F4	=	0.9            #0.1966
        F5	=	0.9            #0.2511
        F6	=	0.9            #0.6160
        F7	=	0.9            #0.4733
        F8	=	0.9            #0.3517
        F9	=	0.9            #0.8308
        F10	=	0.9            #0.5853
        F11	=	0.9            #0.5497
        F12	=	0.9            #0.9172
        F13	=	0.9            #0.2858
        F14_modified =	0.9   #0.7572
        F15	=	0.9            #0.7537
        F16	=	0.9            #0.3804
        F17	=	0.9            #0.5678
        F18	=	0.9            #0.0759
        F19	=	0.9            #0.0540
        F20	=	0.9            #0.5308
        F21	=	0.9            #0.7792
        F22	=	0.9            #0.9340
        F23	=	0.9            #0.1299
        F24	=	0.9            #0.5688
        F25	=	0.9            #0.4694
        F26	=	0.9            #0.0119
        F27	=	0.9            #0.3371
        F28	=	0.9            #0.1622
        F29	=	0.9            #0.7943
        F30	=	0.9            #0.3112
        F31	=	0.9            #0.5285
        F32	=	0.9            #0.1656
        F33	=	0.9            #0.6020
        F34	=	0.9            #0.2630
        F35	=	0.9            #0.6541
        F36	=	0.9            #0.6892
        F37	=	0.9            #0.7482
        F38	=	0.9            #0.4505
        F39	=	0.9            #0.0838
        F40	=	0.9            #0.2290
        F41	=	0.9            #0.9133
        F42	=	0.9            #0.1524
        F43	=	0.9            #0.8258
        F44	=	0.9            #0.5383
        F45	=	0.9            #0.9961
        F46	=	0.9            #0.0782
        F47	=	0.9            #0.4427
        F48	=	0.9            #0.1067
        
        F51	=	0.9            #0.7749
        F52	=	0.9            #0.8173
        F53	=	0.9            #0.8687
        F54	=	0.9            #0.0844
        F55	=	0.9            #0.0716
        
        ##### Feedback parameters for Signalling Network ######
        #### FS_12 is not included intentionally
        F_S1	=	0.9         #0.3998	
        F_S2	=	0.9         #0.2599	
        F_S3	=	0.9         #0.8001	
        F_S4	=	0.9         #0.4314	
        F_S5	=	0.9         #0.9106	
        F_S6	=	0.9         #0.1818	
        F_S7	=	0.9         #0.2638	
        F_S8	=	0.9         #0.1455	
        F_S9	=	0.9         #0.1361	
        F_S10	=	0.9         #0.8693	
        F_S11	=	0.9         #0.5797	
        F_S13	=	0.9         #0.145	
        F_S14	=	0.9         #0.853	
        F_S15	=	0.9         #0.6221	
        F_S16	=	0.9         #0.351	
        F_S17	=	0.9         #0.5132	
        F_S18	=	0.9         #0.4018	
        F_S19	=	0.9         #0.076	
        F_S20	=	0.9         #0.2399	
        F_S21	=	0.9         #0.1233	
        F_S22	=	0.9         #0.1839	
        F_S23	=	0.9         #0.24	
        F_S24	=	0.9         #0.4173	
        F_S25	=	0.9         #0.0497	
        F_S26	=	0.9         #0.9027	
        F_S27	=	0.9         #0.9448	
        F_S28	=	0.9         #0.4909	
        F_S29	=	0.9         #0.4893	
        F_S30	=	0.9         #0.3377	
        F_S31	=	0.9         #0.9001	
        F_S32	=	0.9         #0.3692	
        
        
        ##### Feedback parameters for Gene Regulatory Network ######
        
        FG1	= 0.9             #0.5289
        FG2	= 0.9             #0.6944
        FG3	= 0.9             #0.2124
        FG4	= 0.9             #0.5433
        FG5	= 0.9             #0.7025
        FG6	= 0.9             #0.9564
        FG7	= 0.9             #0.4445
        FG8	= 0.9             #0.0854
        FG9	= 0.9             #0.0573
        FG10 = 0.9            #0.6295
        FG11 = 0.9            #0.7962
        FG12 = 0.9            #0.6912
        FG13 = 0.9            #0.3453
        
        
        
        km  = [km0,km1,km2,km3,km4,km5,km6,km7,km8,km9,km10,km11,km12,km13,km14,km15,km16,km17,km18,km19,km20,km21,km22,km23,km24,km25,km26,km27,km28,km29,km30,km31,km32,km33,km34,km35,km36,km37,km38,km39,km40,km41,km42,km43,km44,km45,km46,km47,km48]
        K   = [K1,K2,K3,K4,K5,K6,K7,K8,K9,K10,K11,K12,K13,K14,K15,K16,K17,K18,K19,K20,K21,K22,K23,K24,K25,K26,K27,K28,K29,K30,K31,K32,K33,K34,K35,K36,K37]
        KS  = [KS1,KS2,KS3,KS4,KS5,KS6,KS7,KS8,KS9,KS10,KS11,KS12,KS13,KS14,KS15,KS16,KS17,KS18,KS19,KS20,KS21,KS22,KS23,KS24,KS25,KS26,KS27,KS28,KS29,KS30,KS31,KS32,KS33,KS34,KS35,KS36,KS37,KS38,KS39,KS43,KS44,KS45,KS46,KS47,KS48,KS49,KS50,KS52,KS53,KS54]
        kbg = [kbg1,kbg2,kbg3,kbg4,kbg5,kbg6,kbg7,kbg8,kbg9,kbg10,kbg11,kbg12,kbg13,kbg14,kbg15,kbg16,kbg17,kbg18,kbg19,kbg20,kbg21,kbg22,kbg23,kbg24,kbg25,kbg26,kbg27,kbg28,kbg29,kbg30,kbg31,kbg32,kbg33,kbg34,kbg35,kbg36,kbg37]         
        kg  = [kg1,kg2,kg3,kg4,kg5,kg6,kg7,kg8,kg9,kg10,kg11,kg12,kg13,kg14,kg15,kg16,kg17,kg18,kg19,kg20,kg21,kg22,kg23,kg24,kg25,kg26,kg27,kg28,kg29,kg30,kg31,kg32,kg33,kg34,kg35,kg36,kg37]
        lg  = [lg1,lg2,lg3,lg4,lg5,lg6,lg7,lg8,lg9,lg10,lg11,lg12,lg13,lg14,lg15,lg16,lg17,lg18,lg19,lg20,lg21,lg22,lg23,lg24,lg25,lg26,lg27,lg28,lg29,lg30,lg31,lg32,lg33,lg34,lg35,lg36,lg37]
        F   = [F1,F2,F3,F4,F5,F6,F7,F8,F9,F10,F11,F12,F13,F14_modified,F15,F16,F17,F18,F19,F20,F21,F22,F23,F24,F25,F26,F27,F28,F29,F30,F31,F32,F33,F34,F35,F36,F37,F38,F39,F40,F41,F42,F43,F44,F45,F46,F47,F48,F51,F52,F53,F54,F55]
        F_S = [F_S1,F_S2,F_S3,F_S4,F_S5,F_S6,F_S7,F_S8,F_S9,F_S10,F_S11,F_S13,F_S14,F_S15,F_S16,F_S17,F_S18,F_S19,F_S20,F_S21,F_S22,F_S23,F_S24,F_S25,F_S26,F_S27,F_S28,F_S29,F_S30,F_S31,F_S32]
        FG  = [FG1,FG2,FG3,FG4,FG5,FG6,FG7,FG8,FG9,FG10,FG11,FG12,FG13]
        
        u1,u2,u3,u4,u5,u6,u7,u8,u9,u10,u11,u12,u13,u14,u15,u16,u17,u18,u19,u20,u21,u22,u23,u24,u25,u26,u27 = x[107:]

        sol = odeint(self.integrated_network, x[:107], t, args=(km,K,KS,kbg,kg,lg,F,F_S,FG,u1,u2,u3,u4,u5,u6,u7,u8,u9,u10,u11,u12,u13,u14,u15,u16,u17,u18,u19,u20,u21,u22,u23,u24,u25,u26,u27))
        end = len(sol)
        sol[end-1]=np.real(sol[end-1])
        
        for j in range(0,107):
            if sol[end-1,j] < 0.01:
                sol[end-1,j] = 0.01 
   
        for j in range(0,107):
            if sol[end-1,j] > 1.0:
                sol[end-1,j] = 1.0 
        
        
        
        return sol[end-1]
        