%301="2007.1" 'PRIMER DATO TRIMESTRAL MENSUALIZADO
%302="2024.3" 'ULTIMO DATO DISPONIBLE DE LA CONTABILIDAD
%303="2025.4" 'ULTIMO DATO DE PREDICCIÓN TRIMESTRAL
%201="2007.01" 'PRIMER DATO MENSUALIZADO
%202="2024.09" 'ULTIMO DATO DISPONIBLE DE LA CONTABILIDAD MENSUALIZADO
%203="2025.12" 'ULTIMO DATO DE PREDICCIÓN TRIMESTRAL
!anos_c=17 'número de años completos de CTR desde 2007 hasta el último año completo de la CTR
!an=19  'número de años disponibles de datos mensualizados desde 2007 hasta 2025
!tri=!an*4
!mes=!an*12
!ser=10
!p_ofe=0.5
!P_dem=1-!p_ofe

WORKFILE Trimestral
smpl {%302}+1 {%303}
!q=@obs(pib)


smpl {%301} {%302}
genr ipib=@pchy(pib)*100
stom(ipib,vtPIB)
copy vtPIB mENSUAL::

'comienzo del proceso de ajuste oferta demanda

workfile mensual
'carga de datos de pesos anuales se cargan tantos datos como años completos !anos_c
smpl 1995.01 1995.01+!anos_c 
IMPORT AFESP.xlsx  range="Nominal_a" @freq m 1995
smpl 1995.01+!anos_c+1  1995.01+!an-1
for %1  CPR CPU INV EXP IMP AGRI INDU CST IMPU SER
genr an_{%1}= an_{%1}(-1)
next
smpl 1995.01 1995.01+!an-1

vector(4) V4=1
Vector(12) v12=1

Matrix(!tri,!tri*!ser) htw =0

Matrix(((!tri*(!ser-1))+!mes),!mes*!ser) hmw=0

VECTOR(!tri) CERO
rowvector agre=@filledrowvector(3,1/3)
matrix Bagre=@kronecker(@identity(!tri),agre)

MATRIX bmagre=@kronecker(@identity(!tri*(!ser-1)),agre)

Matrix(!mes,!mes) Difer
Vector(!mes) dmas=1
Vector(!mes-1) dmenos=-1

Matrix(!tri,!tri) Difert
Vector(!tri) dmast=1
Vector(!tri-1) dmenost=-1
 
Vector((!tri*(!ser-1))+!mes) W =0
scalar n_colt=1
scalar n_colm=1
for %1  CPR CPU INV EXP IMP AGRI INDU CST IMPU SER
stom(an_{%1},p_{%1})
vector WT_{%1}=@kronecker(p_{%1},v4)
vector WM_{%1}=@kronecker(p_{%1},v12)
Matrix WDT_{%1}=@MAKEDIAGONAL(WT_{%1})
Matrix WDM_{%1}=@MAKEDIAGONAL(WM_{%1})
Matplace(HTW,WDT_{%1},1,n_colt)
Matplace(HMW,WDM_{%1},1,n_colm)

n_colt=n_colt+!tri

n_colm=n_colm+!mes
next

Matplace(hmw,bmagre,!mes+1,1)

smpl {%201} {%203}
For %1 M_CPR M_CPU M_INV M_EXP M_IMP M_AGRI M_INDU M_CST  M_IMPU M_SER
{%1}.hpf(lambda=2) {%1}H suavizado

next

For %1    AGRI INDU CST  IMPU SER
genr I_M_{%1}h=-M_{%1}H
genr I_M_{%1}=-M_{%1}
next

Group M_ini M_CPRh M_CPUH M_INVH M_EXPH M_IMPH I_M_AGRIH I_M_INDUH I_M_CSTH  I_M_IMPUH I_M_SERH
Group M_inir M_CPR M_CPU M_INV M_EXP M_IMP I_M_AGRI I_M_INDU I_M_CST  I_M_IMPU I_M_SER

stom(M_ini,M_Y_ini)
stom(M_inir,M_Y_inir)
Matrix T_Y_ini=bagre*M_y_inir
vector VT_Y_Ini=@vec(T_Y_ini)
vector dem_T_ini=@subextract(vT_Y_ini,1,1,(!tri*!ser/2),1)
vector ofe_T_ini=-@subextract(vT_Y_ini,((!tri*!ser/2)+1),1,(!tri*!ser),1)

Difert=@makediagonal(dmast)+@makediagonal(dmenost,-1)
Matrix qt=@transpose(difert)*difert
Matrix omegat=@kronecker(@identity(5),@inverse(qt))
Matrix HTW_d=@subextract(htw,1,1,!tri,(!tri*!ser/2))
Matrix HTW_o=@subextract(htw,1,((!tri*!ser/2)+1),!tri,!tri*!ser)
Vector Pibdem_ini=htw_d*dem_t_ini
Vector Pibofe_ini=htw_o*ofe_t_ini
Vector PIB_T=!p_dem*pibdem_INI+!p_ofe*pibofe_INI
Matplace (PIB_T,vtpib,1,1)

Vector DEM_T=dem_T_ini+(omegat*@TRANSPOSE(HTW_d)*@INVERSE(HTW_d*omegat*@TRANSPOSE(HTW_d))*(PIB_t-(HTW_d*DEM_T_INI)))

Vector OFE_T=OFE_T_ini+(omegat*@TRANSPOSE(HTW_O)*@INVERSE(HTW_O*omegat*@TRANSPOSE(HTW_O))*(PIB_t-(HTW_O*OFE_T_INI)))

vector dem_T_ini=@subextract(vT_Y_ini,1,1,(!tri*!ser/2),1)
vector ofe_T_ini=-@subextract(vT_Y_ini,((!tri*!ser/2)+1),1,(!tri*!ser),1)

vector(!tri*!ser) VT_Y=0
MATPLACE(VT_Y,DEM_T_ini,1,1)
MATPLACE(VT_Y,-OFE_T_ini,((!tri*!ser/2)+1),1)
Vector VM_Y_ini=@vec(M_y_inir)
Difer=@makediagonal(dmas)+@makediagonal(dmenos,-1)
Matrix q=@transpose(difer)*difer
Matrix omega=@kronecker(@identity(10),@inverse(q))

MATPLACE(W,@SUBEXTRACT(vt_Y,1,1,(!tri*(!ser-1)),1),!mes+1,1)
VECTOR VM_Y=VM_Y_INI+(OMEGA*@TRANSPOSE(HMW)*@INVERSE(HMW*OMEGA*@TRANSPOSE(HMW)) *(W-(HMW*VM_Y_INI ) )) 

Matrix(!mes,10) m_Y
For !w=1 to 10
Matplace(M_Y,@SUBEXTRACT(Vm_Y,((!W-1)*(!mes-1)+!w),1,(((!W-1)*!mes)+!mes),1), 1,!w)
next

Vector demanda=@SUBEXTRACT(vm_Y,1,1,5*!mes,1)
Vector Oferta=-@SUBEXTRACT(vm_Y,((5*!mes)+1),1)
Matrix pesodem=@SUBEXTRACT(HMW,1,1,!mes,5*!mes)
Matrix pesoOFE=@SUBEXTRACT(HMW,1,((5*!mes)+1),!mes,!mes*!ser)
VECTOR PIBDEM=PESODEM*DEMANDA
VECTOR PIBOFE=PESOOFE*OFERTA
VECTOR pib_t_INI=BAGRE*PIBDEM
vector PIB_T=BAGRE*PIBDEM
Matplace (PIB_T,vtpib,1,1)

VECTOR DIF_PIB_T=PIB_T-pib_t_ini
matrix(!mes+!tri,!mes+!tri) BFL
matplace(BFL,q,1,1)
matplace(bfl,bagre,!mes+1,1)
matplace(bfl,@transpose(bagre),1,!mes+1)
bfl(1,1)=1
matrix(!mes+!tri,1) ceroerr
matplace (ceroerr,dif_pib_t,!mes+1,1)
matrix SUAVIZADO=@INVERSE(BFL)*CEROERR
VECTOR PIB_fin=PIBDEM+@SUBEXTRACT(Suavizado,1,1,!mes,1)

d finales
group finales
smpl {%201} {%203}
for %1 MF_CPR MF_CPU MF_INV MF_EXP MF_IMP I_MF_AGRI I_MF_INDU I_MF_CST  I_MF_IMPU I_MF_SER
genr {%1}=0
finales.add {%1}
next
MTOS(m_Y,finales)
mtos(pibdem,Mf_PIBDEM)
mtos(pibofe,Mf_PIBOFE)
mtos(pib_fin,Mf_PIB)
For %1    AGRI INDU CST IMPU SER 
genr MF_{%1}=-I_MF_{%1}
next
'd v4  vtpib v12 htw htw_d htw_o  hmw cero agre bagre BMAGRE difer dmas dmenos dmast dmenost w n_colt n_colm M_Y M_Y_ini T_Y_ini VT_Y_ini VT_Y VM_Y_INI Q OMEGA OMEGAt  qt vm_y  DEMANDA OFERTA PESODEM PESOOFE PIBDEM PIBOFE dem_t dem_t_ini ofe_t ofe_t_ini pib_t pibdem_ini pibofe_ini
for %1  CPR CPU INV EXP IMP AGRI INDU CST  IMPU SER
d wt_{%1} wm_{%1} wdt_{%1} wdm_{%1}  p_{%1}
NEXT
for %1   AGRI INDU CST  IMPU SER
d i_M_{%1} i_MF_{%1} 
NEXT
if @isobject("prediccion")=1 then
d prediccion
endif
group prediccion
for %5 PIB CPR CPU INV EXP IMP AGRI INDU CST  IMPU SER
prediccion.add MF_{%5} 
NEXT


smpl {%201} {%203}
write(t=xls) prediccion.xls prediccion
SMPL @FIRST @LAST
WFSAVE Mensual
workfile Trimestral
WFSAVE Trimestral

