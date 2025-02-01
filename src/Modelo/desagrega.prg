
%201="2007.01" 'PRIMER DATO MENSUAL
%202="2025.12" 'ULTIMO DATO MENSUAL DE PREDICCIÓN
%301="2007.1" 'PRIMER DATO TRIMESTRAL
%302="2024.3"'ULTIMO DATO TRIMESTRAL DISPONIBLE
%303="2025.4"'ULTIMO DATO TRIMESTRAL A PREDECIR
WORKFILE TRIMESTRAL
SMPL {%302}+1 {%303} 
!t=@OBS(UNO)

'el punto de partida es una ecuacion estimada cuyo nombre se introduce como variable.
SMPL @FIRST @LAST

for %1  CPR CPU INV EXP IMP AGRI INDU CST SER IMPU 
SMPL {%301} {%302}
equation e{%1}
e{%1}.makeresid residuo
SCALAR SIGMA=@SE^2
SCALAR RO=1-(@dw/2)

scalar filas=@obs(residuo)
smpl if residuo<>na
e{%1}.makeregs group1
scalar columnas=group1.@count
!c=columnas
genr uno=1
%90=group1.@seriesname(1)

if !c>1 then
%91=group1.@seriesname(2)
endif
if !c>2 then
%92=group1.@seriesname(3)
endif
if !c>3 then
%93=group1.@seriesname(4)
endif
if !c>4 then
%94=group1.@seriesname(5)
endif
if !c>5 then
%95=group1.@seriesname(6)
endif
if !c>6 then
%96=group1.@seriesname(7)
endif
if !c>7 then
%97=group1.@seriesname(8)
endif
if !c>8 then
%98=group1.@seriesname(9)
endif
if !c>9 then
%99=group1.@seriesname(10)
endif
if !c>10 then
%100=group1.@seriesname(11)
endif
if !c>11 then
%101=group1.@seriesname(12)
endif
if !c>12 then
%102=group1.@seriesname(13)
endif
smpl if residuo<>na
scalar filasa=filas+!T
scalar columnas=columnas-1
genr endo={%90}
stom(endo,Y)
group1.drop {%90}
stom(group1,x)
copy Y Mensual::
copy x  Mensual::
copy filas  Mensual::
copy filasa  Mensual::
copy columnas  Mensual::
copy sigma  Mensual::
copy ro Mensual::
'd group1 uno un residuo FILAS FILASA columnas endo ro sigma x y
WORKFILE Mensual
Vector(3) V3=1
vector v3=v3*(1/3)
matrix b=@kronecker(@identity(filas),@transpose(v3))
matrix ba=@kronecker(@identity(filasa),@transpose(v3))


if ro<=0 then
scalar rot=0
'Fernandez Procedure
Matrix(3*filas,3*filas) Difer
Matrix(3*filasa,3*filasa) Difera
Vector(3*filas) dmas=1
Vector(3*filas-1) dmenos=-1
Vector(3*filasa) dmasa=1
Vector(3*filasa-1) dmenosa=-1
Difer=@makediagonal(dmas)+@makediagonal(dmenos,-1)
Matrix q=@inverse(@transpose(difer)*difer)
Difera=@makediagonal(dmasa)+@makediagonal(dmenosa,-1)
Matrix qa=@inverse(@transpose(difera)*difera)
endif
if ro>0 then
scalar rot=exp(-0.00302073936725 + 0.459080195566*LOG(RO) - 0.0517950701174*LOG(RO)^2 + 0.0114881891532*LOG(RO)^3 + 0.0019942146442*LOG(RO)^4 + 4.58020371897e-05*LOG(RO)^5)
'Chow-Lin procedure
endif
MATRIX (3*filas,3*filas) M1
MATRIX (3*filasa,3*filasa) M1a
m1=@identity(3*filas)
m1a=@identity(3*filasa)
for !1=1 to 3*filas-1 
VECTOR (3*filas-!1) RHO=rot^!1
m1=m1+@makediagonal(rho,-!1) + @makediagonal(rho,+!1)
next
for !1=1 to 3*filasa-1 
VECTOR (3*filasA-!1) RHOa=rot^!1
m1a=m1a+@makediagonal(rhoa,-!1) + @makediagonal(rhoa,+!1)
next
if rot>0 then
matrix vMENS=M1*(SIGMA/(1-(EXP(LOG(ROt)))^2))
matrix vMENSa=M1a*(SIGMA/(1-(EXP(LOG(ROt)))^2))
endif
if rot<=0 then
matrix vMENS=q*SIGMA
matrix vMENSa=qa*SIGMA
endif


MATRIX V=B*VMENS*@TRANSPOSE(B)
MATRIX VA=Ba*VMENSa*@TRANSPOSE(Ba)
smpl {%201} {%202} '+filasA*3+1
genr cons=1
GROUP REGmens {%91} {%92} {%93} {%94} {%95} {%96} {%97} {%98} {%99} {%100} {%101} {%102} 
STOM(REGmens,Xmens)
MATRIX BETAG=(@INVERSE(@TRANSPOSE(X)*X))*(@TRANSPOSE(X)*Y)
MATRIX ERRmens=Y-(X*BETAG)

MATRIX AJUSTE=ERRMENS(FILAS,1)
MATRIX (filasa,1) ERRmensa
MATPLACE(ERRMENSA,ERRMENS,1,1)
FOR !2=FILAS+1 TO FILASA
MATPLACE(ERRMENSA,AJUSTE,!2,1)
NEXT

VECTOR Ymens=Xmens*BETAG +vmensA*@transpose(bA)*@inverse(vA)*ERRmensA
%20="M_"+%1
MTOS(Ymens,{%20})
%90="  "
%91="  "
%92="  "
%93="  "
%94="  "
%95="  "
%96="  "
%97="  "
%98="  "
%99="  "
%100="  "
WORKFILE TRIMESTRAL
SMPL @FIRST @LAST
next
WORKFILE MENSUAL
SMPL @FIRST @LAST
GENR m_PIBDEM=0.596*m_CPR+0.166*M_CPU+0.267*M_INV+0.286*M_EXP-0.315*M_IMP
GENR m_PIBOFE=M_AGRI*0.0249973+M_INDU*0.1472338+M_CST*0.0539303+M_SER*0.6866292+M_IMPU*0.0872094

