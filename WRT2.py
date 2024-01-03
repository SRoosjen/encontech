# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 08:34:08 2020

@author: sander
"""
import numpy as np
from CoolProp.CoolProp import PropsSI
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import pandas as pd 
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def worthington(dP,T_LOW,T_HIGH,BACKEND,FLUID1,FLUID2,MASS,X0):  
    if X0 == 0:
        FLUID=BACKEND+'::'+FLUID1
    elif X0 == 1:
        FLUID=BACKEND+'::'+FLUID2
    else:
        FLUID=BACKEND+'::'+FLUID1+'['+str(np.round(X0,3))+']&'+FLUID2+\
        '['+str(np.round(1-X0,3))+']'

    try:
        P_LOW = PropsSI('P','Q',0,'T', T_LOW,FLUID)   #Set P_LOW to Bubble pressure 
        
        P_HIGH = P_LOW + dP
    
        Rho = PropsSI('Dmass','T',T_HIGH,'P',P_HIGH,FLUID)       #calculate density at high cycle temp and pres [kg/m3]
        V_0 = MASS / Rho                                   #calculate correspoding volume for defined mass m [m3]
        
        W = V_0*(P_HIGH-P_LOW)                               #calculate work
        
        T_C = PropsSI('T','S',PropsSI('S','T',T_LOW,'P',P_LOW,FLUID),'P',P_HIGH,FLUID)    #T_c is the temperature of the fluid after isentropic compression
        
        W_P = PropsSI('H','T',T_C,'P',P_HIGH,FLUID) - PropsSI('H','T',T_LOW,'P',P_LOW,FLUID) #pumping work is the difference of the enthalpies
        #W_P2 = (P_HIGH-P_LOW)/PropsSI('Dmass','T',T_LOW,'P',P_LOW,FLUID)  #Approximate W_P since fluid is incompressible (better to take enthalpy difference)
        
        T_D = PropsSI('T','H',(PropsSI('H','T',T_HIGH,'P',P_HIGH,FLUID) - W),'P',P_LOW,FLUID)  #temperature of the vapor discharging from the power cylinder 
        T_S = 0 #PropsSI('T','P',P_LOW,'Q',0,FLUID)
    
                   
        H12_Hold = [] 
        N=50
        T_List = np.linspace(T_C,T_D,N)
        for T in T_List:
            H1 = PropsSI('H','P',P_LOW,'T',T,FLUID)
            H2 = PropsSI('H','P',P_HIGH,'T',T,FLUID)
            H12_Hold.append(H1-H2)
            
        DH = np.max(H12_Hold)               
        T_Pinch = T_List[H12_Hold.index(DH)]
    
        Q_H = PropsSI('H','T',T_HIGH,'P',P_HIGH,FLUID) - PropsSI('H','T',T_C,'P',P_HIGH,FLUID)
        Q_R = PropsSI('H','T',T_D,'P',P_LOW,FLUID) - DH - PropsSI('H','T',T_C,'P',P_HIGH,FLUID)   
    
        etaC = 1 - np.sqrt(T_LOW/T_HIGH) #novikov
                
        eta0 = (W - W_P) / Q_H
        etaR = (W - W_P) / (Q_H - Q_R) #efficiency
        
        if (eta0 < 0.0002) or (etaR < eta0):
            etaR = eta0
    
        return P_LOW,P_HIGH,etaC,eta0,etaR,V_0,T_C,W,W_P,T_D,T_S,DH,Q_H,Q_R,T_Pinch,FLUID
    except Exception as e:
        print(e)
        return -1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,'0'


if __name__ == "__main__":
    BACKEND = 'REFPROP' 
    FLUID1 = 'R134a' 
    FLUID2 = ''
    
    T_LOW=10.75 + 273.15       #ignored if 
    T_HIGH=50 + 273.15
    T_STEP = 0.1
    
    TL_List = [T_LOW] #,65+273.15,100+273.15]
    TH_List = np.arange(TL_List[0]+5,T_HIGH+2,T_STEP) #
    dP_List = [7.7e5] #np.linspace(P_LOW+5e5,P_LOW+35e5,30) 16.2
    X0_List = [0.0] #np.linspace(0,1,6)
    
#    P_LOW=12.0e5 Will be calculated as saturation pressure at T_LOW
#    P_HIGH=62e5 Will give dP 
    
    MASS = 1.0
    X0 = 0.0   #Fraction of first fluid in case of pure use 0.0
    
    FLUID=BACKEND+'::'+FLUID1+'['+str(X0)+']&'+FLUID2+'['+str(1 - X0)+']'
    
    columns = ['P_LOW','P_HIGH','dP','T_LOW','T_HIGH','FLUID','MASS','X0','etaC',\
               'eta0','etaR','V_0','T_C','W','W_P','T_D','T_S','DH','Q_H','Q_R',\
               'T_Pinch']
    df = pd.DataFrame(columns=columns)
    

    
    tot_iter = len(TH_List) * len(dP_List) * len(X0_List) * len(TL_List)
    cnt = 0
    for T_LOW in TL_List:
        for T_HIGH in TH_List:       #iterate over temp range 
            for dP in dP_List:   #iterate over pressure range
                for X0 in X0_List:   #iterate over composition range
                    cnt+=1
                    if T_HIGH < T_LOW: continue
                    try:
                        P_LOW,P_HIGH,etaC,eta0,etaR,V_0,T_C,W,W_P,T_D,T_S,DH,Q_H,Q_R,T_Pinch,FLUID = \
                        worthington(dP,T_LOW,T_HIGH,BACKEND,FLUID1,FLUID2,MASS,X0)
                    except:
                        pass
                    if P_LOW == -1: continue
                    df = df.append(pd.Series([P_LOW/1e5,P_HIGH/1e5,dP,T_LOW-273.15,\
                                              T_HIGH-273.15,FLUID,MASS,X0,etaC*100,\
                                              eta0*100,etaR*100,V_0,T_C-273.15,W,W_P,\
                                              T_D-273.15,T_S-273.15,DH,Q_H,Q_R,\
                                              T_Pinch-273.15],index=columns),ignore_index=True)
                    print("\033[H\033[J")
                    print(np.round(cnt/tot_iter*100,2))


prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

X0_List[0], X0_List[-1] = X0_List[-1], X0_List[0]
for TL in TL_List:
    for j in dP_List:
        plotted = False
        fig1, axs = plt.subplots(nrows=2, figsize=(15, 20)) 
        for i, X0 in enumerate(X0_List):

                if X0 == 0:
                    FLUID=BACKEND+'::'+FLUID1
                elif X0 == 1:
                    FLUID=BACKEND+'::'+FLUID2
                else:
                    FLUID=BACKEND+'::'+FLUID1+'['+str(np.round(X0,3))+']&'+FLUID2+\
                    '['+str(np.round(1-X0,3))+']'
                ab = df[(df.X0 == X0) & (df.FLUID == FLUID) & (df.dP == j) & (df.T_LOW == TL-273.15)]
                if ab.size > 0:
                    if not plotted: 
                        axs[0].plot(ab.T_HIGH,ab.etaC,'k--',label='NOVIKOV')
                        plotted = True
                    axs[0].plot(ab.T_HIGH,ab.eta0,c=colors[i%10],label='_nolegend_')
                    axs[0].plot(ab.T_HIGH,ab.etaR,c=colors[i%10],linestyle='--',label=FLUID[9:]+'\nP_LOW='+str(np.round(ab.P_LOW.iloc[0],3))+'[Bar]')
                    axs[1].plot(ab.T_HIGH,(ab.W_P/ab.W)*100,c=colors[i%10])

                    
                   
        EFF_LIM = 20
        axs[0].grid(which='both')
        axs[0].set_title(FLUID1+'+'+FLUID2+' T_low='+str(T_LOW-273.15)+'[°C] dP='+str(j/1e5)+'[Bar]', fontsize=12)
        axs[0].set_xticks(np.arange(T_LOW-273.15,T_HIGH-273.15+5,10))
        axs[0].set_yticks(np.arange(0,EFF_LIM+0.5,1))
        axs[0].set_xlim(T_LOW-273.15,T_HIGH-273.15+5)
        axs[0].set_ylim(0,EFF_LIM+0.5)
        axs[0].set_xlabel('Temperature [°C]')
        axs[0].set_ylabel('Efficiency [%]')
        
        axs[1].grid(which='both')
        axs[1].set_title(FLUID1+'+'+FLUID2+' T_low='+str(T_LOW-273.15)+'[°C] dP='+str(j/1e5)+'[Bar]', fontsize=12)
        axs[1].set_xticks(np.arange(T_LOW-273.15,T_HIGH-273.15+5,10))
        axs[1].set_yticks(np.arange(0,100+0.5,10))
        axs[1].set_xlim(T_LOW-273.15,T_HIGH-273.15+5)
        axs[1].set_ylim(0,100.5)
        
        axs[1].set_xlabel('Temperature [°C]')
        axs[1].set_ylabel('Pumpwork [%]')
        axs[0].legend(loc='best')
        plt.savefig(FLUID1+'+'+FLUID2+' T_low='+str(T_LOW-273.15)+'[°C] dP='+str(j/1e5)+'[Bar].png')
        plt.show()

#df.to_csv(FLUID1+'+'+FLUID2+' T_low='+str(T_LOW-273.15)+'[°C] dP='+str(j/1e5)+'[Bar].csv')

