# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from math import exp, log, floor
import matplotlib.pyplot as plt
import QuantLib as ql
from scipy.optimize import fsolve
import random


fix_T_arr = np.arange(0, 30.5, 0.5)
flt_T_arr = np.arange(0, 30.25, 0.25)

fix_table = pd.DataFrame( fix_T_arr,columns = ['fix_T']) #index 是期数， columns是年份
flt_table = pd.DataFrame(flt_T_arr,columns = ['flt_T'])
fix_table.set_index( ['fix_T'], inplace = True)
flt_table.set_index( ['flt_T'], inplace = True)
#flt_table['R_all'] = 0
# print(flt_table)

fix_delta_T = 0.5
flt_delta_T = 0.25
R_all = {} #在flt table 中增加一列 命名为R_all，存储各期swap rate数值
#R_all[2], R_all[4], R_all[5], R_all[8], R_all[10], R_all[20], R_all[30] = R[0], R[1], R[2], R[3], R[4], R[5], R[6]

given_data = pd.read_csv('swap_rate_data.csv', index_col=0)
for i in given_data.index:
    R_all[i] = given_data.loc[i, 'swap_rate'] / 100
    
discount_factor = {0:1, 0.5:1/(1 + 0.5*4.903/100)}
# print(given_data)

y_vec = {} # generate the zero rate of flt_T_arr（smallest delta t）
y_vec[0] = (1+4.292/100/365)**365-1 # 隔夜拆借利率的有效利率
y_vec[0.5] = -log(1/(1+0.5*4.903/100))/0.5

# part 1 (a) 
def bootstrapping(): #计算所有quarterly的zero rate, DF和所有semiannual的swap rate
    
    #R = given_data['swap_rate']
    
    special_dates = [0.5, 2, 4, 5, 8, 10, 20, 30]
    
    for i in fix_table.index:
        for j in range(len(special_dates)-1):
            if special_dates[j] < i <= special_dates[j+1]:
                date = special_dates[j]
                next_date = special_dates[j+1]
                x = fsolve(compute_NPV, x0=0, args=(date, next_date))
                
                y_vec[next_date] = float(x)
                y_vec[i] = y_vec[date] + (i-date)/(next_date-date) * (y_vec[next_date] - y_vec[date])
                discount_factor[i] = exp(-y_vec[i]*i)
                total_sum = 0
                for k in range(1,int(i*2)+1):
                    total_sum += discount_factor[ k*0.5 ]
                    
                R_all[i] = (discount_factor[0] - discount_factor[i])/(fix_delta_T * total_sum)
                #R_all 是 swap rate 只有i是0.5的倍数时才有
                break
            
    for term in flt_table.index:
        if term not in fix_table.index:
            y_vec[term] = 0.5 * (y_vec[term-0.25] + y_vec[term + 0.25])
            discount_factor[term] = exp(-y_vec[term] * term)
            
    for i in y_vec:
        if i in flt_table.index:
            flt_table.loc[i, 'Rate'] = y_vec[i] * 100
            flt_table.loc[i, 'Discount'] = discount_factor[i]


def compute_NPV( x, date, next_date):
    total_sum = 0
    for k in range(1, int(next_date*2)+1):
        i = k*0.5
        y_vec_i = y_vec[date] + (i-date)/(next_date-date) * (x - y_vec[date])
        total_sum += exp(-y_vec_i*i)
        
    NPV = fix_delta_T * R_all[next_date] * total_sum \
        - (discount_factor[0] - exp(-y_vec_i*i))
    return NPV            

 
def get_plot():
    
    #绘图：双轴折线图
    fig = plt.figure(figsize=(15, 9))
    ax1 = fig.add_subplot(111)
    ax1.plot(flt_table['Rate'], lw=1.5, label='zero_rate')
    ax1.plot(flt_table['Rate'], 'ro')
    
    # 左侧坐标
    ax1.grid()
    ax1.axis('tight')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('zero rate')
    
    ax1.legend(loc=2)
    
    # 右侧坐标
    ax2 = ax1.twinx()
    ax2.plot(flt_table['Discount'], 'g', lw=1.5, label='discount factor')
    ax2.set_ylabel('discount factor')
    ax2.legend(loc=1)
    plt.title('zero rate curve and discount factor curve')
    plt.savefig('zero rate & discount factor curve.png')
    plt.show()

# part 1 (b)
# 计算远期利率 forward term rate
def get_forward_rate(delta_T):
    
    for i in flt_table.index:
        if i == 0:
            flt_table.loc[i, 'F_'+str(delta_T)] = 4.292/100
        elif i > delta_T:
            flt_table.loc[i, 'F_'+str(delta_T)] = \
                (1/delta_T) *(discount_factor[i-delta_T]/discount_factor[i] -1) # F[i] 表示LIBOR@i =f0.25(0, i-0.25, i)

def construct_forward_rate_curve(flt_delta_T_list):
    # 绘图：forward rate 折线图
    plt.figure(figsize=(10,4))
    for t in flt_delta_T_list:
        get_forward_rate(t)
        plt.plot( flt_table['F_' +str(t)], lw=1.5, label = 'delta T=' + str(t))
    
    plt.grid( True )
    plt.legend(loc=2, bbox_to_anchor=(1.05,1.05),borderaxespad = 0.)
    plt.axis( 'tight' )
    plt.xlabel('Year')
    plt.ylabel('forward rate')
    plt.title(' different forward rate curves ')
    plt.savefig('forward_rate.png')
    plt.show()

if __name__ == '__main__':
    bootstrapping()
    get_plot()
    construct_forward_rate_curve(list(np.arange(0.25, 3.25, 0.25))+list(np.arange(4, 9)))


## part 2: Hull White Model
length = 30
timesteps = 60 # 30*delta_t
delta_t = 0.5
k = 0.19
sigma = 0.0033
J = floor( 1/(2*k* delta_t) ) + 1
dr = sigma * np.sqrt(3*delta_t)
# steps = [j for j in range( 1, len(fix_T_arr) )]

# part 2 (a)
# 计算eta及各点的上涨概率,下跌概率,不变概率 
def get_probality():
    eta, up, dp, mp = {}, {}, {}, {}  

    for j in range(len(fix_T_arr)-1): # j∈[0,59],都是站在当前时点计算未来数值，所以j最大是59
        for i in range( -min(j,J), min(j,J)+1 ): # i∈[-6,6]
            if j>=6 and i == -6:
                eta[(i,j)] = (( 1-k*delta_t )*i - (i+1) )*dr
                up[(i,j)] = 0.5*( (sigma**2 * delta_t + eta[(i,j)]**2)/ dr**2 + eta[(i,j)]/dr ) 
                dp[(i,j)] = 0.5*( (sigma**2 * delta_t + eta[(i,j)]**2)/ dr**2 - eta[(i,j)]/dr )
                mp[(i,j)] = 1- (sigma**2 * delta_t + eta[(i,j)]**2)/ dr**2
            elif j>=6 and i == 6:
                eta[(i,j)] = (( 1-k*delta_t )*i - (i-1) )*dr
                up[(i,j)] = 0.5*( (sigma**2 * delta_t + eta[(i,j)]**2)/ dr**2 + eta[(i,j)]/dr ) 
                dp[(i,j)] = 0.5*( (sigma**2 * delta_t + eta[(i,j)]**2)/ dr**2 - eta[(i,j)]/dr )
                mp[(i,j)] = 1- (sigma**2 * delta_t + eta[(i,j)]**2)/ dr**2
            else:
                eta[(i,j)] = (( 1-k*delta_t )*i - i )*dr
                up[(i,j)] = 0.5*( (sigma**2 * delta_t + eta[(i,j)]**2)/ dr**2 + eta[(i,j)]/dr ) 
                dp[(i,j)] = 0.5*( (sigma**2 * delta_t + eta[(i,j)]**2)/ dr**2 - eta[(i,j)]/dr )
                mp[(i,j)] = 1- (sigma**2 * delta_t + eta[(i,j)]**2)/ dr**2
    return eta, up,mp,dp

eta, up, mp, dp = get_probality()

Q = {} 
Q[(0,0)] = 1

r = {}
r[(0,0)] = -np.log(discount_factor[0.5]) / delta_t
theta = {}
# 计算sum_Q
def get_sum_Q(): 
    # Q[(1,1)] = up[(0,0)]*exp(-delta_t * r[(0,0)])*Q[(0,0)]
    # Q[(0,1)] = mp[(0,0)]*exp(-delta_t * r[(0,0)])*Q[(0,0)]
    # Q[(-1,1)] = dp[(0,0)]*exp(-delta_t * r[(0,0)])*Q[(0,0)]
    
    for j in range( 1, len(fix_T_arr)-1): #j∈[1,59]
        sum_Q = {}
        
        for i in range(-min(j,J), min(j,J)):
            #for m in range(-min(j-1,J), min(j-1,J)):
            if i==j:
                sum_Q[(i,j)] = Q[(i-1,j-1)]*up[(i-1,j-1)]*exp(-r[(i-1,j-1)]*delta_t)
            elif i==-j:
                sum_Q[(i,j)] = Q[(i+1,j-1)]*dp[(i+1,j-1)]*exp(-r[(i+1,j-1)]*delta_t)
            elif i== j-1:
                 sum_Q[(i,j)] = Q[(i,j-1)]*mp[(i,j-1)]*exp(-r[(i,j-1)]*delta_t) + Q[(i-1,j-1)]*up[(i-1,j-1)]*exp(-r[(i-1,j-1)]*delta_t)         
            elif i==-j+1:
                sum_Q[(i,j)] = Q[(i,j-1)]*mp[(i,j-1)]*exp(-r[(i,j-1)]*delta_t) + Q[(i+1,j-1)]*dp[(i+1,j-1)]*exp(-r[(i+1,j-1)]*delta_t)         
            else:
                sum_Q[(i,j)] = Q[(i-1,j-1)]*up[(i-1,j-1)]*exp(-r[(i-1,j-1)]*delta_t) + Q[(i,j-1)]*mp[(i,j-1)]*exp(-r[(i,j-1)]*delta_t) + Q[(i+1,j-1)]*dp[(i+1,j-1)]*exp(-r[(i+1,j-1)]*delta_t)  
        return sum_Q
sum_Q = get_sum_Q()        
            
# 计算三叉树每个点的short rate
def get_hw_short_rate(): 
    for j in range( 1, len(fix_T_arr)-1): #j∈[1,59]
        sum_Qexp = {}
        for i in range( -min(j,J), min(j,J)+1 ):
            sum_Qexp[min(j,J)] = np.sum(Q[(i,j)]*exp(-r[(i,j)]*delta_t))
                                        
        for i in range( -min(j,J), min(j,J)+1 ):
            #  Q[(i,j)
            
            if j >=2:
                if np.abs( i+1 ) > j-1:
                    a = 0
                else:
                    a = dp[(i+1,j-1)] * exp(-delta_t * r[(i+1,j-1)]) * Q[(i+1,j-1)]
                    
                if np.abs( i ) > j-1:
                    b = 0
                else:
                    b = mp[(i,j-1)] * exp(-delta_t * r[(i,j-1)]) * Q[(i,j-1)]
                
                if np.abs( i-1 ) > j-1:
                    c = 0
                else:
                    c = up[(i-1,j-1)] * exp(-delta_t * r[(i-1,j-1)]) * Q[(i-1,j-1)]
                    
                    Q[(i,j)] = a + b + c
        
            # r[(i,j)]
            #存储各个节点的short rate
            r[(i,j)] = (1-k*delta_t) * r[(0,j-1)] + i*dr
                
            # theta[j-1]
            
            theta[j-1] = 1/(k*(delta_t)**2) * np.log( sum_Qexp[j] / discount_factor[(j+1)*delta_t])
    
            #update r[(i,j)] 
            r[(i,j)] = r[(i,j)] + k * delta_t * theta[j-1]

    return  Q
Q = get_hw_short_rate()       


def get_hw_short_rate(): 
    Q = {} 
    Q[(0,0)] = 1

    r = {}
    r[(0,0)] = -np.log(discount_factor[0.5]) / delta_t
    theta = {}
    
    Q[(1,1)] = up[(0,0)]*exp(-delta_t * r[(0,0)])
    Q[(0,1)] = mp[(0,0)]*exp(-delta_t * r[(0,0)])
    Q[(-1,1)] = dp[(0,0)]*exp(-delta_t * r[(0,0)])
    
    for j in range( 1, len(fix_T_arr)-1): #j∈[1,59]
        sum_df_Q = {}
        for i in range( -min(j,J), min(j,J)+1 ):
            sum_df_Q[min(j,J)] = np.sum(Q[(i,j)]*exp(-r[(i,j)]*delta_t))
                                        
        for i in range( -min(j,J), min(j,J)+1 ):
            #  Q[(i,j)
            
            if j >=2:
                if np.abs( i+1 ) > j-1:
                    a = 0
                else:
                    a = dp[(i+1,j-1)] * exp(-delta_t * r[(i+1,j-1)]) * Q[(i+1,j-1)]
                    
                if np.abs( i ) > j-1:
                    b = 0
                else:
                    b = mp[(i,j-1)] * exp(-delta_t * r[(i,j-1)]) * Q[(i,j-1)]
                
                if np.abs( i-1 ) > j-1:
                    c = 0
                else:
                    c = up[(i-1,j-1)] * exp(-delta_t * r[(i-1,j-1)]) * Q[(i-1,j-1)]
                    
                    Q[(i,j)] = a + b + c
        
            # r[(i,j)]
            #存储各个节点的short rate
            r[(i,j)] = (1-k*delta_t) * r[(0,j-1)] + i*dr
                
            # theta[j-1]
            
            theta[j-1] = 1/(k*(delta_t)**2) * np.log( sum_df_Q[j] / discount_factor[(j+1)*delta_t])
    
            #update r[(i,j)] 
            r[(i,j)] = r[(i,j)] + k * delta_t * theta[j-1]

    return  Q
Q = get_hw_short_rate()       



S = 20
c1 = 0.04
c2 = 0.05
B = {}
T0 = random.randint(1,40)  # 设T0是[1,40]的随机起始点
print(T0)


def get_coupon_bond_price():      
    for Tj in range(60,T0,-1):  # 不必理会S 所以Tj可以取T0+1~60 period end 
        for i in range(-min(Tj,6), min(Tj,6)+1):
            B[(i,60)] = 1 
            
            if Tj >= 22:
                AA = up[(i,Tj)]*(B[(i+1,Tj+1)] + c2*delta_t)   # 上涨的分支
                BB = mp[(i,Tj)]*(B[(i,Tj+1)] + c2*delta_t)  # 中间的分支
                CC = dp[(i,Tj)]*(B[(i-1,Tj+1)] + c2*delta_t) # 向下的分支
                if abs(i+1)>J:
                    AA = 0
                if abs(i-1)>J:
                    CC = 0
                B[(i,Tj)] = (AA+BB+CC)*exp(-r[(i,Tj)]*delta_t)
            
            if Tj < 22:
                AA = up[(i,Tj)]*(B[(i+1,Tj+1)] + c1*delta_t)   #上涨的分支
                BB = mp[(i,Tj)]*(B[(i,Tj+1)] + c1*delta_t)  #中间的分支
                CC = dp[(i,Tj)]*(B[(i-1,Tj+1)] + c1*delta_t) #向下的分支
                if abs(i+1)>J:
                    AA = 0
                if abs(i-1)>J:
                    CC = 0
                B[(i,Tj)] = (AA+BB+CC)*exp(-r[(i,Tj)]*delta_t)
    return B
B = get_coupon_bond_price()  



K = 0.8    
mean_rj ={}                    
def get_coupon_bond_price():    
    for j in range(1,T0): 
        for i in range(-min(j,J), min(j,J)+1):
            payoff_at_T0 = max(B[(i,T0)]-K, 0)
           
            mean_rj_all = r[(0,0)]
            for s in range(min(j,J)):
                mean_rj[j] = np.mean( up[(s,j-1)]*r[(s+1,j)] + mp[(0,j-1)]*r[(0,j)] + dp[(-s,j-1)]*r[(-s-1,j)] )
                mean_rj_all += mean_rj[j]
    y_0T0 = 1/T0*mean_rj_all
    coupon_bond_price_at_0 = payoff_at_T0*exp(-y_0T0*T0)
    return coupon_bond_price_at_0             
coupon_bond_price_at_0 = get_coupon_bond_price()                  
    