# import libraries for use
import pandas as pd
import numpy as np
import math
import random 
import time
import matplotlib.pyplot as plt

###############################################################################################################
###############################################################################################################
#Cooling/reheating schedule function
###############################################################################################################
###############################################################################################################

# this function takes the current T value and will convert it into the next T value by either cooling or reheating
# input is the current T value, the alpha/beta value,the schedule type (cooling or reheating) and the iteration number 
# out put is the new T value after cooling or reheating is performed

def delta_temp(T,change_factor,sch_type= 'Cool',schedule_type='Linear',i=0):
    # linear schedule
    if schedule_type == 'Linear':
        if sch_type== 'Cool': # if the schedule type is cooling then apply the below formula
            T=max(0.0001,T-change_factor) #avoid division by 0 errors by creating a max
        else:
            T=T+change_factor # applied if the schedule type is reheating
            
    # Geometric Schedule        
    if schedule_type == 'Geometric':
        # would be the same formula for heating and cooling but the value would need to be adjusted
        T=T*change_factor        
            
    if schedule_type == 'Logarithmic':
        #T needs to be the initial T
        if sch_type== 'Cool':
            T=T/math.log(i) # cooling function for logarithmic function
        else:
            T=T*math.log(i) # reheating function for logarithmic function
    if schedule_type == 'Very slow cooling':
        if sch_type== 'Cool':
            T=T/(1+change_factor) # very slow cooling
        else:
            T=T*(1+change_factor) # very fast heating 
    return(T)       

###############################################################################################################
###############################################################################################################
# generate new solution
###############################################################################################################
###############################################################################################################
# this always ensures that the soln is within the feasible region
# input is the current solution, and 2 random numbers
# functions used the range avaiable on either side of the current solution and multply it by a number 
# between 0 and 1 to get the amount to add to the current solution

def generate_neighbour(x,r1,r2):

    if r1>0.5: # random number 1 is used for deciding whether to add or subtract from the current x value
        x1=x+(100000-x)*r2 # r2 is used to scale the amount to add 
    else:
         x1=x+(-100000-x)*r2 # r2 is used to scale the amount to subtract
    return(x1)

###############################################################################################################
###############################################################################################################
# Delta E calc
###############################################################################################################
###############################################################################################################
#Input is the archive a, the new solution and the current solution
# output is the delta_E value, 
#     the list A_tilda, which is the union of the archive,current solution and the new solution
#     the number of solutions that dominate the new solution
#     the solutions that the new solution dominates in the archive , this is used later for removing them 
def delta_E(a,x_new,x):
    A_tilda,A_tilda_x_num,A_tilda_x_new_num,x_dominates_check=num_dominated(a,x_new,x) # calls the function num_dominated
    Delta_E=(-A_tilda_x_num+A_tilda_x_new_num)/len(A_tilda) # calculation for the delta_E
    return(Delta_E,A_tilda,A_tilda_x_new_num,x_dominates_check)
    
    
###############################################################################################################
###############################################################################################################
# number of dominated solns fn
###############################################################################################################
###############################################################################################################
# this function is for computing the number of solutions that dominate the new soln and the current soln
#inputs are the archive, the new soln, the current soln
def num_dominated(a,x_new,x):
    # calls the dominated function, this returns the list of dominated solutions for the current and new soln
    a_tilda,x_dominated_check,x_new_dominated_check,x_dominates_check=domination_eval(x,x_new,a) 
    # number of solns in the union of archive, current soln and new soln that dominate the current soln
    A_tilda_x_num=len([i for i in x_dominated_check if  i == True])
    # number of solns in the union of archive, current soln and new soln that dominate the new soln
    A_tilda_x_new_num=len([i for i in x_new_dominated_check if  i == True])
    return(a_tilda,A_tilda_x_num,A_tilda_x_new_num,x_dominates_check)

###############################################################################################################
###############################################################################################################
# domination evaluation
###############################################################################################################
###############################################################################################################
# this function determines if a solution is dominated or not
# inputs are the archive, new and current solutions
# outputs the union of the archive, new and current solutions
# a vector of which solutions from  A_tilda dominate the currenr solution
# a vector of which solutions from  A_tilda dominate the new solution
# a vector of true/ false that tells if the new solution is better than each of the solutions in A_tilda

def domination_eval(x,x_new,a):
    a_tilda= a.copy() # makes a deep copy of the archive
    a_tilda.append(x_new) # union the new solution to archive
    a_tilda.append(x)    # union the current solution to the archive
    z_x=obj_fns(x)      # evaluate the objective function for the current solution
    z_x_new=obj_fns(x_new)  # evaluate the objective function for the new solution

    # create empty lists for storage
    x_dominated_check=[]  
    x_new_dominated_check=[]
    x_dominates_check=[]
    
    for i in a_tilda:
        # appends whether or not a soln in A-tilda dominates the current soln
        x_dominated_check.append(obj_fns(i)[0]<z_x[0] and obj_fns(i)[1]<z_x[1])   
        # appends whether or not a soln in A-tilda dominates the new soln
        x_new_dominated_check.append(obj_fns(i)[0]<z_x_new[0] and obj_fns(i)[1]<z_x_new[1])
        # appends whether or not a soln in A-tilda is dominated by the new soln
        x_dominates_check.append(obj_fns(i)[0]>z_x_new[0] and obj_fns(i)[1]>z_x_new[1])

    return(a_tilda,x_dominated_check,x_new_dominated_check,x_dominates_check)
        
    
###############################################################################################################
###############################################################################################################
# evaluate the obj fn
###############################################################################################################
###############################################################################################################
# evaluates the objective functions for a given solution
# input a solution
# output a fitness for the input solution comprising of the 2 objective function values in a list
def obj_fns(x):
    z1=x**2
    z2=(x-2)**2
    return(z1,z2)

###############################################################################################################
###############################################################################################################
# Function for diversity density calculation
###############################################################################################################
###############################################################################################################
# diversity density calculation, depnding on the method chosen
# inputs the new solution, the archive, which method to use and a threshold for the kernel function
# outputs a density value
def diversity_check(x_new,a,method,sigma=0.001):
    if method == 'Kernel':
        #kernel function
        
        dist=[] # create an empty list for storage
        for i in a: # loop through the archive
            # apply sharing function
            # if distance between the new soln and the element in the archive is greater than the threshold 
            if x_new-i < sigma: 
                dist.append(1-((x_new-i)/sigma)) 
            else:
                dist.append(0)
        density_value=sum(dist)         #sums the sharing function for all elements in the archive
        
        
        
    elif method== 'NN':
        #nearest neighbour function - Crowding dist
        density=[] # create an empty list for storage
        
        # for each element in the archive against each element in the archive
        for j in a+[x_new]:
            f1=[] # create an empty list for storage
            f2=[] # create an empty list for storage
            dist=[] # create an empty list for storage
            for i in a+[x_new]:
                if i!=j: # ensure that i is never equal to j for the calculation
                    # calculates the distance between the 2 elements in the objective space in terms of f1
                    f1.append(obj_fns(i)[0]-obj_fns(j)[0]) 
                    # calculates the distance between the 2 elements in the objective space in terms of f2
                    f2.append(obj_fns(i)[1]-obj_fns(j)[1])
                    # calculates the distance between the 2 elements in the objective space in terms of f1 and f2
                    dist.append(((obj_fns(i)[1]-obj_fns(j)[1])**2+(obj_fns(i)[0]-obj_fns(j)[0])**2)**0.5) # euclidean dist
            # converts to a dataframe for easy manipulation
            soln=pd.DataFrame(list(zip(dist,f1,f2)),columns=['dist','f1','f2']) 
            
            # finding the closest point to the left of the solution ( non dominated)
            bound1=soln[(soln['dist']==soln[(soln['f1']<0) & (soln['f2']>0)]['dist'].min())&(soln['f1']<0) & (soln['f2']>0)]
            
            # finding the closest point to the right of the solution ( non dominated)
            bound2=soln[(soln['dist']==soln[(soln['f1']>0) & (soln['f2']<0)]['dist'].min())&(soln['f1']>0) & (soln['f2']<0)]
            
            # if the current point has no neighbours( edge points) then it needs a very big value as a density score
            if len(bound1)>0 and len(bound2)>0:
                # calculates the circumference of the bounding box between neighbours and current 
                density.append(list(2*abs(bound1['f1'].values-bound2['f1'].values)+2*abs(bound1['f2'].values-bound2['f2'].values))[0])
            else:
                density.append(100000000000000)
        #ranking the solutions by their density score, the higher the value the better the rank
        density_value=pd.DataFrame(list(zip(density)),columns=['density']).rank(method='min',ascending=False)['density'][len(density)-1]/ (len(density)-1)  
    
    
    else: 
        #Histogram method
        # simplfied version used for this
        # predefined histograms of 0.1 by 0.1 between 0 and 2
        f1_grid=[i/10 for i in range(0,2001,1)] # initial grid 
        f2_grid=[i/10 for i in range(0,2001,1)] # initial grid
        
        # finding which block in the grid the new solution belongs to
        x_new_f1_upperbound=min([i if i>= obj_fns(x_new)[0] else math.ceil(obj_fns(x_new)[0]) for i in f1_grid ])  # f1
        x_new_f2_upperbound=min([i if i>= obj_fns(x_new)[1] else math.ceil(obj_fns(x_new)[0]) for i in f2_grid  ]) #f2
        f1_upperbound=[] # empty list for storage
        f2_upperbound=[] # empty list for storage
        for j in a: # loop through the elements in the archive
            # extends the grid so that a solution outside the 0-2 range will still belong to a grid 
            # this saves time searching for all others 
            #- could even just use the rounding function to the nearest grid bound
            f1_grid=[i/10 for i in range(0,math.ceil(obj_fns(j)[0])*11,1)] 
            f2_grid=[i/10 for i in range(0,math.ceil(obj_fns(j)[1])*11,1)]
            f1_upperbound.append(min([i for i in f1_grid if i>= obj_fns(j)[0]])) # stores the f1 upper bound grid value
            f2_upperbound.append(min([i for i in f2_grid if i>= obj_fns(j)[1]])) # stores the f1 upper bound grid value
        
        # convert to data frame for easy manipulations
        df=pd.DataFrame(list(zip(f1_upperbound+[x_new_f1_upperbound],f2_upperbound+[x_new_f2_upperbound])),columns=['f1','f2'])
        # counts the number of entries in each block on the grid
        density=df.groupby(['f1','f2']).size().reset_index(name="Count")
        density=pd.DataFrame(density) # converts to datframe again so that the results can be joined 
        #join back to the original dataframe and 
        #get the last value from there which is the density value of the new solution
        density_value=df.join(density.set_index(['f1','f2']), on=['f1','f2'])["Count"][len(df)-1]
    return(density_value)
        

###############################################################################################################
###############################################################################################################
# Function to wrap the DBMOSA 
###############################################################################################################
###############################################################################################################
# function for the DBMOSA implementation
# Inputs
# 1)    x - initial soln
# 2)    i_max - max number of epochs
# 3)    c_max - max number of solns accepted
# 4)    d_max - max number of soln rejects
# 5)    T - Initial temperature
# 6)    Beta - cooling rate
# 7)    Alpha - reheating rate
# 8)    termination_criteria - based on final temperature vs max epochs
# 9)    epoch_length - 'Static' or 'Dynamic' - dynamic depends on the number of solns accepted and rejected
#                                            - static depends on the number of epochs that ave passed
# 10)   cool_reheat - schedule is linear, gemetric, very slow or logarithmic
# 11)   diversity_method - 'Kernel','Histogram' or 'NN'
# 12)   diversity_preserve - True or false - switches on the diversity function 
# 13)   num_elements_in_A_before_diversity - at what point does the diversity function start working 
#                                          - this is for perfomance
# 14)   threshold_histo - value at which the the density is considered too high
#                         ,solutions with a value higher than this will be rejected
# 15)   static_T - number of epoch to change the temperature after

# Outputs 
# 1) the final best solution



def dbmosa(x,i_max,c_max,d_max,T,Beta,Alpha,termination_criteria
           ,epoch_length,cool_reheat,diversity_method
           ,diversity_preserve,num_elements_in_A_before_diversity
           ,threshold_histo,static_T):
    #Initialise starting variables
    i=1
    c=0
    d=0
    t=1
    a=[]
    a.append(x)#initial soln
    end =0
    T_epoch=0
    Worse_delta_E_accepted=[]
    
    # create a continuous loop which will only end when end =1
    while end!=1:

        # termination criteria 
        # if the termination criteria is epoch based and the number of epochs = the max number of epochs, end
        if i == i_max and termination_criteria== 'epoch':
            end=1
            print(1)
        # if the termination criteria is temperature based and T= the stopping temperature, end
        elif stopping_temp>=T and termination_criteria=='temperature':
            end=1
        else:
            
            
            if d==d_max and epoch_length== 'Dynamic' : #Implement a dynamic epoch length
                #increase T
                T=delta_temp(T,Alpha,sch_type= 'Heat',schedule_type=cool_reheat,i=0)
                i+=1
                c=0
                d=0
                
            elif c==c_max and epoch_length== 'Dynamic': #Implement a dynamic epoch length

                #decrease temp
                T=delta_temp(T,Beta,sch_type= 'Cool',schedule_type=cool_reheat,i=0)
                i+=1
                c=0
                d=0

            elif T_epoch>=static_T and epoch_length == 'Static': #Implement a static length epoch
                T=delta_temp(T,Beta,sch_type= 'Cool',schedule_type=cool_reheat,i=0)
                T_epoch=0
                i+=1

            else:
                x_new = generate_neighbour(x,random.random(),random.random()) #generates a new neighbour
                Delta_E,A_tilda,A_tilda_x_new_num,x_dominates_check=delta_E(a,x_new,x) # calculates the delta_E
                
                # Introduce the diversity based criterion

                if diversity_method=='Kernel' and diversity_preserve==True and len(a)>num_elements_in_A_before_diversity:
                    #kernel acceptance
                    density_value=diversity_check(x_new,a,method=diversity_method)
                    if density_value>0: # prevent divide by 0 errors
                        Delta_E=Delta_E/density_value # calculating the new delta_E by dividing by the density value

                elif diversity_method=='NN' and diversity_preserve==True and len(a)>num_elements_in_A_before_diversity:
                    #NN acceptance
                    density_value=diversity_check(x_new,a,method=diversity_method,sigma=0.01)
                    # function output the reletive rank, rank/ number of elements in the archive
                    if density_value>=1: # if the reletive rank = 1 then it is the worst solution, in terms of diversity
                        Delta_E=1000000000*T #large value ensure very small chance of accepting a solution


                elif diversity_method=='Histogram' and diversity_preserve==True and len(a)>num_elements_in_A_before_diversity:
                    #histo acceptance
                    density_value=diversity_check(x_new,a,method=diversity_method)
                    if density_value>threshold_histo:
                        Delta_E=1000000000*T #large value ensure very small chance of accepting a solution


                if random.random()> min(1,math.exp(-(Delta_E/T))):
                    #reject x_new
                    t+=1
                    d+=1
#                     print(T)

                else:

                    #accept and update
                    x=x_new
                    c+=1

                    Worse_delta_E_accepted.append(Delta_E) # used for getting the acceptance deviation value of T

                    if A_tilda_x_new_num==0: # if x_new is not dominated 
                        # remove solutions taht are worse than x_new form the archive
                        remove_from_a=[A_tilda[i] for i in range(0,len(x_dominates_check)) if  x_dominates_check[i] == True] # create a list of dominated in A_tilda
                        a=list(set(a.copy())-set(remove_from_a.copy())) # removes the dominated solutions in A
                        a.append(x) #adds x to set A 
                        print(a)
                        print(T)
                    t+=1
                T_epoch+=1
                
    return(a)





#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
# Use of DBMOSA begins here
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################

# # Declare global variables
i_max=20000#200
c_max=200
d_max=150
x=random.randrange(-100000,100000)

# Starting temperature
T=10000000000 # accept all - set T0 extremely high
#T=1.7038 # Acceptance deviation

# Epoch length
# epoch_length = 'Static'
# static_T = 100
epoch_length= 'Dynamic'


#Cooling Reheating schedule
# cool_reheat='Linear'
# Beta=0.001
cool_reheat='Geometric'
Beta=0.9999
Alpha=0.5

# Search termination Criteria
termination_criteria= 'temperature'
stopping_temp=0.0001
# termination_criteria= 'epoch'

## Diversity-based criterion
# diversity_method='Kernel'
# diversity_method='NN'
diversity_method='Histogram'
threshold_histo=5
num_elements_in_A_before_diversity=5
diversity_preserve=False
# diversity_preserve=True

a=dbmosa(x,i_max,c_max,d_max,T,Beta,Alpha,termination_criteria
           ,epoch_length,cool_reheat,diversity_method
           ,diversity_preserve,num_elements_in_A_before_diversity
           ,threshold_histo,static_T)           
  
