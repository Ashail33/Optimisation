#import libraries
import pandas as pd
import numpy as np
import random

################################################################################################################
################################################################################################################

################################################################################################################
################################################################################################################
#function to create a random starting population
#inputs are the number of cities and the number of solutions to create in the population
#outputs the random population
################################################################################################################
def random_population(num_pop,num_cities):
    population=[] #intialise the variable
    for i in range(0,num_pop): #loop for creating num_pop number of solutions
        population.append(random.sample(range(0,num_cities),num_cities))# appends the solution to the population
    return(population)
################################################################################################################

################################################################################################################
################################################################################################################
#function to evaluate the fitness of a solution 
#Inputs are a solution and the distiance matrix
#outputs the fitness of th esolution
################################################################################################################
def eval_fitness(solution, dist_matrix ):
    z=0 #initialise the fitness to be 0
    for i in range(1,len(solution)): #loop through the solution to get the edges between the nodes in solution
        z+=dist_matrix[solution[i]][solution[i-1]] # adds the distance of the edge to the fitness
    return(z)

#function to evaluate the fitness of the population
#input is the population
#output is the fitness of the population

def eval_pop_fitness(pop):
    fitness=[] #initialise the fitness to an empty list
    for i in pop: # loop through each solution in the population
        fitness.append(eval_fitness(i,dist_matrix)) # appends the fitness of each solution to the list
    return(fitness)
################################################################################################################

################################################################################################################
################################################################################################################
#function for selecting the parents using tournament
#input is the population,their fitness and the number of parents that need to be chosen
################################################################################################################
def parent_selection(pop,fitness,num_parents=6):
    weights=[] # initialise the weights
    Parent_list=[] #initialise the parent list
    fit_total=0 # initialise the fitness total to 0
    for i in fitness: 
        fit_total+= 1/i # this is used as there is minimisation problem, fitness is 1/distance
    for i in fitness:
        weights.append(((1/i)/fit_total)) # used to get the weighted fitness for each solution

        
    # calls a libarary for choosing 1 solution at a time with a weighted probablity 
    #(emulating the tournament) without replacement
    # this is done 6 times to get the required parents
    
    choices=np.random.choice(range(0,len(pop)),num_parents,replace=False, p=weights) 
    
    for i in choices:
        Parent_list.append(pop[i]) #append the chosen parents to the list
    return(Parent_list)   
################################################################################################################

################################################################################################################
################################################################################################################
# function for crossing over 
# input 2 parent solutions
# output the children in a list
################################################################################################################
def crossover(parent1,parent2):
    child_1=parent2.copy() #creates a child that is a copy of the parent 2
    child_2=parent1.copy() #creates a child that is a copy of the parent 1
    cross_over_points= random.sample(range(0,len(parent1)-2),2) # randomly selecting 2 crossover points
    cross_over_points.sort() #sorts the crossover into numrical order ascending

    cross_over_point1= cross_over_points[0] # first point for crossover
    cross_over_point2= cross_over_points[1] # second point for crossover
    
    # implementing the 2 point crossover logic 
    #the points in the genome that fall outside the crossover region will be excluded from the parent 
    #2 pieces that will replace the points between the 2 cross over points
    # this is done to ensure there is no repair needed
    child_2[cross_over_point1+1:cross_over_point2+2]=list(set(parent2)
                                -set(set(parent1)- set(parent1[cross_over_point1+1:cross_over_point2+2])))
    child_1[cross_over_point1+1:cross_over_point2+2]=list(set(parent1)
                                -set(set(parent2)- set(parent2[cross_over_point1+1:cross_over_point2+2])))
    
    return([child_1,child_2])
################################################################################################################

################################################################################################################
################################################################################################################
# function used for the reproduction logic
# input is the parent list
# output is the children list
# the function takes the 1st 2 parents as a couple as they were selected together 
# and calls the crossover over function, the parents will be deleted from the list 
# and the next couple will be used, until the list is empty
################################################################################################################
def reproduction(Parent_list):
    children_list=[] # initialise the children list to empth list
    parent_temp_list=Parent_list.copy() # make a deep copy of the parents list
    while len(parent_temp_list)>0: # check to see if the list is not empty
        #Create the children and append to the children list
        # using the first 2 parents of the list
        children_list.append(crossover(parent_temp_list[0],parent_temp_list[1])[0]) 
        children_list.append(crossover(parent_temp_list[0],parent_temp_list[1])[1])
        
        # remove the first 2 parents from the list
        parent_temp_list.pop(0)
        parent_temp_list.pop(0)
    return(children_list)
################################################################################################################

################################################################################################################
################################################################################################################
# function for the mutation of the children
# input children list
# output children list with 1 randomly chosen child that has mutated

def mutation(children_list):
    chosen_for_mutation_index=random.sample(range(0,len(children_list)),1)[0] # selection of a child for mutation
    mutating_child=children_list[chosen_for_mutation_index].copy()  # create a deep copy of the selected childs' solution
    mutating_bits=random.sample(range(0,len(mutating_child)),2)     # Randomly select the 2 bits that will be swapped 
    
    # perform the bit swapping mutation
    children_list[chosen_for_mutation_index][mutating_bits[0]] = mutating_child[mutating_bits[1]] 
    children_list[chosen_for_mutation_index][mutating_bits[1]]= mutating_child[mutating_bits[0]]
    return(children_list,mutating_child)
################################################################################################################    

################################################################################################################
################################################################################################################
# function for removing duplicated solutions
# input list of solutions
# out put list of distinct solutions
################################################################################################################
def get_unique_solutions(seq): 
   # order preserving
    checked = [] # initialise the list of unique solutions
    for e in seq:
        # logic for checking if a solution is in the unique list of solutions
        if e not in checked:
            checked.append(e)
    return checked
################################################################################################################

################################################################################################################
################################################################################################################
# function for the replacement strategy
# input children and parent lists and the number of best solutions. 
# n is optional and when not given is assumed to be 8
# outputs list of top n solutions, S* and Z* for the current top n solutions
################################################################################################################
def replacement_strategy(children_list,Parent_list, n=8):
    # merge the 2 lists
    all=children_list+Parent_list
    #Remove the duplicated solutions between the children and parent population
    all=get_unique_solutions(all)
    #Create a table structure for easy manipulation
    temp_table=pd.DataFrame(list(zip(all,eval_pop_fitness(all))),
                                columns=['Genome','Z'])
    # drop all the other solutions that are not the min top n solutions
    temp_table=temp_table.nsmallest(n, 'Z')
    return(temp_table['Genome'],temp_table[temp_table['Z']==temp_table['Z'].min()]['Genome'].values[0]
           ,temp_table['Z'].min())
 ################################################################################################################   

################################################################################################################
################################################################################################################
# use of the genetic algorithm example
# global variable declaration
################################################################################################################
iterations_without_improvement=0
iterations=0
history_store=[]
#Setting the distance matrix to a variable
dist_matrix = [[0,41,26,31,27,35]
               ,[41,0,29,32,40,33]
               ,[26,29,0,25,34,42]
               ,[31,32,25,0,28,34]
               ,[27,40,34,28,0,36]
               ,[35,33,42,34,36,0]]
history_store.append([["generation number"],["entire population"],["population fitness"],["selected parents"]
                      ,["child selected for mutation"],["mean pop fitness"]])

# initialise the population, incumbent solution (s_star) and its' value (z_star)
pop=random_population(8,len(dist_matrix)) 
pop,s_star,z_star=replacement_strategy(pop,[])
pop=list(pop)

while iterations_without_improvement<=10:
    generation_info=[] #initialise local variable for genrational info
    
    fitness=eval_pop_fitness(pop) # call function for fitness evaluation of population
    Parent_list=parent_selection(pop,fitness,6) # select the parents using the fitness 
    children_list=reproduction(Parent_list) # let the parent's reproduce to create the children
    children_list,mutated_child=mutation(children_list) # mutate 1 child
    mean_pop_fitness = sum(fitness)/len(fitness)
    
    # generational info collection
    # This is done to answer the question and not part of the actual model
    generation_info.append(iterations)
    generation_info.append(pop)
    generation_info.append(fitness)
    generation_info.append(Parent_list)
    generation_info.append(mutated_child)
    generation_info.append(mean_pop_fitness)
    
    # Store the genrational info in the history table. 
    # This is done to answer the question and not part of the actual model
    history_store.append(generation_info)

    #use replacement strategy to update the population
    pop,best_genome,best_fitness=replacement_strategy(children_list,Parent_list) 
    pop = list(pop)
    
    # if the best_fitness in the new population better than the incumbent then update 
    # and set num iterations without replacement to 0
    # else count the number of iterations without improvement
    if best_fitness<z_star:
        z_star=best_fitness
        s_star=best_genome
        iterations_without_improvement=0
    else :
        iterations_without_improvement+=1
    #increment the iteration - aka generation number
    iterations+=1
table=pd.DataFrame(history_store)
################################################################################################################
