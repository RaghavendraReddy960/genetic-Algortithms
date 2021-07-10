import numpy as np #pylint: disable=E0401
import client as server
from tabulate import tabulate #pylint: disable=E0401
import random


MAX_DEG = 11 
SECRETKEY = 'VdvaARr1smFWIjhN8PAUCJ87dYJfDW01BOrrRigkyRYel25elX'

#initial parameters changed according to get best fit vector
pop_size = 20
surity = 2
cross_select = 5
crossovernumber = 5
gen_count = 1



#mutation function
def mutation_operator(temp,prob, mutate_index):
    vector = np.copy(temp)
    for i in range(len(vector)):
        fact=random.uniform(-mutate_index, mutate_index)
        vector[i] = np.random.choice([vector[i]*(fact+1), vector[i]], p=[prob,1-prob])
        if(vector[i]<-10) :
            vector[i]=-10
        elif(vector[i]>10) :
            vector[i]=10
    
    return vector

#cross over function
def crossover(vector1, vector2, mutate_index,mutation_crs, index=-1):
    send1 = vector1.tolist()
    send2 = vector2.tolist()

    
    a = np.random.choice(np.arange(0, 11),5, replace=False)

    for i in a.tolist():
        send1[i] = np.copy(vector2[i])
        send2[i] = np.copy(vector1[i])

    return mutation_operator(send1,mutation_crs,mutate_index), mutation_operator(send2,mutation_crs,mutate_index),send1,send2,a

#crossover randomness fnction
def crossover_new(perrors_main, num):
    return random.sample(range(num), 2)


def main():

    mutate_index=0.1
    mutation_crs = 0.9
    print("pop_size:", pop_size, "gen_count:", gen_count, "cross_select",cross_select)
    print("surity",surity,"mutation_crs",mutation_crs,"mutate_index",mutate_index)

    overfit_vector = [0.0, -1.45799022e-12, -2.28980078e-13,  4.62010753e-11, -1.75214813e-10, -1.83669770e-15,  8.52944060e-16,  2.29423303e-05, -2.04721003e-06, -1.59792834e-08,  9.98214034e-10]
    final_vector_best = [-20, -20, -20, -20, -20, -20, -20, -20, -20, -20, -20]
    min_main_error = -1
    min_tainingerror = -1
    min_validationerror = -1

    perrors_main = np.zeros(pop_size)
    perrors_te = np.zeros(pop_size)
    perrors_ve = np.zeros(pop_size)
    population = np.zeros((pop_size, MAX_DEG))

    #step 1: generation 
    for i in range(pop_size):
        temp = np.copy(overfit_vector)
        population[i] = np.copy(mutation_operator(temp,0.85,mutate_index))

    #step 2: errors for each genaration
    for j in range(pop_size):
        temp = population[j].tolist()
        err = server.get_errors(SECRETKEY, temp)
        perrors_main[j] = np.copy((err[0]+err[1]))
        perrors_te[j] = np.copy((err[0]))
        perrors_ve[j] = np.copy((err[1]))

    #changing muatation parametrs.Note: this is subject to change regarding to get bestfit function
    for gen_count_num in range(gen_count):

        if((gen_count_num)%5==0 and gen_count_num!=0):
            mutate_index-=0.009
            mutation_crs+=0.009
            print("new mutate index :  ", mutate_index)



        
        arr_1=np.zeros((int(pop_size/2),2))
        arr_2=np.zeros((int(pop_size/2),5))

        
        childrens_list=np.zeros((pop_size,MAX_DEG))
        children_errors_mutated=np.zeros((pop_size,MAX_DEG))
        childern_error_main=np.zeros((pop_size))

        print("\n\n\n\ncurrent phase\t\t"+str(gen_count_num)+"-/-/-/-/-/")

        parenerrorsinds = perrors_main.argsort()
        perrors_main = np.copy(perrors_main[parenerrorsinds[::1]])
        perrors_te = np.copy(perrors_te[parenerrorsinds[::1]])
        perrors_ve = np.copy(perrors_ve[parenerrorsinds[::1]])
        population = np.copy(population[parenerrorsinds[::1]])

            
        
        # error value for 10 genearted persons
        for j in range(pop_size):
            print("pool " + str(j)+" errormain\t" + str(perrors_main[j]))
            print("pool " + str(j)+" errortrainig\t" + str(perrors_te[j]))
            print("pool " + str(j)+" errorvalidation" + str(perrors_ve[j]))
            print("\tpoollist"+str(population[j])+"\n\n")

        child_population = np.zeros((pop_size, MAX_DEG))
        new_gen_count = 0

        while(new_gen_count < pop_size):

            
            arr = crossover_new(perrors_main, cross_select)

            
            temp = crossover(population[arr[0]], population[arr[1]],mutate_index,mutation_crs)
            if temp[0].tolist() == population[arr[0]].tolist() or temp[1].tolist() == population[arr[0]].tolist() or temp[0].tolist() == population[arr[1]].tolist() or temp[1].tolist() == population[arr[1]].tolist():
                continue
            
            arr_1[int(new_gen_count/2)][0]=np.copy(arr[0])
            arr_1[int(new_gen_count/2)][1]=np.copy(arr[1])
            arr_2[int(new_gen_count/2)]=np.copy(np.sort(temp[4]))

            childrens_list[new_gen_count]=np.copy(temp[2])
            children_errors_mutated[new_gen_count]=np.copy(temp[0])
            child_population[new_gen_count] = np.copy(temp[0])
            new_gen_count += 1

            childrens_list[new_gen_count]=np.copy(temp[3])
            children_errors_mutated[new_gen_count]=np.copy(temp[1])
            child_population[new_gen_count] = np.copy(temp[1])
            new_gen_count += 1

        childerrors = np.zeros(pop_size)
        chidren_te = np.zeros(pop_size)
        chidren_ve = np.zeros(pop_size)

        # generate errors for each child
        for j in range(pop_size):
            temp = child_population[j].tolist()
            err = server.get_errors(SECRETKEY, temp)
            childerrors[j] = np.copy((err[0]+err[1]))
            chidren_te[j] = np.copy((err[0]))
            chidren_ve[j] = np.copy((err[1]))
            childern_error_main[j]=np.copy(childerrors[j])

        #sorting genrated children here
        children_population = np.copy(childerrors.argsort())
        childerrors = np.copy(childerrors[children_population[::1]])
        chidren_te = np.copy(chidren_te[children_population[::1]])
        chidren_ve = np.copy(chidren_ve[children_population[::1]])
        child_population = np.copy(child_population[children_population[::1]])
        
        tempstore_main = np.zeros(pop_size)
        tempstore_te = np.zeros(pop_size)
        tempstore_ve = np.zeros(pop_size)
        tempstore= np.zeros((pop_size, MAX_DEG))
        
        for j in range(surity):
            
            #adjusting the parent array
            tempstore[j]=np.copy(population[j])
            tempstore_main[j]=np.copy(perrors_main[j])
            tempstore_te[j]=np.copy(perrors_te[j])
            tempstore_ve[j]=np.copy(perrors_ve[j])
            
            #adjusting the child array
            tempstore[j+surity]=np.copy(child_population[j])
            tempstore_main[j+surity]=np.copy(childerrors[j])
            tempstore_te[j+surity]=np.copy(chidren_te[j])
            tempstore_ve[j+surity]=np.copy(chidren_ve[j])

        # single array for parents and childream
        
        parent_child = np.copy(np.concatenate([population[surity:], child_population[surity:]]))
        parnet_child_mainerror = np.copy(np.concatenate([perrors_main[surity:], childerrors[surity:]]))
        parent_child_te = np.copy(np.concatenate([perrors_te[surity:], chidren_te[surity:]]))
        parent_child_ve = np.copy(np.concatenate([perrors_ve[surity:], chidren_ve[surity:]]))

        # sorting the above array
        parent_child_sorted = parnet_child_mainerror.argsort()
        parnet_child_mainerror = np.copy(parnet_child_mainerror[parent_child_sorted[::1]])
        parent_child_te = np.copy(parent_child_te[parent_child_sorted[::1]])
        parent_child_ve = np.copy(parent_child_ve[parent_child_sorted[::1]])
        parent_child = np.copy(parent_child[parent_child_sorted[::1]])

        
        cand_gen_count = 0

        while(cand_gen_count + 2*surity < pop_size):
            tempstore[cand_gen_count+2*surity] = np.copy(parent_child[cand_gen_count])
            tempstore_main[cand_gen_count+2*surity] = np.copy(parnet_child_mainerror[cand_gen_count])
            tempstore_te[cand_gen_count+2*surity] = np.copy(parent_child_te[cand_gen_count])
            tempstore_ve[cand_gen_count+2*surity] = np.copy(parent_child_ve[cand_gen_count])
            cand_gen_count += 1


        #new population 
        population=np.copy(tempstore)
        perrors_main=np.copy(tempstore_main)
        perrors_te=np.copy(tempstore_te)
        perrors_ve=np.copy(tempstore_ve)

        #soting for evaluating min error here
        parenerrorsinds = perrors_main.argsort()
        perrors_main = np.copy(perrors_main[parenerrorsinds[::1]])
        perrors_te = np.copy(perrors_te[parenerrorsinds[::1]])
        perrors_ve = np.copy(perrors_ve[parenerrorsinds[::1]])
        population = np.copy(population[parenerrorsinds[::1]])


        if(min_main_error == -1 or min_main_error > perrors_main[0]):
            final_vector_best = np.copy(population[0])
            min_main_error = np.copy(perrors_main[0])
            min_tainingerror = np.copy(perrors_te[0])
            min_validationerror = np.copy(perrors_ve[0])
            nochange=0

        else:
            print("nothing imporved doing next phase")
            nochange+=1
            print(nochange)
        print("/./././././././././././././././././././\n")
        print("Min error = ", min_main_error, "\n\n")
        print("Min error train = ", min_tainingerror, "\n\n")
        print("Min error validation = ", min_validationerror, "\n\n")

        

    return final_vector_best



#printstatments and checking responce and submitting for server
final_vector_best = main()
#print("submitting final_best_vector",server.submit(SECRETKEY, final_vector_best.tolist()))
print("final_vector:\n")
print(final_vector_best)
print("genetic algorithm done")
