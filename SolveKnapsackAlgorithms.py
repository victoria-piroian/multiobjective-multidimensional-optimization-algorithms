# imports
import time
import copy
import collections 
import queue as Q 
import numpy as np 
import pandas as pd 
import scipy as sp
import os
import gurobipy as gp
from gurobipy import GRB
import random

gp.setParam("OutputFlag", 0)
gp.setParam("MIPGap", 0.000001)

curr_dir = os.getcwd() + '/'

def SolveKnapsack(filename, method=1):
  groupNo = 7
  methodName = ''
  startTime = time.time()
  
  if method == 1:
    methodName = "BF"

    # n,b,c,a = read_input(filename)
    # feas = [] 
    # arr = np.empty(n,dtype=int)

    # generateAllBinaryCombos(n, arr, 0, a, b, feas)

    # z = findFeasibleImages(feas, c)

    # removeDuplicates(z)

    # ndp_array = removeDominated(z)
    
    ''' IMPROVED '''
    n,b,c,a = read_input(filename)
    foundNDPs = []
    feas = [] 
    arr = np.empty(n,dtype=int)

    generateAllBinaryCombos(n, arr, 0, a, b, feas)

    for i in feas:
      break_out_flag = False
      z = []
      for j in c: # c is a list of lists of the objective function coefficients
        z.append(np.dot(i,j)) # calculate a z-point
      if z not in foundNDPs:
        dominated = []
        for ndp in foundNDPs:
            compare_less = (np.asarray(z) <= np.asarray(ndp))
            if (np.all(compare_less, axis = 0) == False and np.any(compare_less, axis = 0) == False): # z is dominated by ndf so break (we don't want to add to foundNDFs)
              break_out_flag = True
              break
            # we know z != ndp, so if all respective points are less or equal, then there is
            # at least one point that dominates the other respective point, therefore z dominates ndp and 
            # ndp should not be in foundNDPs
            if (np.all(compare_less, axis = 0) == True):
              dominated.append(ndp)
        if break_out_flag == False:
          for k in dominated:
            foundNDPs.remove(k)
          foundNDPs.append(z)

    foundNDPs = np.asarray(foundNDPs)

    runtime = time.time() - startTime

    ndp_array_sorted = sortArrayLexicographically(foundNDPs)
    
    # summary = [Solution time measured in seconds, Number of obtained NDPs, 0]
    summary = np.array([runtime, np.shape(foundNDPs)[0], 0])  

  elif method == 2:
    methodName = "RDM"
    # Read and solve an instance via Rectangle Divison Method (RDM)

    foundNDPs = []
    num_recs_searched = 0

    # Read input file
    n,b,c,a = read_input(filename)
    m =  len(b)
    J = 2

    # define gurobi model
    model = get_model(n, m, J, c, a, b)    

    # get most western point
    z_nw = lexmin(J, model, first_obj=1)
    foundNDPs.append(z_nw)

    # get most southern point
    z_se = lexmin(J, model, first_obj=2)
    if (z_nw != z_se):
      foundNDPs.append(z_se)

    # initialize list of rectangles
    rectangles_list = [[z_nw,z_se]]

    while len(rectangles_list) != 0:
      num_recs_searched += 1
      R = rectangles_list[0]
      # print(R)
      rectangles_list.remove(R)
      
      z1 = R[0]
      z2 = R[1]
      
      # bisect to create bottom rectangle
      R2 = [[z1[0],(z1[1]+z2[1])/2],z2]

      # look for an NDP in bottom rectangle (get most western point)
      z_hat = lexmin(J, model, first_obj=1,NW=R2[0],SE=R2[1])

      if (z_hat is not None):

        if (z_hat != z2): 
          foundNDPs.append(z_hat) # a new NDP is found
          rectangles_list.append([z_hat,z2])  

        # create refined top rectangle
        R3 = [z1,[z_hat[0]-0.0001,(z1[1]+z2[1])/2]]

        # look for an NDP in top rectangle (get most southern point)
        z_squiggly = lexmin(J, model, first_obj=2,NW=R3[0],SE=R3[1])

        if (z_squiggly is not None):
          if (z_squiggly != z1):
            foundNDPs.append(z_squiggly)  # a new NDP is found
            rectangles_list.append([z1,z_squiggly]) # Refine the top rectangle

    '''IMPROVED RDM

    foundNDPs = []

    # Read input file
    n,b,c,a = read_input(filename)
    feas = [] 
    arr = np.empty(n,dtype=int)
    m =  len(b)
    J = 2
    model = get_model(n, m, J, c, a, b)    

    # get most north-western point
    z_nw = lexmin(J, model, first_obj=1)
    foundNDPs.append(z_nw)
    
    # get most south-eastern point
    z_se = lexmin(J, model, first_obj=2)
    foundNDPs.append(z_se)

    rectangles_list = [[z_nw,z_se]]

    while len(rectangles_list) != 0:
      rec = rectangles_list[0]
      z1 = rec[0]
      z2 = rec[1]

      rectangles_list.remove(rec)

      z1x = z1[0] #x-coordinate of the nw point
      z1y = z1[1] #y-coordinate of the nw point
      z2x = z2[0] #x-coordinate of the se point
      z2y = z2[1] #y-coordinate of the se point 
      
      rec_height = z1y -z2y
      z_mid_y = (z1y + z2y)/2 #midpoint (y axis)
      
      if rec_height == 2:
          #no need to bisect, points can only dominate others if they are more west than the rest
          R = [[z1x,z1y-0.0001],[z2x-0.0001 ,z2y]]
          z_hat_west = lexmin(J, model, first_obj=1,NW=R[0],SE=R[1])
          
          if (z_hat_west is not None):
              foundNDPs.append(z_hat_west)
              
      elif rec_height == 3:
          R = [[z1x,z1y-0.0001],[z2x-0.0001 ,z2y]]
          z_hat_west = lexmin(J, model, first_obj=1,NW=R[0],SE=R[1])
          z_hat_south = lexmin(J, model, first_obj=2,NW=R[0],SE=R[1])
          
          if (z_hat_west is not None and z_hat_south is not None):
              if (z_hat_west == z_hat_south):
                  foundNDPs.append(z_hat_west)
              else:
                  foundNDPs.append(z_hat_west)
                  foundNDPs.append(z_hat_south)
                  # no need to add rectangle since it is certain that it would not contain any points
                  
      elif rec_height == 4:
          R2 = [[z1x,z_mid_y],[z2x-0.0001 ,z2y]]
          z_hat_west = lexmin(J, model, first_obj=1,NW=R2[0],SE=R2[1])
          z_hat_south = lexmin(J, model, first_obj=2,NW=R2[0],SE=R2[1])
          
          if (z_hat_west is not None and z_hat_south is not None):
          
              if (z_hat_west == z_hat_south):
                  foundNDPs.append(z_hat_west)
              else:
                  foundNDPs.append(z_hat_west)
                  foundNDPs.append(z_hat_south)
              
              R3 = [[z1x,z1y-0.0001],[z_hat_west[0]-0.0001,z_mid_y]]
          
          else:
              R3 = [[z1x,z1y-0.0001],[z2x-0.0001,z_mid_y]] 
          
          z_squiggly_west = lexmin(J, model, first_obj=1,NW=R3[0],SE=R3[1])
      
          if (z_squiggly_west is not None):
              foundNDPs.append(z_squiggly_west)
      
      elif rec_height == 5:
          R2 = [[z1x,z_mid_y],[z2x-0.0001 ,z2y]]
          z_hat_west = lexmin(J, model, first_obj=1,NW=R2[0],SE=R2[1])
          z_hat_south = lexmin(J, model, first_obj=2,NW=R2[0],SE=R2[1])
          
          if (z_hat_west is not None and z_hat_south is not None):
          
              if (z_hat_west == z_hat_south):
                  foundNDPs.append(z_hat_west)
              else:
                  foundNDPs.append(z_hat_west)
                  foundNDPs.append(z_hat_south)
              
              R3 = [[z1x,z1y-0.0001],[z_hat_west[0]-0.0001,z_mid_y]]
          
          else:
              R3 = [[z1x,z1y-0.0001],[z2x-0.0001,z_mid_y]] 
          
          z_squiggly_west = lexmin(J, model, first_obj=1,NW=R3[0],SE=R3[1])
          z_squiggly_south = lexmin(J, model, first_obj=2,NW=R3[0],SE=R3[1])
      
          if (z_squiggly_west is not None and z_squiggly_south is not None):
              
              if (z_squiggly_west == z_squiggly_south):
                  foundNDPs.append(z_squiggly_south)
              else:
                  foundNDPs.append(z_squiggly_west)
                  foundNDPs.append(z_squiggly_south)
                  
      elif rec_height == 6:
          R2 = [[z1x,z_mid_y],[z2x-0.0001 ,z2y]]
          z_hat_west = lexmin(J, model, first_obj=1,NW=R2[0],SE=R2[1])
          z_hat_south = lexmin(J, model, first_obj=2,NW=R2[0],SE=R2[1])
          
          if (z_hat_west is not None and z_hat_south is not None):
          
              if (z_hat_west == z_hat_south):
                  foundNDPs.append(z_hat_west)
              else:
                  foundNDPs.append(z_hat_west)
                  foundNDPs.append(z_hat_south)
                  rectangles_list.append([z_hat_west,z_hat_south])
                  
              
              R3 = [[z1x,z1y-0.0001],[z_hat_west[0]-0.0001,z_mid_y]]
          
          else:
              R3 = [[z1x,z1y-0.0001],[z2x-0.0001,z_mid_y]] 
          
          z_squiggly_west = lexmin(J, model, first_obj=1,NW=R3[0],SE=R3[1])
          z_squiggly_south = lexmin(J, model, first_obj=2,NW=R3[0],SE=R3[1])
      
          if (z_squiggly_west is not None and z_squiggly_south is not None):
              
              if (z_squiggly_west == z_squiggly_south):
                  foundNDPs.append(z_squiggly_south)
              else:
                  foundNDPs.append(z_squiggly_west)
                  foundNDPs.append(z_squiggly_south)
                  
      else:
          R2 = [[z1x,z_mid_y],[z2x-0.0001 ,z2y]]

          # get most western point within R2
          z_hat_west = lexmin(J, model, first_obj=1,NW=R2[0],SE=R2[1])
          z_hat_south = lexmin(J, model, first_obj=2,NW=R2[0],SE=R2[1])
          
          if (z_hat_west is not None and z_hat_south is not None):
              
              if (z_hat_west == z_hat_south):
                  foundNDPs.append(z_hat_west)
              else:
                  foundNDPs.append(z_hat_west)
                  foundNDPs.append(z_hat_south)
                  rectangles_list.append([z_hat_west,z_hat_south])
                  
              R3 = [[z1x,z1y-0.0001],[z_hat_west[0]-0.0001,z_mid_y]]
              
          else:
              R3 = [[z1x,z1y-0.0001],[z2x-0.0001,z_mid_y]] 
              
          z_squiggly_west = lexmin(J, model, first_obj=1,NW=R3[0],SE=R3[1])
          z_squiggly_south = lexmin(J, model, first_obj=2,NW=R3[0],SE=R3[1])
          
          if (z_squiggly_west is not None and z_squiggly_south is not None):
              
              if (z_squiggly_west == z_squiggly_south):
                  foundNDPs.append(z_squiggly_south)
              else:
                  foundNDPs.append(z_squiggly_west)
                  foundNDPs.append(z_squiggly_south)
                  rectangles_list.append([z_squiggly_west,z_squiggly_south])'''

    foundNDPs = np.asarray(foundNDPs)

    runtime = time.time() - startTime

    ndp_array_sorted = sortArrayLexicographically(foundNDPs)
    
    # summary = [Solution time measured in seconds, Number of obtained NDPs, number of rectangles searched]
    summary = np.array([runtime, np.shape(foundNDPs)[0], num_recs_searched])  

  elif method == 3:
    methodName = "SPM"
    foundNDPs = []
    num_regions_searched = 0
    
    # Read input file
    n,b,c,a = read_input(filename)
    arr = np.empty(n,dtype=int)

    J = len(c)
    M = len(b)

    supernal_point = []
    for i in range(J):
        supernal_point.append(0)

    # initialize the list of regions
    regions = []
    regions.append(supernal_point)

    # pick the most north-east region
    reg = regions[0]
    
    lmbd = np.random.rand(J)
    lmbd = lmbd/np.sum(lmbd)

    model_ws = get_weighted_sum_model(c, a, b, n, M, J, reg, lmbd)

    while (len(regions)):
        num_regions_searched += 1

        # pick the most north-east region
        reg = regions[0]

        for i, r in enumerate(reg):
            model_ws._z[i].ub = r
        model_ws.update()
        # solve min lambda*z in reg
        model_ws.optimize()

        if model_ws.status == 2:
            z_star = get_supernal_z(n, c, model_ws)

            # add z to found NPDs
            foundNDPs.append(z_star)

            regions_temp = []
            regions_to_remove = []
            for region in regions:

                count = 0
                for i in range(J):
                    # check if equality
                    if z_star[i] <= region[i]:
                        count += 1
                
                # pseudo line 11

                if count == J:
                    regions_to_remove.append(region)

                    new_reg = []
              
                    # i is the number of regions you are adding
                    # j is the index in that region
                    for i in range(J):
                        new_reg = list(region)

                        for j in range(J):
                  
                            if i == j:
                                # plus one or minus 1                        
                                new_reg[j] = z_star[j] - 1

                        regions_temp.append(new_reg)

            for r in regions_to_remove:
                regions.remove(r)
            for r in regions_temp:
                regions.append(r)
                
            if J >= 3:
                removeDominated(np.array(regions))
              
        else:
            regions.remove(reg)
    
    foundNDPs = np.asarray(foundNDPs)

    runtime = time.time() - startTime

    ndp_array_sorted = sortArrayLexicographically(foundNDPs)
    
    # summary = [Solution time measured in seconds, Number of obtained NDPs, number of rectangles searched]
    summary = np.array([runtime, np.shape(foundNDPs)[0], num_regions_searched])  

  elif method == 4:
    methodName = "COMP_2D"

    foundNDPs = []
    num_regions_searched = 0
    
    # Read input file
    n,b,c,a = read_input(filename)
    arr = np.empty(n,dtype=int)

    J = len(c)
    M = len(b)

    supernal_point = []
    for i in range(J):
        supernal_point.append(0)

    # initialize the list of regions
    regions = []
    regions.append(supernal_point)

    # pick the most north-east region
    reg = regions[0]

    # imrprovement: intialize the lambdas statically
    lmbd = [1]*J

    model_ws = get_weighted_sum_model(c, a, b, n, M, J, reg, lmbd)

    while (len(regions)):
        # improvement: dynamically change lambda's to prioritze weighting for the favoured objective function
        # reset lambda
        for j in range(J):
          lmbd[j] = 0
        # calculate the distance from the origin
        for ndp in foundNDPs:
          for j in range(J):
            lmbd[j] += ndp[j]

        num_regions_searched += 1

        # improvement: randomly select the next region to be solved next
        reg = random.choice(regions)

        for i, r in enumerate(reg):
            model_ws._z[i].ub = r
            model_ws._lmbd_new[i] = lmbd[i]
        model_ws.update()
        # solve min lambda*z in reg
        model_ws.optimize()
        
        if model_ws.status == 2:
            z_star = get_supernal_z(n, c, model_ws)

            # add z to found NPDs
            foundNDPs.append(z_star)

            regions_temp = []
            regions_to_remove = []

            # improvement: remove regions before the loop on individual regions
            if J >= 3:
                removeDominated(np.array(regions))
            for region in regions:

                count = 0
                for i in range(J):
                    # check if equality
                    if z_star[i] <= region[i]:
                        count += 1
                
                # pseudo line 11

                if count == J:
                    regions_to_remove.append(region)

                    new_reg = []
              
                    # i is the number of regions you are adding
                    # j is the index in that region
                    for i in range(J):
                        new_reg = list(region)

                        for j in range(J):
                  
                            if i == j:
                                # plus one or minus 1                        
                                new_reg[j] = z_star[j] - 1

                        regions_temp.append(new_reg)

            for r in regions_to_remove:
                regions.remove(r)
            for r in regions_temp:
                regions.append(r)
                
              
        else:
            regions.remove(reg)

    foundNDPs = np.asarray(foundNDPs)

    runtime = time.time() - startTime

    ndp_array_sorted = sortArrayLexicographically(foundNDPs)
    
    # summary = [Solution time measured in seconds, Number of obtained NDPs, 0]
    summary = np.array([runtime, np.shape(foundNDPs)[0], 0])  

  elif method == 5:
    methodName = "COMP_3D"
    
    foundNDPs = []
    num_regions_searched = 0
    
    # Read input file
    n,b,c,a = read_input(filename)
    arr = np.empty(n,dtype=int)

    J = len(c)
    M = len(b)

    supernal_point = []
    for i in range(J):
        supernal_point.append(0)

    # initialize the list of regions
    regions = []
    regions.append(supernal_point)

    # pick the most north-east region
    reg = regions[0]

    # imrprovement: intialize the lambdas statically
    lmbd = [1]*J

    model_ws = get_weighted_sum_model(c, a, b, n, M, J, reg, lmbd)

    while (len(regions)):
        # improvement: dynamically change lambda's to prioritze weighting for the favoured objective function
        # reset lambda
        for j in range(J):
          lmbd[j] = 0
        # calculate the distance from the origin
        for ndp in foundNDPs:
          for j in range(J):
            lmbd[j] += ndp[j]

        num_regions_searched += 1

        # improvement: randomly select the next region to be solved next
        reg = random.choice(regions)

        for i, r in enumerate(reg):
            model_ws._z[i].ub = r
            model_ws._lmbd_new[i] = lmbd[i]
        model_ws.update()
        # solve min lambda*z in reg
        model_ws.optimize()
        
        if model_ws.status == 2:
            z_star = get_supernal_z(n, c, model_ws)

            # add z to found NPDs
            foundNDPs.append(z_star)

            regions_temp = []
            regions_to_remove = []

            # improvement: remove regions before the loop on individual regions
            if J >= 3:
                removeDominated(np.array(regions))
            for region in regions:

                count = 0
                for i in range(J):
                    # check if equality
                    if z_star[i] <= region[i]:
                        count += 1
                
                # pseudo line 11

                if count == J:
                    regions_to_remove.append(region)

                    new_reg = []
              
                    # i is the number of regions you are adding
                    # j is the index in that region
                    for i in range(J):
                        new_reg = list(region)

                        for j in range(J):
                  
                            if i == j:
                                # plus one or minus 1                        
                                new_reg[j] = z_star[j] - 1

                        regions_temp.append(new_reg)

            for r in regions_to_remove:
                regions.remove(r)
            for r in regions_temp:
                regions.append(r)
                
              
        else:
            regions.remove(reg)
    
    foundNDPs = np.asarray(foundNDPs)

    runtime = time.time() - startTime

    ndp_array_sorted = sortArrayLexicographically(foundNDPs)
    
    # summary = [Solution time measured in seconds, Number of obtained NDPs, number of rectangles searched]
    summary = np.array([runtime, np.shape(foundNDPs)[0], num_regions_searched])
  
  # Output result
  ndp_filename = f'{methodName}_NDP_{groupNo}.txt' 
  summary_filename = f'{methodName}_SUMMARY_{groupNo}.txt'

  np.savetxt(curr_dir + ndp_filename, ndp_array_sorted, delimiter='\t', newline='\n')
  np.savetxt(curr_dir + summary_filename, summary, delimiter='\t', newline='\n')
  
  return

def read_input(input_file):

  f = open(input_file)

  n = int(f.readline()[:-1])  # number of items

  b = []
  c = []
  a = []

  for line in f:
    x = line[:-1].split()
    x = [eval(i) for i in x]
    if (len(x) == 1): # won't work if n == 1, because c and a will also have length 1
      b.append(x[0])
    else:
      if (x[0] < 0):
        c.append(x) # c contains j vectors where j is the # of objective functions. Each vector contains n obj function coefficients.
      else:
        a.append(x) # a contains m vectors. Each vector contains n costraint coefficients.

  f.close()

  return n,np.asarray(b),np.asarray(c),np.asarray(a)

# Recursive function to generate all binary combos / enumerate all points in X (backtracking algorithm)
def generateAllBinaryCombos(n, arr, i, a, b, feas):
 
    if i == n:
      satisfied_constraints = True
      for i in range(len(b)):
        if (np.dot(arr,a[i]) > b[i]):
          satisfied_constraints = False
          break
      if satisfied_constraints == True:
        feas.append(arr.copy())
      return
     
    # First assign "0" at ith position and try for all other permutations for remaining positions
    arr[i] = 0
    generateAllBinaryCombos(n, arr, i + 1, a, b, feas)
 
    # And then assign "1" at ith position and try for all other permutations for remaining positions
    arr[i] = 1
    generateAllBinaryCombos(n, arr, i + 1, a, b, feas)


def findFeasibleImages(x, c):
  z = []
  for i in x:
    point_z = []
    for j in c:
      point_z.append(np.dot(i,j))
    z.append(point_z)
  return np.array(z)
  

def removeDuplicates(z):
  return np.unique(z, axis=0)


def removeDominated(z):
  '''
  want to maximize benefit (objective functions)

  check if all elements are >= (e.g. a >= b)
    if true, check if all elements are ==
      if true, it is not an improvement
      if false, then it is an improvement

  Need to compare all elements of z with each other (n^2 time?)
  '''

  ndf = z
  i = 0
  z_remaining = True

  # to dominate, at least one needs to be greater and the rest can be equal
  # need to loop through every item remaining in the ndf

  while i < len(ndf):
    curr_z = ndf[i]
    dominated = []
    for test_z in ndf:
      if np.array_equal(curr_z, test_z) == False:
        compare_less = (curr_z <= test_z)
        if (np.all(compare_less, axis = 0) == True):
          # we know curr_z != test_z, so if all respective points are greater or equal, then there is
          # at least one point that dominates the other respective point, therefore curr_z dominates test_z and 
          # test_z should not be in the ndf
          dominated.append(test_z)
    # remove dominated points from ndf
    for k in dominated:
      list_ndf = ndf.tolist()
      list_ndf.remove(k.tolist())
      ndf = np.asarray(list_ndf)
    if (len(dominated) > 0):
      i = np.maximum(i-len(dominated),0)
    else:
      i+=1

  return ndf

def sortArrayLexicographically(ndp_array):
  # sorts an array in lexicographical decreasing order
  # i.e., first sorted in terms of the first objective value from the highest to the lowest, 
  # then in terms of the second objective value, and so on

  ndp_array = ndp_array[ndp_array[:, 0].argsort()]
  
  for i in range(1,np.shape(ndp_array)[1]):
    ndp_array = ndp_array[ndp_array[:, i].argsort(kind='mergesort')]

  return ndp_array

def get_model(n, m, J, C, A, B):
    """Get the model min_{x in X} alpha_1 * z_1 + alpha_2 * z_2"""
    model = gp.Model(f'z_model')

    # x is a binary decision variable with n dimensions
    x = model.addVars(n, vtype='B', name='x')

    # Define variables for objective
    z = []
    for i in range(J):
        z.append(model.addVar(vtype='I', name=f'z_{i}', obj=1))
        
    # Attach variables to model
    model._x, model._z = x, z    
        
    # Set the z values
    for i in range(J):
        model.addConstr(z[i] == gp.quicksum(C[i][j]*x[j] for j in range(n)))


    # The x in \mathcal X constraint
    for i in range(m):
        model.addConstr(gp.quicksum(A[i][j]*x[j] for j in range(n)) <= B[i])

    # The constraints imposed by the region. Since we have defined the objective as 
    # a variable, we can simply modify its upper bound to impose the constraint.
    for i in range(J):
        z[i].ub = 0
        z[i].lb = -gp.GRB.INFINITY

    # Objective
    # alpha_1 and alpha_2 is 1 for now
    model.setObjective(z[0] + z[1], sense=gp.GRB.MINIMIZE)
    
    return model
  
  
def lexmin(J, model=None, first_obj=1, NW=None, SE=None):    
    # set the first and second obj index    
    assert 1 <= first_obj <= 2
    
    z1, z2 = model._z[0], model._z[1]
    if NW == None and SE == None:        
        z1.ub, z1.lb = 0, -gp.GRB.INFINITY        # -infty <= z1 <= 0        
        z2.ub, z2.lb = 0, -gp.GRB.INFINITY        # -infty <= z2 <= 0
    elif NW is not None and SE is not None:        
        z1.ub, z1.lb = SE[0], NW[0]               # NW.x <= z1 <= SE.x                
        z2.ub, z2.lb = NW[1], SE[1]               # SE.y <= z2 <= NW.y
    else:
        raise ValueError('Invalid NW and SE')
        
    # .Obj allows you modify the objective coefficient of a given variable
    # Modify the objective to: 1 x z_1 + 0 x z_2 = z_1 if first_obj == 1
    # Or modify the objective to: 0 x z_1 + 1 x z_2 = z_2 if first_obj == 2
    if first_obj == 1:
        z1.Obj, z2.Obj = 1, 0 
    else:
        z1.Obj, z2.Obj = 0, 1
    
    # Optimize
    model.update()
    model.optimize()
    
    # Checking the model status to verify if the model is solved to optimality
    if model.status == 2:
        first_obj_val = int(np.round(model.objval))
        
        # Update bound and objective coefficients
        if first_obj == 1:
            z1.ub = first_obj_val
            z1.Obj, z2.Obj = 0, 1
        else:
            z2.ub = first_obj_val
            z1.Obj, z2.Obj = 1, 0
              
        # Optimize
        model.update()
        model.optimize()
        
        if model.status == 2:
            second_obj_val = int(np.round(model.objval))
            
            return [first_obj_val, second_obj_val] if first_obj == 1 else [second_obj_val, first_obj_val]
                        
    return None

def get_weighted_sum_model(C, A, B, n, M, J, region, lam):
    model = gp.Model()

    # x is a binary decision variable with n dimensions
    x = model.addVars(n, vtype='B', name='x')
    
    # Define variables for objective
    z = []
    for i in range(J):
        z.append(model.addVar(vtype='I', name=f'z_{i}'))

    # improvement: dynamically change the lambda's
    lmbd_new = []
    for i in range(J):
        lmbd_new.append(model.addVar(vtype='I', name=f'lmbd_new_{i}'))

    # Attach the vars to the model object
    model._x = x
    model._z = z
    model._lmbd_new = lmbd_new
        
    # Set the objectives
    for i in range(J):
        model.addConstr(z[i] == gp.quicksum(C[i][j]*x[j] for j in range(n)))

    # The x in \mathcal X constraint
    for i in range(M):
        model.addConstr(gp.quicksum(A[i][j]*x[j] for j in range(n)) <= B[i])

    # The constraints imposed by the region. Since we have defined the objective as 
    # a variable, we can simply modify its upper bound to impose the constraint.
    for i in range(J):
      z[i].ub = region[i]
      z[i].lb = -gp.GRB.INFINITY
      lmbd_new[i] = lam[i]

    # Objective
    model.setObjective(gp.quicksum(lmbd_new[i]*z[i] for i in range(J)), 
                       sense=gp.GRB.MINIMIZE)
    
    return model

def get_supernal_z(n, C, model):
  x_var = model._x
  x_sol = [int(np.round(x_var[i].x)) for i in range(n)]

  return np.dot(C, x_sol)