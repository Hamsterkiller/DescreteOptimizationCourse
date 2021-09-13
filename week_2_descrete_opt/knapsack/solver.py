#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple, OrderedDict
from operator import attrgetter
Item = namedtuple("Item", ['index', 'value', 'weight', 'value_per_unit'])

def solve_relaxed_prob(items, capacity):
        # get the result for the linear relaxation problem

        # create better variable for the sorting
        for item in items:
            item['value_per_unit'] = item['value'] / item['unit']

        # sort items by value (decreasing)
        sorted(items, key=attrgetter('value_per_unit'), reverse=True)

        value = 0
        weight = 0
        taken = [0]*len(items)
        for item in items:
            if weight + item.weight <= capacity:
                taken[item.index] = 1
                value += item.value
                weight += item.weight
            elif (weight + item.weight >= capacity) & (weight < capacity):
                item_part_value = item.value * (capacity - weight) / item.capacity 
                value += item_part_value
                taken[item.index] = (capacity - weight) / item.capacity 

        return taken, relaxed_value

def solve_it_greedy(items, capacity):
    # greedy algorithm

    # sort items by value (decreasing)
    items = sorted(items, key=attrgetter('value_per_unit'), reverse=True)

    value = 0
    weight = 0
    taken = {}
    for item in items:
        taken[item.index] = 0
    for item in items:
        #print(item.index)
        if weight + item.weight <= capacity:
            taken[item.index] = 1
            value += item.value
            weight += item.weight

    taken = OrderedDict(sorted(taken.items()))

    return dict(taken), value

def solve_it(input_data):
    print(input_data)
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    items = []
    
    for i in range(1, item_count+1):
        line = lines[i]
        parts = line.split()
        items.append(Item(i-1, int(parts[0]), int(parts[1]), int(parts[0]) / int(parts[1])))

    taken, value = solve_it_dp(items, capacity)
    
    # prepare the solution in the specified output format
    output_data = str(value) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, taken.values()))
    return output_data

def solve_it_dp(items, capacity):
    import numpy as np
    # solver using dynamic programming approach
    
    items_count = len(items)
    #print(items[0].value)
    
    # define data structure - matrix of size K x I
    # where K - is the capacity of the knapsack
    # I - number of items in the problem
    dp_matrix = np.zeros((capacity, items_count+1))
    print(dp_matrix.shape)
    
    for i in range(1, items_count+1):
        value = items[i-1].value
        weight = items[i-1].weight
        for r in range(0, capacity):
            #print("""{} - {}""".format(i, r))
            if i == 1:
                if r + 1 >= weight:
                    dp_matrix[r, i] = value
            elif i == 2:
                if (r + 1 < np.minimum(items[i-2].weight, weight)):
                    dp_matrix[r, i] = dp_matrix[r, i-1]
                elif (r + 1 >= np.minimum(items[i-2].weight, weight)) & (r + 1 < np.maximum(items[i-2].weight, weight)):
                    dp_matrix[r, i] = np.minimum(items[i-2].value, value)
                elif (r + 1 >= max(items[i-2].weight, weight)) & (r + 1 < weight + items[i-2].weight):
                    dp_matrix[r, i] = np.maximum(items[i-2].value, value)
                else:
                    dp_matrix[r, i] = value + items[i-2].value
            else:
                if (r + 1) < weight:
                    dp_matrix[r, i] = dp_matrix[r, i-1]
                elif ((r + 1) >= weight) & ((r + 1) < sum([el.weight for el in items[0:i]])):
                    dp_matrix[r, i] = np.maximum(dp_matrix[r, i-1], value + dp_matrix[np.maximum(r - weight, 0), i-1])
                else:
                    dp_matrix[r, i] = sum([el.value for el in items[0:i-1]])
                        
    print('Max value of the knapsack = {}'.format(dp_matrix[capacity-1, items_count]))
    # find optimal set of items
    r = capacity-1
    c = items_count
    taken = {}
    optimal_values_set = []
    optimal_weight_set = []
    while c >= 1:
        if (dp_matrix[r, c] != dp_matrix[r, c-1]):
            taken[c] = 1
            optimal_values_set.append(items[c-1].value)
            optimal_weight_set.append(items[c-1].weight)
            r = np.maximum(r - items[c-1].weight, 0)
        else:
            taken[c] = 0
        c -= 1
        
    print('Values taken: \n')
    print(optimal_values_set)                
    value = sum(optimal_values_set)  
    print('Weight of the knapsack = {}'.format(sum(optimal_weight_set)))
    taken = OrderedDict(sorted(taken.items()))
    
    return dict(taken), value

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')

