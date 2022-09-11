import sys

# Read command line arguments and convert to a list of integers
arr = sys.argv[1].split(',')
my_numbers = [None]*len(arr)
for idx, arr_val in enumerate(arr):
    my_numbers[idx] = int(arr_val)

# Print
print(f'Before sorting {my_numbers}')

def partition(array, lo, hi):
    pivot = array[hi]

    i = lo - 1

    for j in range(lo, hi):
        if array[j] <= pivot:
            i += 1
            array[i],array[j] = array[j],array[i]
           

    i += 1
    array[i],array[hi] = array[hi],array[i]
    return i

def quicksort(array, lo, hi):

    if lo >= hi:
        return

    p = partition(array, lo, hi)
    quicksort(array, lo, p - 1)
    quicksort(array, p + 1, hi)

quicksort(my_numbers,0,len(my_numbers)-1)

# Print
print(f'After sorting {my_numbers}')