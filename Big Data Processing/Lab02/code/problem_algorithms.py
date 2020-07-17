
# --------------------------------------------------------
#           PYTHON PROGRAM
# Here is where we are going to define our set of...
# - Imports
# - Global Variables
# - Functions
# ...to achieve the functionality required.
# When executing > python 'this_file'.py in a terminal,
# the Python interpreter will load our program,
# but it will execute nothing yet.
# --------------------------------------------------------

# ------------------------------------------------
#
#   ALGORITHM 1: count_inversions_n2
#
#   COMPLEXITY: n2
#
# ------------------------------------------------

# ------------------------------------------
# FUNCTION count_inversions_n2
# ------------------------------------------
def count_inversions_n2(my_rank_list):
    # 1. We create the output variable
    res = 0

    # 2. We compute the length of the list
    size = len(my_rank_list)

    # 3. We iterate to compute the number of inversions
    for i in range(size):
        for j in range(i+1, size):
            # 3.1. If A[i] > A[j] we increase the number of inversions
            if my_rank_list[i] > my_rank_list[j]:
                res = res + 1

    # 4. We return res
    return res

# ------------------------------------------------
#
#   ALGORITHM 2: mergesort_and_count_inversions
#
#   COMPLEXITY: n logn
#
# ------------------------------------------------

# ------------------------------------------
# FUNCTION: merge_count_inversions
# ------------------------------------------
def merge_count_inversions(l1, l2):
    # 1. We create the output variable
    res = ()

    # 1.1. We create the sort_list
    sort_list = []

    # 1.2. We create the num_inversions
    num_inversions = 0

    # 2. Auxiliary variables

    # 2.1. size_l1: Size of sub-list l1
    size_l1 = len(l1)

    # 2.2. size_l2: Size of sub-list l2
    size_l2 = len(l2)

    # 2.3. i: index being explored in sub-list l1
    i = 0

    # 2.4. j: index being explored in sub-list l2
    j = 0

    # 3. We merge both lists and count
    while (size_l1 > 0) or (size_l2 > 0):
        # 3.1. Case where both l1 and l2 have elements
        if (size_l1 > 0) and (size_l2 > 0):
            # 3.1.1. If the first element is the smallest
            if (l1[i] <= l2[j]):
                # I. We add it to the list
                sort_list.append(l1[i])
                # II. We get one element less in the first list
                size_l1 = size_l1 - 1
                # III. We increase the index for the first list
                i = i + 1

            # 3.1.2. If the second element is the smallest
            else:
                # I. We add it to the list
                sort_list.append(l2[j])
                # II. We get one element less in the first list
                size_l2 = size_l2 - 1
                # III. We increase the index for the second list
                j = j + 1
                # IV. We increase the amount of inversions found
                num_inversions = num_inversions + size_l1

        # 3.2. Case where one of the lists is empty
        else:
            # 3.2.1. When l1 is non-empty
            if (size_l1 > 0):
                # I. We concatenate the result with the rest of the list
                sort_list = sort_list + l1[i:]

                # II. We set the remaining unexplored part of the list to zero
                size_l1 = 0

            # 3.2.2. When l2 is non-empty
            else:
                # I. We concatenate the result with the rest of the list
                sort_list = sort_list + l2[j:]

                # II. We set the remaining unexplored part of the list to zero
                size_l2 = 0

    # 4. We assign res
    res = (sort_list, num_inversions)

    # 5. We return res
    return res

# ------------------------------------------
# FUNCTION mergesort_and_count_inversions
# ------------------------------------------
def mergesort_and_count_inversions(l):
    # 1. We create the output variable
    res = ()

    # 1.1. We create the sort_list
    sort_list = []

    # 1.2. We create the num_inversions
    num_inversions = 0

    # 2. We get the length of the list
    length = len(l)

    # 3. If the list is empty or contains just one item
    if (length <= 1):
        # 3.1. We get a fresh (deep) copy of the list
        sort_list = l[:]

        # 3.2. The number of inversions is 0
        num_inversions = 0

    # 4. If the list contains two or more elements
    else:
        # 4.1. We get the middle index of the list
        half = length // 2

        # 4.2. We get a fresh (deep) copy of each half of the list
        f = l[:half]
        s = l[half:]

        # 4.3. We solve recursively the 2 sub-problems
        (sort_list_1, num_inversions_1) = mergesort_and_count_inversions(f)
        (sort_list_2, num_inversions_2) = mergesort_and_count_inversions(s)

        # 4.4. We merge the two results and count the flipped inversions
        (sort_list, num_inversions_3) = merge_count_inversions(sort_list_1, sort_list_2)

        # 4.5. We count the total number of inversions
        num_inversions = num_inversions_1 + num_inversions_2 + num_inversions_3

    # 5. We assign res
    res = (sort_list, num_inversions)

    # 6. We return res
    return res

# ------------------------------------------
# FUNCTION count_inversions_nlogn
# ------------------------------------------
def count_inversions_nlogn(my_rank_list):
    # 1. We create the output variable
    res = 0

    # 2. We create an auxiliary variable to host the sort_list
    sort_list = None

    # 3. We call to mergesort_and_count_inversions
    (sort_list, res) = mergesort_and_count_inversions(my_rank_list)

    # 4. We return res
    return res
