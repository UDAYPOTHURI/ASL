# from timeit import default_timer as timer
# start_time = timer()

# #training

# end_time = timer()

def print_train_time(start, end, device=None):
    
    total_time = end - start
    print(f"\nTrain time on {device}: {total_time:.3f} seconds")
    return total_time