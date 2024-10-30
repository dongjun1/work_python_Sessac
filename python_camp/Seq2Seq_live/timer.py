import time 

def measure_runtime(func_to_measure):
    def f(*args, **kargs):
        from time import time 
        begin = time()  
        res = func_to_measure(*args, **kargs)
        end = time()
        print(f'function {func_to_measure} took {round(end - begin, 4)} sec')
        return res 
    return f 


def cache(func_to_cache):
    def f(*args, **kargs):        
        import os, pickle
        pickle_name = str((args, kargs))
        if pickle_name not in os.listdir():
            res = func_to_cache(*args, **kargs)
            pickle.dump(res, open(pickle_name, 'wb+'))
        else:
            res = pickle.load(open(pickle_name, 'rb'))
        
        return res 
    return f 


if __name__ == '__main__':
    from test import a 

    print(a) 

    with open('test.py', 'w+', encoding = 'utf-8') as f:
        f.write('a=2')
    print(a)
    # @measure_runtime 
    def some_function(t):
        print('Actually execute the function!')
        time.sleep(1)
        return t
    # measure_runtime(some_function)()
    print(cache(some_function)(2))

    # some_function()