import os
import sys
import copy
import importlib

back = os.environ['KERAS_BACKEND']
GPU_NUM = int(os.environ['GPU_NUM'])
metrics = ["training_time", "max_memory", "training_accuracy", "test_accuracy"]
example_list=list()
result = dict()


def run_benchmark():
    result[back] = dict()
    test_summary = open('test_summary_' + str(back) +
                        str(GPU_NUM) + '.txt', 'w')

    #If example_list is not empty, use it as example set
    #Otherwise run all examples under keras_example folder
    example_dir = 'keras_example/'
    example_set = [os.path.join(example_dir, fname) for fname in os.listdir(example_dir)] \
                  if len(example_list) == 0 else example_list
    for fname in example_set:
        module = fname[:-1] if fname.endswith('.py') else fname
        try:
            example = importlib.import_module(module)
            result[back][module] = copy.deepcopy(example.ret_dict)
            output = ''
            output += "{backend:<20}\n".format(backend=back)
            output += "{describe:<40}".format(describe='exampe/metric')
            for metric in metrics:
                output += "{metric:<25}".format(metric=metric)
            output += '\n'
            output += "{module:<40}".format(module=module)
            for metric in metrics:
                output += "{metric:<25}".format(metric=result[back][module][metric])
            output += '\n'
        except Exception as e:
            output = ''
            output += '%s on %s with %s GPU(s) returned error\n%s\n:' \
                      % (module, back, str(GPU_NUM), str(e))
        finally:
            test_summary.write(output)
            print output

        del sys.modules[module]
    test_summary.close()

if __name__ == '__main__':
    run_benchmark()
