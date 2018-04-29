
import matplotlib.pyplot as plt

def plot_nn_metric(history):
    if type(history) != dict:
        print("History input not dictionary...")
        print("Ending function")
        return()

    n_keys = len(list(history.keys()))
    keyss = list(history.keys())

    if n_keys > 2:
        print("More than 2 objects in model history")
        return(0)
    plt.ioff()

    plt.plot(history[keyss[0]])
    plt.title(keyss[0] + ' vs epoch')
    plt.ylabel(keyss[0])
    plt.xlabel('epoch')

    if n_keys > 1:
        plt.plot(history[keyss[1]])
        plt.legend([keyss[0], keyss[1]], loc='upper left')

    plt.show()
    print("Finished plotting")
    return()
