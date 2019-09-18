import glob
import pickle

import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib


def plot(test_loss, title):
    test_loss = np.array(test_loss)
    avg_test_loss = np.mean(test_loss, axis=0)
    yerr = np.std(test_loss, axis=0)*2

    plt.figure()
    plt.plot(np.arange(len(avg_test_loss)), avg_test_loss, label="Test loss")
    plt.fill_between(np.arange(len(avg_test_loss)), avg_test_loss - yerr,
                     avg_test_loss + yerr, alpha=0.2)
    plt.ylabel("MSE")
    plt.xlabel("# iterations")
    plt.legend()
    tikzplotlib.save("averaged-" + title + "-loss-iter.tex")


def main():

    data = []
    for filepath in glob.iglob("./cartpole-run-*.pkl"):
        with open(filepath, "rb") as f:
            data.append(pickle.load(f))

    all_times, avg_diff, test_loss, train_loss, lagr = zip(*data)

    plot(test_loss, "cartpole")
    plt.show()


if __name__ == "__main__":
    main()