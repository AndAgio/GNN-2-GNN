import os
import random
import matplotlib.pyplot as plt
import numpy as np


def scatter_plot_acc_vs_footprint(dictionary_of_method_points, dataset='nas101', out_path='out_plots'):
    # Define filename for output
    filename = os.path.join(out_path, 'distribution.pdf')
    # Get keys names
    keys_names = list(dictionary_of_method_points.keys())
    # Setup figure and matplotlib
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.rcParams.update({'font.size': 25})
    plt.rcParams['text.usetex'] = True
    # Define random colors for the different classes of points
    if 'RANDOM' in keys_names:
        colors = ['blue', 'red', 'orange', 'yellow', 'forestgreen']
    else:
        colors = ['blue', 'orange', 'yellow', 'greenyellow', 'forestgreen']
    # Setup markers
    if 'RANDOM' in keys_names:
        markers = ['o', 'x', '^', '^', '*']
    else:
        markers = ['o', '^', '^', '*', '*']
    # Setup other plot parameters
    zorders = [i+1 for i in range(len(dictionary_of_method_points))]
    sizes = [30, 90, 90, 100, 100]
    # Iterate over entry of dictionary and scatter them
    index = 0
    for name, list_of_points in dictionary_of_method_points.items():
        xs = [acc_vs_foot[1] for acc_vs_foot in list_of_points]
        if dataset == 'nats':
            ys = [acc_vs_foot[0] for acc_vs_foot in list_of_points]
        else:
            ys = [acc_vs_foot[0]*100 for acc_vs_foot in list_of_points]
        plt.scatter(xs, ys, s=sizes[index], color=colors[index], marker=markers[index], alpha=1, label=name, zorder=zorders[index])
        index += 1
    plt.legend(loc='lower right')
    if dataset == 'nas101':
        plt.xlabel('# Parameters', fontsize=25)
    elif dataset == 'nats':
        plt.xlabel('Footprint (MB)', fontsize=25)
    plt.ylabel('Accuracy (%)', fontsize=25)
    plt.xticks(fontsize=25)
    plt.yticks(np.arange(0, 100, step=10), fontsize=25)
    if dataset == 'nas101':
        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
    plt.savefig(filename)
    plt.show()


def bar_plot_acc_vs_footprint(dictionary_of_method_points, dataset='nas101', out_path='out_plots'):
    # Define filename for output
    filename = os.path.join(out_path, 'bar_plot.pdf')
    # Extract accuracies only from the dictionary
    acc_bins = [i for i in range(0, 100, 5)]
    if dataset == 'nats':
        dictionary_of_method_points = {key: [acc_vs_foot[0] for acc_vs_foot in list_of_points] for
                                       key, list_of_points in dictionary_of_method_points.items()}
    else:
        dictionary_of_method_points = {key: [acc_vs_foot[0]*100 for acc_vs_foot in list_of_points] for key, list_of_points in dictionary_of_method_points.items()}
    # Get keys names
    keys_names = list(dictionary_of_method_points.keys())
    # Setup figure and matplotlib
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.rcParams.update({'font.size': 25})
    plt.rcParams['text.usetex'] = True
    # Define random colors for the different classes of points
    if 'RANDOM' in keys_names:
        colors = ['blue', 'red', 'orange', 'yellow', 'forestgreen']
    else:
        colors = ['blue', 'orange', 'yellow', 'greenyellow', 'forestgreen']
    # Setup markers
    if 'RANDOM' in keys_names:
        markers = ['o', 'x', '^', '^', '*']
    else:
        markers = ['o', '^', '^', '*', '*']
    # Setup other plot parameters
    zorders = [i+1 for i in range(len(dictionary_of_method_points))]
    sizes = [30, 90, 90, 100, 100]
    # Iterate over entry of dictionary and scatter them
    index = 0
    for name, accs in dictionary_of_method_points.items():
        # Get bin count from accuracies and accuracy ranges
        freqs = np.bincount(np.digitize(accs, acc_bins))/float(len(accs))
        print('accs: {}'.format(accs))
        print('freqs: {}'.format(freqs))
        plt.bar(acc_bins, freqs, width=5, color=colors[index], alpha=0.5, label=name, zorder=zorders[index])
        index += 1
    plt.legend(loc='upper left')
    plt.xlabel(r'Accuracy $(\%)$', fontsize=25)
    plt.ylabel('Frequency', fontsize=25)
    plt.xticks(np.arange(0, 100, step=10), fontsize=25)
    plt.yticks(fontsize=25)
    plt.savefig(filename)
    plt.show()


def format_func(value, position):
    value = int(np.round(value / 1e7))
    if value == 0:
        return "0"
    else:
        return r'' + str(value) + '$\cdot 10^{7}$'



# def scatter_plot_acc_vs_footprint(dictionary_of_method_points, dataset='nas101'):
#     fig = plt.figure()
#     axis = fig.add_subplot(111)
#     colors = {'nas101': 'b',
#               'nats': 'b',
#               'random': 'r',
#               'gnn2gnn': 'g',}
#     markers = {'nas101': 'o',
#               'nats': 'o',
#               'random': 'x',
#               'gnn2gnn': 's',}
#     alphas = {'nas101': 0.2,
#                'nats': 0.2,
#                'random': 1,
#                'gnn2gnn': 1, }
#     zorders = {'nas101': 1,
#               'nats': 1,
#               'random': 2,
#               'gnn2gnn': 3, }
#     for name, list_of_points in dictionary_of_method_points.items():
#         if dataset == 'nas101':
#             xs = [acc_vs_foot[1]*100 for acc_vs_foot in list_of_points]
#         elif dataset == 'nats':
#             xs = [acc_vs_foot[1] for acc_vs_foot in list_of_points]
#         ys = [acc_vs_foot[0] for acc_vs_foot in list_of_points]
#         axis.scatter(xs, ys, s=20, c=colors[name], marker=markers[name], alpha=alphas[name], label=name.upper(), zorder=zorders[name])
#     plt.legend(loc='lower right')
#     if dataset == 'nas101':
#         plt.xlabel('# Parameters')
#     elif dataset == 'nats':
#         plt.xlabel('Footprint (MB)')
#     plt.ylabel('Accuracy (%)')
#     plt.show()