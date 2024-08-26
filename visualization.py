import numpy as np
import matplotlib.pyplot as plt

def opinion_trajectory_plot(
        trajectories, 
        timepoints,
        d, 
        V,
        global_section = None):
    
    n_cols = len(V) // 2 if len(V) % 2 == 0 else len(V) // 2 + 1
    fig, axs = plt.subplots(2, n_cols, figsize=(20, 10))

    topics = {
        i:chr(65 + i) for i in range(d)
    }

    # Determine the global minimum and maximum y-values
    y_min = float('inf')
    y_max = float('-inf')

    for i in V:
        for j in range(d):
            y_min = min(y_min, min(trajectories[:, i*d+j]))
            y_max = max(y_max, max(trajectories[:, i*d+j]))

    y_min -= 0.3
    y_max += 0.3

    for c, i in enumerate(V):
        ax = axs[c//n_cols, c % n_cols]
        for j in range(d):
            ax.plot(timepoints, trajectories[:, i*d+j], label=f'Topic {topics[j]}')

            if global_section is not None:
                if j == 0:
                    ax.axhline(y=global_section[i*d+j], 
                               color='r', 
                               linestyle='--', 
                               linewidth = 0.3, 
                               label='Projection of $x_0$ on sheaf laplacian null space')
                else:
                    ax.axhline(y=global_section[i*d+j], 
                               color='r', 
                               linewidth = 0.3, 
                               linestyle='--')
                    
            ax.set_title(f'Opinion dynamic of agent {i}')
        
        ax.set_ylim(y_min, y_max)  # Set the same y-axis limits for all subplots
    axs[-1, -1].axis('off')

    handles, labels = axs[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.85, 0.25))

    plt.show()