import os
import glob

import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def plot_policy_returns(
    env_name,
    dpmorl_output_dir,
    save_dir,
    batch_size=1,
    final_episodes=100
):
    """
    Plot the return distributions of the trained DPMORL policies.

    Parameters
    ----------
    env_name : str
        Environment name used for the plot title.

    dpmorl_output_dir : str
        Directory containing the DPMORL MORL*.npz files.

    save_dir : str
        Directory where the plot should be saved.

    batch_size : int
        Number of episodes to average into one plotted point.

    final_episodes : int
        Number of final episodes from each policy to plot.
    """

    os.makedirs(save_dir, exist_ok=True)

    file_paths = sorted(
        glob.glob(os.path.join(dpmorl_output_dir, "MORL*.npz"))
    )

    if len(file_paths) == 0:
        print(
            f"No MORL .npz files found in:\n"
            f"{dpmorl_output_dir}"
        )
        return

    print(
        f"\nPlotting return distributions for "
        f"{len(file_paths)} policies..."
    )

    colors = matplotlib.colormaps["gist_rainbow"](
        np.linspace(0, 1, len(file_paths))
    )

    markers = ["o", "v", "^", "s", "p"]

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")

    handles = []
    labels = []

    for policy_index, file_path in enumerate(file_paths):

        data = np.load(file_path)

        if "episode_vec_returns" not in data:
            print(
                f"Skipping {file_path}: "
                "'episode_vec_returns' not found."
            )
            continue

        episode_vec_returns = data["episode_vec_returns"]

        # We only need the first three objectives for this 3D plot.
        episode_vec_returns = episode_vec_returns[:, :3]

        # Take only the final episodes.
        final_returns = episode_vec_returns[
            -final_episodes * batch_size:
        ]

        # Average episodes according to batch size.
        episode_batches = [
            np.mean(
                final_returns[i:i + batch_size],
                axis=0
            )
            for i in range(
                0,
                len(final_returns),
                batch_size
            )
        ]

        episode_batches = np.asarray(episode_batches)

        if len(episode_batches) == 0:
            continue

        scatter = ax.scatter(
            episode_batches[:, 0],
            episode_batches[:, 1],
            episode_batches[:, 2],
            alpha=0.6,
            marker=markers[
                (policy_index // 10) % len(markers)
            ],
            color=colors[policy_index]
        )

        handles.append(scatter)
        labels.append(f"Policy {policy_index}")

    ax.set_title(
        f"{env_name} - Final Policy Return Distribution"
    )

    ax.set_xlabel("Return 1")
    ax.set_ylabel("Return 2")
    ax.set_zlabel("Return 3")

    ax.legend(
        handles,
        labels,
        title="Policies",
        loc="center left",
        bbox_to_anchor=(1.18, 0.5),
        borderaxespad=0.0,
        fontsize=10
    )

    output_path = os.path.join(
        save_dir,
        f"{env_name}_policy_returns.png"
    )

    plt.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Policy return plot saved to: {output_path}")