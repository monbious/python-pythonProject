import matplotlib.pyplot as plt
from IPython import display


mngr = plt.get_current_fig_manager()
mngr.window.wm_geometry("+50+80")  # 调整窗口在屏幕上弹出的位置
plt.ion()


def plot(scores, mean_scores):

    display.clear_output(wait=True)
    display.display(plt.gcf())

    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.ylim(ymin=0)
    plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))
    plt.pause(0.01)
    plt.cla()
    plt.plot(scores)
    plt.plot(mean_scores)
