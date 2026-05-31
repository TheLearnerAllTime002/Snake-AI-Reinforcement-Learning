import atexit
import multiprocessing
from queue import Empty, Full


_plot_process = None
_plot_queue = None


def _draw_plot(axes, scores, mean_scores):
    axes.clear()
    axes.set_title('Training Progress')
    axes.set_xlabel('Number of Games')
    axes.set_ylabel('Score')
    axes.plot(scores, label='Score', color='#4f86f7')
    axes.plot(mean_scores, label='Mean Score', color='#e07a2f')
    axes.set_ylim(bottom=0)
    axes.text(len(scores) - 1, scores[-1], str(scores[-1]))
    axes.text(len(mean_scores) - 1, mean_scores[-1], f'{mean_scores[-1]:.2f}')
    axes.legend(loc='upper left')


def _run_plot_window(updates):
    import matplotlib.pyplot as plt

    plt.ion()
    figure, axes = plt.subplots(num='Snake AI Training')
    figure.canvas.manager.set_window_title('Snake AI Training Progress')
    figure.show()

    while plt.fignum_exists(figure.number):
        latest = None
        try:
            latest = updates.get(timeout=0.1)
            while True:
                latest = updates.get_nowait()
        except Empty:
            pass

        if latest is not None:
            scores, mean_scores = latest
            _draw_plot(axes, scores, mean_scores)
            figure.tight_layout()
            figure.canvas.draw_idle()

        plt.pause(0.05)

    plt.close(figure)


def _start_plot_window():
    global _plot_process, _plot_queue

    context = multiprocessing.get_context('spawn')
    _plot_queue = context.Queue(maxsize=1)
    _plot_process = context.Process(target=_run_plot_window, args=(_plot_queue,))
    _plot_process.daemon = True
    _plot_process.start()


def plot(scores, mean_scores):
    if not scores or not mean_scores:
        return

    if _plot_process is None or not _plot_process.is_alive():
        _start_plot_window()

    update = (list(scores), list(mean_scores))
    try:
        _plot_queue.put_nowait(update)
    except Full:
        try:
            _plot_queue.get_nowait()
        except Empty:
            pass
        _plot_queue.put_nowait(update)


def close_plot():
    global _plot_process, _plot_queue

    if _plot_process is not None and _plot_process.is_alive():
        _plot_process.terminate()
        _plot_process.join(timeout=1)

    _plot_process = None
    _plot_queue = None


atexit.register(close_plot)
