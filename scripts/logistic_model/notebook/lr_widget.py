from ipywidgets import VBox, Button, Layout, HTML
from matplotlib import pyplot as plt
from tensorflow.keras.utils import plot_model
from tqdm.notebook import tqdm

from scripts.logistic_model.model import LRModel


class LRWidget(VBox):

    @property
    def model(self) -> LRModel:
        return self._model

    def __init__(
            self,
            heading,
            **kwargs
    ):
        self._model = None
        self._title = HTML(
            '<h2>{0}</h2>'.format(heading)
        )
        self._start = Button(
            description='Train Model (loading...)',
            layout=Layout(width='auto'),
            button_style='success'
        )

        self._start.disable = True
        self._pbars = dict()
        self._children = [
            self._title,
            self._start
        ]

        super().__init__(
            children=[
                self._title,
                self._start
            ],
            layout=Layout(width='300px'),
            **kwargs
        )

    def on_model_loaded(self, model: LRModel):
        self._model = model

        self._model.add_observer(self._render_single_plot)
        self._model.add_observer(self._render_progress_bar)
        self._model.add_observer(self._update_progress_bar)

        self._start.on_click(self.on_start_clicked)
        self._start.disabled = False
        self._start.description = 'Train Model'

    def on_start_clicked(self, _):
        self._start.disabled = True
        self._start.description = 'Training...'
        self.model.start()

    def _render_multi_plot(self, errors_l1, errors_l2):
        x = self.model.lambdas
        plt.xscale('log', subs=range(1, 6))
        plt.xlabel('Regularization Strength')
        plt.ylabel('MSE')
        plt.plot(x, errors_l1, label='L1')
        plt.plot(x, errors_l2, label='L2')
        plt.legend()
        plt.show()

    def _render_single_plot(self, lambdas, errors):
        plt.title(f"{self.model.name} - {list(self._pbars.keys())[-1]} regularization")
        plt.xlabel('Regularization Strength')
        plt.ylabel('MSE')
        plt.xscale('log')
        plt.plot(lambdas, errors, linewidth=2)
        plt.show()

    def _render_progress_bar(self, reg_name, lambdas):
        self._pbars[reg_name] = tqdm(lambdas)

    def _update_progress_bar(self, reg_name, reg_val):
        self._pbars[reg_name].update()

    def __repr__(self):
        str_ = """Linear_Regression_Widget(
            model = {0}
        )""".format(self._title)
        return str_
