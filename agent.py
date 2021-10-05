from torch_inputs import *
import numpy as np
from data_loader import Dataset

class AbstractAgent():
    def __init__(self, params):
        # CUDA for PyTorch
        use_cuda = torch.cuda.is_available()
        self._device = torch.device("cuda" if use_cuda else "cpu")

        self._train_data = Dataset(params, mode='train', device=self._device)
        self._test_data = Dataset(params, mode='test', device=self._device)
        self._valid_data = Dataset(params, mode='valid', device=self._device)

        self._net_par = {'i_dim': self._train_data.nfeatures,
                         'o_dim': 1,
                         'h_dim': params.latent_dim,
                         'n_layers': params.n_layers}

        self._nepochs = params.n_epochs
        self._batchsize = params.batch_size

        self._model = None
        self._optimizer = None
        self._loss = None

    def train(self):
        for epoch in range(self._nepochs):
            for (p, v, y) in self._train_data:
                y_pred = self.predict(p, v)
                loss = self.compute_loss(y_pred, y, p, v)
                self.propagate_loss(loss)
            self.print_report(epoch)
            # TODO
            #self.validation_step(epoch)

    def predict(self, p, v):
        return self._model(p, v)

    def propagate_loss(self, loss):
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

    def validation_step(self, epoch):
        acc = list()
        data = self._valid_data
        with torch.set_grad_enabled(False):
            for (x, y) in data:
                y_pred = self._model.predict(x).flatten()
                acc.append(np.sum(y_pred == y.numpy().flatten()) / len(y))
        print(f"epoch: {epoch}, validation accuracy: {np.mean(acc)}")

    def print_report(self, epoch):
        pass

    def test(self):
        acc, didi = list(), list()
        data = self._test_data
        with torch.set_grad_enabled(False):
            for (x, y) in data:
                y_pred = self._model.predict(x).flatten()
                acc.append(np.sum(y_pred == y.numpy().flatten()) / len(y))
        print(f"Test accuracy: {np.mean(acc)}")

    def compute_loss(self, y_pred, y, p=None, v=None):
        pass

    def plot(self):
        pass
