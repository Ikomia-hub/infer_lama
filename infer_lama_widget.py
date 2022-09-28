# Copyright (C) 2021 Ikomia SAS
# Contact: https://www.ikomia.com
#
# This file is part of the IkomiaStudio software.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# PyQt GUI framework
from PyQt5.QtWidgets import *

from ikomia import core, dataprocess
from ikomia.utils import pyqtutils, qtconversion
from torch.cuda import is_available
from infer_lama.infer_lama_process import InferLamaParam


# --------------------
# - Class which implements widget associated with the process
# - Inherits PyCore.CWorkflowTaskWidget from Ikomia API
# --------------------
class InferLamaWidget(core.CWorkflowTaskWidget):

    def __init__(self, param, parent):
        core.CWorkflowTaskWidget.__init__(self, parent)

        if param is None:
            self.parameters = InferLamaParam()
        else:
            self.parameters = param

        # Create layout : QGridLayout by default
        self.gridLayout = QGridLayout()

        # Set unpainting method
        self.combo_method = pyqtutils.append_combo(self.gridLayout, "Methods:")
        self.combo_method.addItem("default")
        self.combo_method.addItem("refine")
        self.combo_method.setCurrentText(self.parameters.method)
        self.combo_method.currentIndexChanged.connect(self.on_combo_dataset_changed)

        # Cuda
        self.check_cuda = pyqtutils.append_check(
                        self.gridLayout, "Cuda",
                        self.parameters.cuda and is_available())

        # Initial image resolution
        self.ini_resolution = pyqtutils.append_double_spin(
                            self.gridLayout, "Initial image resolution (%):",
                            self.parameters.ini_res,
                            min = 1., max = 100., step = 1, decimals = 1)

        # Number of iteration
        self.iterations = pyqtutils.append_double_spin(
                        self.gridLayout, "Iterations:", self.parameters.iter,
                        min = 1, max = 15, step = 1, decimals = 0)

        self.iterations.setVisible(False if self.parameters.method == "default" else True)

        # PyQt -> Qt wrapping
        layout_ptr = qtconversion.PyQtToQt(self.gridLayout) 

        # Set widget layout
        self.setLayout(layout_ptr)

        self.gridLayoutRefine = QGridLayout()

    def on_combo_dataset_changed(self,index):
        if self.combo_method.itemText(index) == "default":
            self.check_cuda.setVisible(True)
            self.ini_resolution.setVisible(True)
            self.iterations.setVisible(False)
        else:
            self.check_cuda.setVisible(True)
            self.ini_resolution.setVisible(True)
            self.iterations.setVisible(True)

    def onApply(self):
        # Apply button clicked slot
        self.parameters.update = True

        # Get parameters from widget
        self.parameters.method = self.combo_method.currentText()
        self.parameters.cuda = self.check_cuda.isChecked()
        self.parameters.ini_res = self.ini_resolution.value()
        self.parameters.iter = self.iterations.value()

        # Send signal to launch the process
        self.emitApply(self.parameters)


# --------------------
# - Factory class to build process widget object
# - Inherits PyDataProcess.CWidgetFactory from Ikomia API
# --------------------
class InferLamaWidgetFactory(dataprocess.CWidgetFactory):

    def __init__(self):
        dataprocess.CWidgetFactory.__init__(self)
        # Set the name of the process ->
        # it must be the same as the one declared in the process factory class
        self.name = "infer_lama"

    def create(self, param):
        # Create widget object
        return InferLamaWidget(param, None)