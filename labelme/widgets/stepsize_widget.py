from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets


class StepsizeWidget(QtWidgets.QSpinBox):

    def __init__(self, value=1):
        super(StepsizeWidget, self).__init__()
        self.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.setRange(1, 1000)
        self.setValue(value)
        self.setToolTip('Image Skip')
        self.setStatusTip(self.toolTip())
        self.setAlignment(QtCore.Qt.AlignCenter)

    def minimumSizeHint(self):
        height = super().minimumSizeHint().height()
        fm = QtGui.QFontMetrics(self.font())
        width = fm.width(str(self.maximum()))
        return QtCore.QSize(width, height)
