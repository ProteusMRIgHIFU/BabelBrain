# This Python file uses the following encoding: utf-8
import os
import platform
import re
import sys
from pathlib import Path

import yaml
from PySide6.QtCore import QAbstractTableModel, Qt, Slot
from PySide6.QtWidgets import (QApplication, QDialog, QFileDialog,
                               QMessageBox, QStyle, QWidget)
import Localization
from Localization import TR
from BuildInfo import TitleSuffix
from CreateTransducers.transducer_creator import (CustomTransducer,
                                                  get_class_name)
from GUIComponents.custom_transducer_dialog import (CUSTOM_TRANSDUCER_OPTION,
                                                    CUSTOM_TRANSDUCER_PREFIX,
                                                    CustomTransducerDialog,
                                                    custom_transducer_display_name)
from GUIComponents.custom_transducer_manager_dialog import CustomTransducerManagerDialog
import RemoteServers
from TranscranialModeling.babel_integration.babel_integration_helpers import SpeedofSoundWebbDataset
# Important:
# You need to run the following command to generate the ui_form.py file
#     pyside6-uic form.ui -o ui_form.py, or
#     pyside2-uic form.ui -o ui_form.py
from .ui_form import Ui_SelFilesDialog
from Utils.paths import bundle_root, custom_transducers_root
from Utils.transducer_registry import DEFAULT_TRANSDUCERS


def show_error_dialog(
    parent: QWidget | None,
    error: Exception | str,
    title: str = "Error",
) -> None:
    message_box = QMessageBox(parent)
    message_box.setIcon(QMessageBox.Icon.Critical)
    message_box.setWindowTitle(title)
    message_box.setText(f"Unable to create transducer\n\n{error}")
    message_box.setStandardButtons(QMessageBox.StandardButton.Ok)
    message_box.exec()


class TableModel(QAbstractTableModel):
    def __init__(self, data):
        super(TableModel, self).__init__()
        self._data = data

    def data(self, index, role):
        if role == Qt.DisplayRole:
            value = self._data.iloc[index.row(), index.column()]
            return str(value)
        elif role == Qt.ItemDataRole.TextAlignmentRole:
            return Qt.AlignmentFlag.AlignCenter

    def rowCount(self, index):
        return self._data.shape[0]

    def columnCount(self, index):
        return self._data.shape[1]

    def headerData(self, section, orientation, role):
        # section is the index of the column/row.
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._data.columns[section])

            if orientation == Qt.Vertical:
                return str(self._data.index[section])


ORIGINAL_BABELBRAIN_SELECTION = {'real CT': 19, 'ZTE': 19, 'PETRA': 19}


def ValidThermalProfile(fProf):
    msgDetails = ''
    try:
        with open(fProf, 'r') as f:
            profile = yaml.safe_load(f)
    except:
        msgDetails = "Invalid profile YAML file"
        return False, msgDetails

    if 'BaseIsppa' not in profile:
        msgDetails = "BaseIsppa entry must be in YAML file"
        return False, msgDetails

    if type(profile['BaseIsppa']) is not float:
        msgDetails = "BaseIsppa must be a single float"
        return False, msgDetails

    if 'AllDC_PRF_Duration' not in profile:
        msgDetails = "AllDC_PRF_Duration entry must be in YAML file"
        return False, msgDetails

    if type(profile['AllDC_PRF_Duration']) is not list:
        msgDetails = "AllDC_PRF_Duration must be a list"
        return False, msgDetails

    for n, entry in enumerate(profile['AllDC_PRF_Duration']):
        if type(entry) is not dict:
            msgDetails = "entry %i in AllDC_PRF_Duration must be a dictionary" % (n)
            return False, msgDetails
        for k in ['DC', 'PRF', 'Duration', 'DurationOff']:
            if k not in entry:
                msgDetails = "entry %i in AllDC_PRF_Duration must have a key %s" % (n, k)
                return False, msgDetails
            if type(entry[k]) is not float:
                msgDetails = "key %s in entry %i of AllDC_PRF_Duration must be float" % (k, n)
                return False, msgDetails
        if 'Repetitions' in entry:
            if type(entry['Repetitions']) is not int:
                msgDetails = "key Repetitions in entry %i of AllDC_PRF_Duration must be integer" % (n)
                return False, msgDetails
            if entry['Repetitions'] < 1:
                msgDetails = "key Repetitions in entry %i of AllDC_PRF_Duration must be larger or equal than 1" % (n)
                return False, msgDetails
        if 'NumberGroupedSonications' in entry:
            if type(entry['NumberGroupedSonications']) is not int:
                msgDetails = "key NumberGroupedSonications in entry %i of AllDC_PRF_Duration must be integer" % (n)
                return False, msgDetails
            if entry['NumberGroupedSonications'] < 1:
                msgDetails = "key NumberGroupedSonications in entry %i of AllDC_PRF_Duration must be larger than 1" % (n)
                return False, msgDetails
            if 'PauseBetweenGroupedSonications' not in entry:
                msgDetails = "key PauseBetweenGroupedSonications in entry %i of AllDC_PRF_Duration must be present if NumberGroupedSonications is specified" % (n)
                return False, msgDetails
        if 'PauseBetweenGroupedSonications' in entry:
            if type(entry['PauseBetweenGroupedSonications']) is not float:
                msgDetails = "key PauseBetweenGroupedSonications in entry %i of AllDC_PRF_Duration must be float" % (n)
                return False, msgDetails
            if entry['PauseBetweenGroupedSonications'] < 0.0:
                msgDetails = "key PauseBetweenGroupedSonications in entry %i of AllDC_PRF_Duration must be larger than 0.0" % (n)
                return False, msgDetails
            if 'NumberGroupedSonications' not in entry:
                msgDetails = "key NumberGroupedSonications in entry %i of AllDC_PRF_Duration must be present if PauseBetweenGroupedSonications is specified" % (n)
                return False, msgDetails
        for k in entry:
            if k not in ['DC', 'PRF', 'Duration', 'DurationOff', 'Repetitions', 'NumberGroupedSonications', 'PauseBetweenGroupedSonications']:
                msgDetails = "key %s in entry %i of AllDC_PRF_Duration is unknown. It must be either 'DC', 'PRF', 'Duration',  'DurationOff', 'Repetitions', 'NumberGroupedSonications' or 'PauseBetweenGroupedSonications'" % (k, n)
                return False, msgDetails
    return True, msgDetails


class SelFiles(QDialog):
    def __init__(self, parent=None,Trajectory='',T1W='',
                    SimbNIBS='',CTType=0,CoregCT=1,CT='',
                    SimbNIBSType=0,TrajectoryType=0,
                    GPU='CPU',
                    Backend='Metal',
                    defaultCTMap=ORIGINAL_BABELBRAIN_SELECTION['real CT'],
                    bTVUS_OPERATION=False):
        super().__init__(parent)
        self.ui = Ui_SelFilesDialog()
        self.ui.setupUi(self)

        self._PopulateTransducerComboBox()   # populate from transducer_list.yaml
        self.AddCustomTransducersToList()  # Add saved custom transducers
        # Apply the shared compact app style on top of the .ui layout.
        from GUIComponents.AppStyle import app_qss, apply_native_spinbox_style
        self.setStyleSheet(app_qss(self))
        apply_native_spinbox_style(self)  # Windows: compact stacked spin arrows
        self._UpdateTransducerPopupWidth()  # re-measure with the styled font
        with open(os.path.join(bundle_root(__file__), 'version-gui.txt'), 'r') as f:
            version = f.readlines()[0]
        self.bb_version = version.strip()
        # This is the first screen users see, so a dev/test build has to say so
        # here too - same annotation as the main window. Empty for source runs
        # and stable releases. rstrip() because readlines()[0] keeps the file's
        # trailing newline, which would otherwise sit in the middle of the title.
        self._bTVUS_OPERATION = bTVUS_OPERATION
        self._version = version.rstrip()
        self._ApplyWindowTitle()
        self._BuildLanguagePicker()
        self.ui.SelTrajectorypushButton.clicked.connect(self.SelectTrajectory)
        self.ui.SelT1WpushButton.clicked.connect(self.SelectT1W)
        self.ui.SelCTpushButton.clicked.connect(self.SelectCT)
        self.ui.SelSimbNIBSpushButton.clicked.connect(self.SelectSimbNIBS)
        self.ui.SelTProfilepushButton.clicked.connect(self.SelectThermalProfile)
        self.ui.ContinuepushButton.clicked.connect(self.Continue)
        self.ui.CTTypecomboBox.currentIndexChanged.connect(self.SelectCTType)
        self.ui.MultiPointTypecomboBox.currentIndexChanged.connect(self.SelectMultiPoint)
        self.ui.TransducerTypecomboBox.currentIndexChanged.connect(self.SelectTransducer)
        self.ui.SelMultiPointProfilepushButton.clicked.connect(self.SelectMultiPointProfile)
        self.ui.CancelpushButton.clicked.connect(self.Cancel)

        self.ui.SelTrajectorypushButton.setIcon(self.style().standardIcon(QStyle.SP_FileIcon))
        self.ui.SelT1WpushButton.setIcon(self.style().standardIcon(QStyle.SP_FileIcon))
        self.ui.SelCTpushButton.setIcon(self.style().standardIcon(QStyle.SP_FileIcon))
        self.ui.SelTProfilepushButton.setIcon(self.style().standardIcon(QStyle.SP_FileIcon))
        self.ui.SelSimbNIBSpushButton.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))

        if len(Trajectory) > 0:
            self.ui.TrajectorylineEdit.setText(Trajectory)
            self.ui.TrajectorylineEdit.setCursorPosition(len(Trajectory))
        if len(T1W) > 0:
            self.ui.T1WlineEdit.setText(T1W)
            self.ui.T1WlineEdit.setCursorPosition(len(T1))
        if len(SimbNIBS) > 0:
            self.ui.SimbNIBSlineEdit.setText(SelectSimbNIBS)
            self.ui.SimbNIBSlineEdit.setCursorPosition(len(SelectSimbNIBS))
        if len(CT) > 0:
            self.ui.CTlineEdit.setText(CT)
            self.ui.CTlineEdit.setCursorPosition(len(CT))
        self.ui.CTTypecomboBox.setCurrentIndex(CTType)
        self.ui.SimbNIBSTypecomboBox.setCurrentIndex(SimbNIBSType)
        self.ui.TrajectoryTypecomboBox.setCurrentIndex(TrajectoryType)
        self.ui.CoregCTcomboBox.setCurrentIndex(CoregCT)
        self.ui.ResetCTMapOriginalpushButton.clicked.connect(self.ResetOriginalCTCombo)

        self._previous_transducer_index = self.ui.TransducerTypecomboBox.currentIndex()
        self.custom_transducer_config = ''

        self._GPUs = self.GetAvailableGPUs()

        # The computing-engine dropdown lists local GPUs, any saved remote
        # servers, and an "Add / remove remote server…" action. A machine with no
        # GPU is no longer a dead end: it can offload work to a remote BabelBrain
        # server (see RemoteServers.py / server.py).
        self.ui.ComputingEnginecomboBox.activated.connect(self.OnComputeEngineActivated)
        self._PopulateComputeEngines(GPU=GPU, Backend=Backend)

        if len(self._GPUs) == 0 and not any(it['kind'] == 'remote' for it in self._computeItems):
            msgBox = QMessageBox()
            msgBox.setText(TR("No GPUs were detected on this machine.\n\nTo run simulations, "
                           "add a remote BabelBrain server via the computing-engine "
                           "dropdown ('Add / remove remote server…')."))
            msgBox.exec()

        df = SpeedofSoundWebbDataset()
        for index, row in df.iterrows():
            self.ui.CTMappingcomboBox.addItem(', '.join(index))
        self._dfCTParams = df
        self.ui.CTMappingcomboBox.setCurrentIndex(defaultCTMap)

        self.setWindowFlags(self.windowFlags() | Qt.CustomizeWindowHint)
        # disable (but not hide) close button
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowCloseButtonHint)

    def _ApplyWindowTitle(self):
        """(Re)build the window title. Called again after a language change,
        because the title is assembled here rather than by retranslateUi()."""
        # This is the first screen users see, so a dev/test build has to say so
        # here too - same annotation as the main window. Empty for source runs
        # and stable releases.
        if self._bTVUS_OPERATION:
            root_title = 'BabelBrain-TVUS mode'
        else:
            root_title = 'BabelBrain'
        self.setWindowTitle(root_title + " V" + self._version + TitleSuffix() +
                            TR(" - Select input files ..."))

    def _BuildLanguagePicker(self):
        """Language selector, bottom-left of the dialog.

        This screen is the right home for it: everything else in the app - the
        main window, the transducer and thermal panels, Advanced Options, the
        viewer - is constructed only after this dialog closes, so a change here
        applies to the whole session with no restart. This dialog itself is
        refreshed in place by retranslateUi().

        Added programmatically (the .ui uses absolute geometry, so there is no
        layout to fit into) to keep form.ui and its generated module untouched.
        """
        self.LanguageLabel = QLabel(TR("Language"), self)
        self.LanguageLabel.setObjectName("LanguageLabel")
        self.LanguageLabel.setGeometry(QRect(12, 396, 74, 22))

        self.LanguagecomboBox = QComboBox(self)
        self.LanguagecomboBox.setObjectName("LanguagecomboBox")
        self.LanguagecomboBox.setGeometry(QRect(88, 392, 180, 30))
        # The language code lives in the item DATA. The visible names are
        # localized; the codes must never be, and nothing may look them up by
        # display text.
        self.LanguagecomboBox.addItem(TR("Follow system language"),
                                      Localization.LANG_SYSTEM)
        for code, name in Localization.available_languages():
            self.LanguagecomboBox.addItem(name, code)
        idx = self.LanguagecomboBox.findData(Localization.get_ui_language())
        self.LanguagecomboBox.setCurrentIndex(idx if idx >= 0 else 1)
        self.LanguagecomboBox.currentIndexChanged.connect(self.SelectLanguage)

    def _SnapshotFormState(self):
        """Everything the user has entered or chosen in this dialog.

        retranslateUi() re-applies every property the .ui file declares, and in
        this form that includes the *text* of the path fields - which is the
        "..." placeholder. Left alone it would wipe the paths the user just
        picked, so the values are saved across the call and put back.
        """
        state = {'text': {}, 'index': {}, 'checked': {}, 'value': {}}
        for w in self.findChildren(QLineEdit):
            state['text'][w.objectName()] = w.text()
        for w in self.findChildren(QComboBox):
            state['index'][w.objectName()] = w.currentIndex()
        for w in self.findChildren(QCheckBox):
            state['checked'][w.objectName()] = w.isChecked()
        for w in self.findChildren(QAbstractSpinBox):
            if hasattr(w, 'value'):
                state['value'][w.objectName()] = w.value()
        return state

    def _RestoreFormState(self, state):
        """Put back what _SnapshotFormState() captured.

        Signals are blocked while restoring: the values are identical to what
        the user had, so the handlers have nothing to react to, and firing
        SelectCTType/SelectMultiPoint here would re-run their side effects.
        """
        groups = ((QLineEdit, 'text', 'setText'),
                  (QComboBox, 'index', 'setCurrentIndex'),
                  (QCheckBox, 'checked', 'setChecked'),
                  (QAbstractSpinBox, 'value', 'setValue'))
        for cls, key, setter in groups:
            for w in self.findChildren(cls):
                name = w.objectName()
                if name not in state[key] or not hasattr(w, setter):
                    continue
                blocked = w.blockSignals(True)
                getattr(w, setter)(state[key][name])
                w.blockSignals(blocked)

    @Slot()
    def SelectLanguage(self):
        """Switch language immediately and remember the choice.

        Installing a translator does not rewrite widgets that already exist, so
        this dialog is re-translated explicitly; every other window is built
        later and picks up the new catalogue on its own.
        """
        code = self.LanguagecomboBox.currentData()
        if not code:
            return
        Localization.set_ui_language(code)
        Localization.install_translators(
            QApplication.instance(),
            Localization.resolve_language(code),
            Localization.mode_for_config(self._bTVUS_OPERATION))

        state = self._SnapshotFormState()
        self.ui.retranslateUi(self)
        self._RestoreFormState(state)

        # Not part of the form, so retranslateUi() does not cover these.
        self._ApplyWindowTitle()
        self.LanguageLabel.setText(TR("Language"))
        self.LanguagecomboBox.setItemText(0, TR("Follow system language"))
    def _PopulateTransducerComboBox(self):
        """Fill TransducerTypecomboBox from DEFAULT_TRANSDUCERS, storing each
        entry's dict as item data so callers can query it via currentData()."""
        combo = self.ui.TransducerTypecomboBox
        combo.blockSignals(True)
        combo.clear()
        for tx in DEFAULT_TRANSDUCERS:
            combo.addItem(tx['name'], tx)
        combo.addItem(CUSTOM_TRANSDUCER_OPTION, None)
        combo.blockSignals(False)

    def GetAllTransducers(self):
        """
        Returns a list of all transducers available in BabelBrain
        """
        return [
            self.ui.TransducerTypecomboBox.itemText(i).replace(CUSTOM_TRANSDUCER_PREFIX,"")
            for i in range(self.ui.TransducerTypecomboBox.count())
            if self.ui.TransducerTypecomboBox.itemText(i) != CUSTOM_TRANSDUCER_OPTION
        ]

    @Slot()
    def ManageCustomTransducers(self):
        CustomTransducerManagerDialog(self).exec()

    def CreateCustomTransducer(self):
        """
        Open the custom transducer dialog and create the transducer. Returns the
        new transducer's combobox text, or None if the user cancelled.
        """
        dialog = CustomTransducerDialog(self, config_file=self.custom_transducer_config)
        temp_tx = None

        while True:
            result = dialog.exec()

            # User cancelled or closed the dialog
            if result != QDialog.DialogCode.Accepted:
                break

            self.custom_transducer_config = dialog.config_file

            try:
                gpu, computing_backend = self.GetSelectedComputingEngine()
                self.remote_server = self.GetSelectedServer()
                temp_tx = CustomTransducer(
                    bb_version=self.bb_version,
                    transducer_yaml=self.custom_transducer_config,
                    computing_backend=computing_backend,
                    gpu=gpu,
                    remote_server=self.remote_server)

            except Exception as error:
                if "Cancel Action" not in str(error):
                    show_error_dialog(
                        self,
                        error,
                        "Unable to create transducer",
                    )

                # Reopen the same dialog as though Accept had not succeeded.
                continue
            finally:
                # Refresh transducer list.
                self.AddCustomTransducersToList()

            # Transducer was created successfully.
            break

        if temp_tx is None:
            return None

        new_tx_name = CUSTOM_TRANSDUCER_PREFIX + get_class_name(temp_tx.name)
        new_tx_index = self.ui.TransducerTypecomboBox.findText(new_tx_name)
        if new_tx_index >= 0:
            self.ui.TransducerTypecomboBox.setCurrentIndex(new_tx_index)
            return new_tx_name
        return None

    def _custom_tx_item_data(self, tx_name: str) -> dict:
        """Build the item-data dict for a custom transducer by reading its default.yaml."""
        tx_default_yaml = custom_transducers_root() / f"babel_{tx_name}" / "default.yaml"
        try:
            with open(tx_default_yaml, "r") as f:
                tx_params = yaml.safe_load(f)
            geometry_type = tx_params.get('geometry_type', '')
            steering_axes = tx_params.get('steering_axes', [])
            steering = len(steering_axes) == 3
        except Exception:
            geometry_type = ''
            steering = False
        return {
            'name': tx_name,
            'module_name': tx_name,
            'custom': True,
            'steering': steering,
            'transducer_type': geometry_type,
        }

    def AddCustomTransducersToList(self):
        """
        Look for saved custom transducers in .config and add them to list of all transducers available in BabelBrain
        """

        # When custom transducers are added to list, item indices change resulting in SelectTransducer being called again
        # and opeing another create transducer dialog. We block signals from transducer combobox here to prevent this
        self.ui.TransducerTypecomboBox.blockSignals(True) 

        # Define the transducers folder path if not already created. In server
        # mode this is the staging folder inside the job's workspace, not the
        # user's .config -- see Utils.paths.custom_transducers_root.
        transducers_folder = custom_transducers_root()
        if not os.path.exists(transducers_folder):
            # Create the directory safely
            transducers_folder.mkdir(parents=True, exist_ok=True)

        try:
            valid_custom_txs = set()

            # Loop through each custom transducer and add to list
            tx_folders = [f.name for f in Path(transducers_folder).iterdir() if f.is_dir()]
            for tx_folder in tx_folders:
                tx_folder_found = re.search("(?<=babel_).*", str(tx_folder))
                if tx_folder_found:
                    tx_name = tx_folder_found[0]

                    item_text = custom_transducer_display_name(tx_name)
                    existing_index = self.ui.TransducerTypecomboBox.findText(item_text)
                    valid_custom_txs.add(item_text)

                    if existing_index < 0:
                        tx_data = self._custom_tx_item_data(tx_name)
                        insert_index = self.ui.TransducerTypecomboBox.count() - 1 # Accounting for 'Add Custom Transducer' option
                        self.ui.TransducerTypecomboBox.insertItem(insert_index, item_text, tx_data)

            # Delete custom transducers from list that no longer have files
            for index in range(self.ui.TransducerTypecomboBox.count() - 1, -1, -1):
                item_text = self.ui.TransducerTypecomboBox.itemText(index)

                # Adjust this condition if custom entries use another identifier.
                is_custom_transducer = item_text.startswith(CUSTOM_TRANSDUCER_PREFIX)

                if is_custom_transducer and item_text not in valid_custom_txs:
                    self.ui.TransducerTypecomboBox.removeItem(index)
        finally:
            self.ui.TransducerTypecomboBox.blockSignals(False)

        self._UpdateTransducerPopupWidth()

    def _UpdateTransducerPopupWidth(self):
        """Widen the transducer dropdown's popup list to fit its longest entry,
        without changing the width of the combobox itself."""
        # Measure the text directly: sizeHintForColumn() is unreliable until the
        # popup has been shown, so it gave too narrow a width at startup.
        combo = self.ui.TransducerTypecomboBox
        view = combo.view()
        view.ensurePolished()
        metrics = view.fontMetrics()
        text_width = max((metrics.horizontalAdvance(combo.itemText(i))
                          for i in range(combo.count())), default=0)
        view.setMinimumWidth(text_width +
                             view.verticalScrollBar().sizeHint().width() + 30)

    # ── Computing-engine dropdown (local GPUs + remote servers) ──────────────
    def _engineKey(self, it):
        """A stable identity for a combo row, used to reselect after a repopulate."""
        if it['kind'] == 'gpu':
            return ('gpu', it['device'], it['backend'])
        if it['kind'] == 'remote':
            return ('remote', it['server']['name'])
        return ('action',)

    def _PopulateComputeEngines(self, GPU='CPU', Backend=''):
        """(Re)build the dropdown: local GPUs, saved remote servers, then the
        add/remove action. Selects the requested engine when possible."""
        combo = self.ui.ComputingEnginecomboBox
        combo.blockSignals(True)
        combo.clear()
        self._computeItems = []
        for dev in self._GPUs:
            self._computeItems.append({'kind': 'gpu', 'device': dev[0],
                                       'backend': dev[1],
                                       'label': dev[0] + ' -- ' + dev[1]})
        for srv in RemoteServers.load_servers():
            self._computeItems.append({'kind': 'remote', 'server': srv,
                                       'label': 'Remote: %s (%s:%d)'
                                       % (srv['name'], srv['host'], srv['port'])})
        self._computeItems.append({'kind': 'action',
                                   'label': '➕  Add / remove remote server…'})
        for it in self._computeItems:
            combo.addItem(it['label'])
        combo.blockSignals(False)

        target = None
        if Backend == 'Server':
            name = GPU[len('Remote: '):] if GPU.startswith('Remote: ') else GPU
            target = next((i for i, it in enumerate(self._computeItems)
                           if it['kind'] == 'remote' and it['server']['name'] == name), None)
        if target is None:
            target = next((i for i, it in enumerate(self._computeItems)
                           if it['kind'] == 'gpu' and GPU in it['device']
                           and (GPU == 'CPU' or Backend in it['backend'])), None)
        if target is None:                       # fall back to the first real engine
            target = next((i for i, it in enumerate(self._computeItems)
                           if it['kind'] in ('gpu', 'remote')), 0)
        combo.setCurrentIndex(target)
        self._prevEngineKey = self._engineKey(self._computeItems[target])

    def OnComputeEngineActivated(self, index):
        """Open the server manager when the action row is chosen; otherwise just
        remember the pick so we can restore it after managing servers."""
        items = self._computeItems
        if not (0 <= index < len(items)):
            return
        if items[index]['kind'] != 'action':
            self._prevEngineKey = self._engineKey(items[index])
            return
        self.ManageRemoteServers()

    def ManageRemoteServers(self):
        """Open the remote-server manager, then rebuild the computing-engine
        dropdown, keeping the previously selected engine when it still exists."""
        from GUIComponents.RemoteServerDialog import RemoteServerManagerDialog
        RemoteServerManagerDialog(self).exec()
        prev = getattr(self, '_prevEngineKey', None)
        self._PopulateComputeEngines()
        for i, it in enumerate(self._computeItems):     # restore previous choice
            if prev is not None and self._engineKey(it) == prev:
                self.ui.ComputingEnginecomboBox.setCurrentIndex(i)
                return
        for i, it in enumerate(self._computeItems):     # else first real engine
            if it['kind'] in ('gpu', 'remote'):
                self.ui.ComputingEnginecomboBox.setCurrentIndex(i)
                return

    def SelectComputingEngine(self, GPU='CPU', Backend=''):
        for sel, it in enumerate(self._computeItems):
            if it['kind'] == 'gpu' and GPU in it['device'] and (GPU == 'CPU' or Backend in it['backend']):
                self.ui.ComputingEnginecomboBox.setCurrentIndex(sel)
                return
            if it['kind'] == 'remote' and Backend == 'Server':
                name = GPU[len('Remote: '):] if GPU.startswith('Remote: ') else GPU
                if it['server']['name'] == name:
                    self.ui.ComputingEnginecomboBox.setCurrentIndex(sel)
                    return

    def SelectTxSystem(self, TxSystem='CTX_500', is_custom_tx=False):
        if is_custom_tx:
            TxSystem = CUSTOM_TRANSDUCER_PREFIX + TxSystem
        index = self.ui.TransducerTypecomboBox.findText(TxSystem)
        if index >= 0:
            self.ui.TransducerTypecomboBox.setCurrentIndex(index)

    def _CurrentEngineItem(self):
        idx = self.ui.ComputingEnginecomboBox.currentIndex()
        if 0 <= idx < len(self._computeItems):
            return self._computeItems[idx]
        return None

    def GetSelectedComputingEngine(self):
        it = self._CurrentEngineItem()
        if it is None:
            return ['CPU', '']
        if it['kind'] == 'remote':
            return ['Remote: ' + it['server']['name'], 'Server']
        if it['kind'] == 'gpu':
            return [it['device'], it['backend']]
        return ['CPU', '']                       # action row (not a real engine)

    def GetSelectedServer(self):
        """The remote-server dict when a remote engine is selected, else None."""
        it = self._CurrentEngineItem()
        return it['server'] if it and it['kind'] == 'remote' else None

    def GetAvailableGPUs(self):
        AllDevices = []
        if 'Darwin' in platform.system():
            from BabelViscoFDTD.StaggeredFDTD_3D_With_Relaxation_METAL import ListDevices
            devices = ListDevices()
            print('Available Metal Devices', devices)
            for dev in devices:
                AllDevices.append([dev, 'Metal'])
                # AllDevices.append([dev,'MLX']) #we disable this for the time being until MLX fixes their support to large arrays
        else:
            # we try to import CUDA and OpenCL in Win/Linux systems, if it fails, it means some drivers are not correctly installed
            try:
                from BabelViscoFDTD.StaggeredFDTD_3D_With_Relaxation_CUDA import ListDevices
                devices = ListDevices()
                print('Available CUDA Devices', devices)
                for dev in devices:
                    AllDevices.append([dev, 'CUDA'])
            except:
                pass
            try:
                from BabelViscoFDTD.StaggeredFDTD_3D_With_Relaxation_OPENCL import ListDevices
                devices = ListDevices()
                print('Available OPENCL Devices', devices)
                for dev in devices:
                    AllDevices.append([dev, 'OpenCL'])
            except:
                pass
        return AllDevices

    def ValidateIndivTrajectory(self, fTraj):
        with open(fTraj) as f:
            lines = f.readlines()
        lines = str(lines).lower()
        if self.ui.TrajectoryTypecomboBox.currentIndex() == 0:  # Brainsight
            if re.search("brainsight", lines):
                return True
            else:
                self.msgDetails = "Selected trajectory file is not a Brainsight file"
                return False
        elif self.ui.TrajectoryTypecomboBox.currentIndex() == 1:  # Slicer
            # TODO: we need something better for this
            if re.search("(?<!bra)insight", lines):  # insight, but not brainsight in text
                return True
            else:
                self.msgDetails = "Selected trajectory file is not a Slicer file"
                return False
        else: # Localite
            import xml.etree.ElementTree as ET

            def is_valid_xml(path):
                try:
                    for _ in ET.iterparse(path):
                        pass
                    return True
                except ET.ParseError:
                    return False
            if is_valid_xml(fTraj):
                return True
            else:
                self.msgDetails = "Selected trajectory file is not a XML file"
                return False

    def ValidTrajectory(self):
        fTraj = self.ui.TrajectorylineEdit.text()

        if not os.path.isfile(fTraj):
            self.msgDetails = "Trajectory file was not specified"
            return False

        if os.path.splitext(fTraj)[1].lower() in ['.txt', '.xml']:
            return self.ValidateIndivTrajectory(fTraj)
        else:  # this is a yaml file for 3D Slicer trajectories
            try:
                with open(fTraj) as f:
                    trajectories = yaml.safe_load(f)
            except:
                self.msgDetails = "Unable to load YAML file for 3D Slicer trajectories"
                return False
            if type(trajectories) is not dict:
                print(type(trajectories), trajectories, fTraj)
                self.msgDetails = "3D Slicer trajectories file must be a simple dictionary\n" + \
                                  'with "key: path" pairs to individual linear transforms'
                return False
            for k in trajectories:
                if not os.path.isfile(trajectories[k]):
                    self.msgDetails = f"File path for trajectory {k} does not exist"
                    return False
                if not self.ValidateIndivTrajectory(trajectories[k]):
                    self.msgDetails = f"For trajectory {k}\n" + self.msgDetails
                    return False
                return True

    def ValidSimNIBS(self):
        folderSimNIBS = self.ui.SimbNIBSlineEdit.text()

        if not os.path.isdir(folderSimNIBS):
            self.msgDetails = "SimNIBS Directory was not specified"
            return False

        files = os.listdir(folderSimNIBS)
        files = str(files).lower()

        if self.ui.SimbNIBSTypecomboBox.currentIndex() == 0:  # Charm
            if "charm" in files:
                return True
            else:
                self.msgDetails = "Selected SimbNIBS folder was not Charm generated"
                return False
        else:  # Headreco
            if "headreco" in files:
                return True
            else:
                self.msgDetails = "Selected SimbNIBS folder was not Headreco generated"
                return False

    def ValidThermalProfile(self):
        fProf = self.ui.ThermalProfilelineEdit.text()
        retValue, self.msgDetails = ValidThermalProfile(fProf)
        return retValue

    def ValidateMultiPointProfile(self):
        tx_data = self.ui.TransducerTypecomboBox.currentData()
        if tx_data is None or not tx_data.get('steering', False):
            return True
        if self.ui.MultiPointTypecomboBox.currentIndex() == 0:
            return True

        fProf = self.ui.MultiPointlineEdit.text()

        if not os.path.isfile(fProf):
            self.msgDetails = "Profile file was not specified"
            return False

        try:
            with open(fProf, 'r') as f:
                profile = yaml.safe_load(f)
        except:
            self.msgDetails = "Invalid profile YAML file"
            return False
        if 'MultiPoint' not in profile:
            self.msgDetails = "YAML file missing 'MultiPoint' entry"
            return False
        tx_data = self.ui.TransducerTypecomboBox.currentData()
        if tx_data is None or not tx_data.get('steering', False):
            self.msgDetails = "MultiPoint in profile can only be specified with a phased array-type transducer"
            return False
        if type(profile['MultiPoint']) is not list:
            self.msgDetails = "MultiPoint must be a list" 
            return False
        for n, entry in enumerate(profile['MultiPoint']):
            if type(entry) is not dict:
                self.msgDetails = "entry %i in MultiPoint must be a dictionary" % (n)
                return False
            for k in ['X', 'Y', 'Z']:
                if k not in entry:
                    self.msgDetails = "entry %i in MultiPoint must have a key %s" % (n, k)
                    return False
                if type(entry[k]) is not float:
                    self.msgDetails = "key %s in entry %i of MultiPoint must be float" % (k, n)
                    return False
        return True
        # we convert to mm

    @Slot()
    def SelectTrajectory(self):
        curfile = self.ui.TrajectorylineEdit.text()
        bdir = os.path.dirname(curfile)
        if not os.path.isdir(bdir):
            bdir = os.getcwd()
        if self.ui.TrajectoryTypecomboBox.currentIndex() == 0:
            # brainsight
            file_extension = '*.txt'
        elif self.ui.TrajectoryTypecomboBox.currentIndex() == 1:
            # slicer
            file_extension = '*.txt *.yaml *.yml'
        else:
            # localite
            file_extension = '*.xml *.XML'
        fTraj = QFileDialog.getOpenFileName(self, "Select trajectory", bdir, f"Trajectory ({file_extension})")[0]
        if len(fTraj) > 0:
            self.ui.TrajectorylineEdit.setText(fTraj)
            self.ui.TrajectorylineEdit.setCursorPosition(len(fTraj))

    @Slot()
    def SelectT1W(self):
        curfile = self.ui.T1WlineEdit.text()
        bdir = os.path.dirname(curfile)
        if not os.path.isdir(bdir):
            bdir = os.getcwd()
        fT1W = QFileDialog.getOpenFileName(self, "Select T1W", bdir, "Nifti (*.nii *.nii.gz)")[0]
        if len(fT1W) > 0:
            self.ui.T1WlineEdit.setText(fT1W)
            self.ui.T1WlineEdit.setCursorPosition(len(fT1W))

    @Slot()
    def SelectCT(self):
        curfile = self.ui.CTlineEdit.text()
        bdir = os.path.dirname(curfile)
        if not os.path.isdir(bdir):
            bdir = os.getcwd()
        fCT = QFileDialog.getOpenFileName(self, "Select CT", bdir, "Nifti (*.nii *.nii.gz)")[0]
        if len(fCT) > 0:
            self.ui.CTlineEdit.setText(fCT)
            self.ui.CTlineEdit.setCursorPosition(len(fCT))

    @Slot()
    def SelectThermalProfile(self):
        curfile = self.ui.ThermalProfilelineEdit.text()
        bdir = os.path.dirname(curfile)
        if not os.path.isdir(bdir):
            bdir = os.getcwd()
        fThermalProfile = QFileDialog.getOpenFileName(self, "Select thermal profile", bdir, "yaml (*.yaml)")[0]
        if len(fThermalProfile) > 0:
            print('fThermalProfile', fThermalProfile)
            self.ui.ThermalProfilelineEdit.setText(fThermalProfile)

    @Slot()
    def SelectMultiPointProfile(self):
        curfile = self.ui.MultiPointlineEdit.text()
        bdir = os.path.dirname(curfile)
        if not os.path.isdir(bdir):
            bdir = os.getcwd()
        fMultiPointProfile = QFileDialog.getOpenFileName(self, "Select multi point profile", bdir, "yaml (*.yaml)")[0]
        if len(fMultiPointProfile) > 0:
            print('fMultiPointProfile', fMultiPointProfile)
            self.ui.MultiPointlineEdit.setText(fMultiPointProfile)

    @Slot()
    def SelectSimbNIBS(self):
        bdir = self.ui.SimbNIBSlineEdit.text()
        if not os.path.isdir(bdir):
            bdir = os.getcwd()
        fSimbNIBS = QFileDialog.getExistingDirectory(self, "Select SimbNIBS directory", bdir)
        if len(fSimbNIBS) > 0:
            self.ui.SimbNIBSlineEdit.setText(fSimbNIBS)
            self.ui.SimbNIBSlineEdit.setCursorPosition(len(fSimbNIBS))

    @Slot()
    def SelectCTType(self, value):
        bv = value > 0
        bvCTScanner = value > 0 and value != 4
        self.ui.CTlineEdit.setEnabled(bv)
        self.ui.SelCTpushButton.setEnabled(bv)
        self.ui.CoregCTlabel.setEnabled(bv)
        self.ui.CoregCTlabel_2.setEnabled(bv)
        self.ui.CoregCTlabel_3.setEnabled(bv)
        self.ui.CoregCTcomboBox.setEnabled(bv)
        self.ui.CTMappingcomboBox.setEnabled(bvCTScanner)
        self.ui.ResetCTMapOriginalpushButton.setEnabled(bvCTScanner)
        self.ResetOriginalCTCombo()

    @Slot()
    def SelectMultiPoint(self, value):
        bv = value > 0
        self.ui.MultiPointlineEdit.setEnabled(bv)
        self.ui.SelMultiPointProfilepushButton.setEnabled(bv)

    @Slot()
    def SelectTransducer(self, value):
        sel_tx = self.ui.TransducerTypecomboBox.currentText()

        # Open custom transducer manager if option is selected
        if sel_tx == CUSTOM_TRANSDUCER_OPTION:
            # Restore the previous selection before opening the manager so the
            # combobox never sits on the 'Add Custom Transducer' option while
            # the manager edits the list.
            combo = self.ui.TransducerTypecomboBox
            combo.blockSignals(True)
            if combo.itemText(self._previous_transducer_index) == CUSTOM_TRANSDUCER_OPTION:
                combo.setCurrentIndex(0)
            else:
                combo.setCurrentIndex(self._previous_transducer_index)
            combo.blockSignals(False)

            self.ManageCustomTransducers()

        tx_data = self.ui.TransducerTypecomboBox.currentData()
        steering_enabled = tx_data is not None and tx_data.get('steering', False)

        if not steering_enabled:
            self.ui.MultiPointTypecomboBox.setCurrentIndex(0)
        self.ui.MultiPointTypecomboBox.setEnabled(steering_enabled)

        self._previous_transducer_index = self.ui.TransducerTypecomboBox.currentIndex()

    @Slot()
    def ResetOriginalCTCombo(self):
        if self.ui.CTTypecomboBox.currentText() != 'NO':
            if self.ui.CTTypecomboBox.currentText() in ORIGINAL_BABELBRAIN_SELECTION:
                self.ui.CTMappingcomboBox.setCurrentIndex(ORIGINAL_BABELBRAIN_SELECTION[ self.ui.CTTypecomboBox.currentText()])

    @Slot()
    def Continue(self):
        self.msgDetails = ""
        if not self.ValidTrajectory() or\
           not self.ValidSimNIBS() or\
           not self.ValidThermalProfile() or\
           not self.ValidateMultiPointProfile() or\
           not os.path.isfile(self.ui.T1WlineEdit.text()) or\
           (self.ui.CTTypecomboBox.currentIndex() > 0 and not os.path.isfile(self.ui.CTlineEdit.text())):
            msgBox = QMessageBox()
            msgBox.setText(TR("Please indicate valid entries"))
            print(self.msgDetails)
            msgBox.setDetailedText(self.msgDetails)
            msgBox.exec()
        else:
            self.accept()

    @Slot()
    def Cancel(self):
        self.done(-1)


if __name__ == "__main__":

    app = QApplication(sys.argv)
    widget = SelFiles()
    widget.show()
    sys.exit(app.exec())
