import os
import shutil
import tempfile
import sacred

from . import utils


class Experiment(sacred.Experiment):

    _temp_dir: str
    _artifacts_dir: str

    def __init__(self, name, template=None, logger=None, runs_dir='runs', temp_dir='temp'):
        super().__init__(name)
        self._name = name  # `self.path` in the parent class
        self._runs_dir = runs_dir
        self._temp_dir = temp_dir
        self._artifacts_dir = None
        if logger is not None:
            self.logger = logger
        self.observers.append(
            sacred.observers.FileStorageObserver.create(runs_dir, template=template)
        )

    def run(self, *args):
        try:
            self._before_run()
            result = super().run(*args)
        finally:
            self._after_run()
        return result

    def _before_run(self):
        if not os.path.exists(self._temp_dir):
            os.makedirs(self._temp_dir)
        self._artifacts_dir = tempfile.mkdtemp(prefix='%s_%s_' % (utils.get_timestamp(), self.path),
                                               dir=self._temp_dir)

    def _after_run(self):
        self._add_artifacts(self._artifacts_dir)
        self._copy_artifacts(self._artifacts_dir, self.runs_dir)
        shutil.rmtree(self._artifacts_dir)
        self._artifacts_dir = None

    def _add_artifacts(self, source_dir):
        archive_file_path = shutil.make_archive(source_dir, 'gztar', source_dir)
        self.add_artifact(archive_file_path)
        os.remove(archive_file_path)

    def _copy_artifacts(self, source_dir, destination_dir):
        shutil.copytree(source_dir, os.path.join(destination_dir, os.path.basename(source_dir)))

    @property
    def runs_dir(self):
        return os.path.join(self._runs_dir, self.current_run._id)

    @property
    def artifacts_dir(self):
        return self._artifacts_dir
