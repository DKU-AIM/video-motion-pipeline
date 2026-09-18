import importlib.util
from pathlib import Path
import tempfile
import unittest
import zipfile

spec = importlib.util.spec_from_file_location('smpl_download', Path(__file__).with_name('download_smpl.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ExtractTests(unittest.TestCase):
    def test_extracts_only_named_neutral_model(self):
        with tempfile.TemporaryDirectory() as root:
            base=Path(root)
            archive=base/'source.zip'
            with zipfile.ZipFile(archive,'w') as z:
                z.writestr('models/'+module.MODEL_NAME, b'neutral-model-fixture')
                z.writestr('../../other.txt', b'do not extract')
            target=base/'private/SMPL_NEUTRAL.pkl'
            module.install_neutral(archive,target)
            self.assertEqual(target.read_bytes(), b'neutral-model-fixture')
            self.assertEqual(target.stat().st_mode & 0o777, 0o600)
            self.assertEqual(sorted(p.name for p in target.parent.iterdir()), ['SMPL_NEUTRAL.pkl'])

    def test_existing_model_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as root:
            base=Path(root)
            archive=base/'source.zip'
            with zipfile.ZipFile(archive,'w') as z:
                z.writestr(module.MODEL_NAME, b'replacement')
            target=base/'SMPL_NEUTRAL.pkl'; target.write_bytes(b'existing')
            with self.assertRaises(FileExistsError):
                module.install_neutral(archive,target)
            self.assertEqual(target.read_bytes(),b'existing')

    def test_missing_or_ambiguous_model_is_rejected(self):
        for names in (['other.pkl'], ['a/'+module.MODEL_NAME,'b/'+module.MODEL_NAME]):
            with self.subTest(names=names), tempfile.TemporaryDirectory() as root:
                archive=Path(root)/'source.zip'
                with zipfile.ZipFile(archive,'w') as z:
                    for name in names: z.writestr(name,b'fixture')
                with self.assertRaises(ValueError):
                    module.install_neutral(archive,Path(root)/'SMPL_NEUTRAL.pkl')


if __name__ == '__main__':
    unittest.main()
