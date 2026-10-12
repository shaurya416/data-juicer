import json
import os
import tempfile
import unittest
from unittest.mock import patch

from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase
from tools.postprocess.data_mixture import run_mixture


class DataMixtureTest(DataJuicerTestCaseBase):

    def setUp(self):
        super().setUp()
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.paths = {}
        for name in ('a', 'b'):
            path = os.path.join(self.tmp_dir.name, f'{name}.jsonl')
            with open(path, 'w') as f:
                for i in range(50):
                    f.write(json.dumps({'text': f'{name} {i}'}) + '\n')
            self.paths[name] = path
        self.export_path = os.path.join(self.tmp_dir.name, 'mixed.jsonl')

    def tearDown(self):
        self.tmp_dir.cleanup()
        super().tearDown()

    def _run(self, *extra_args):
        argv = [
            'data_mixture.py', '--data_path', '0.8', self.paths['a'], '0.2',
            self.paths['b'], '--export_path', self.export_path,
            '--num_proc', '1', *extra_args
        ]
        with patch('sys.argv', argv):
            run_mixture()
        with open(self.export_path) as f:
            texts = [json.loads(line)['text'] for line in f]
        return [text.split()[0] for text in texts]

    def test_max_samples_mixes_by_weight(self):
        sources = self._run('--max_samples', '10')
        self.assertEqual(len(sources), 10)
        self.assertEqual(sources.count('a'), 8)
        self.assertEqual(sources.count('b'), 2)

    def test_without_max_samples_keeps_all_samples(self):
        sources = self._run()
        self.assertEqual(len(sources), 100)


if __name__ == '__main__':
    unittest.main()
