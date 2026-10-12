import json
import os
import tempfile
import unittest

from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase
from tools.postprocess import deserialize_meta
from tools.preprocess import (raw_alpaca_cot_merge_add_meta,
                              raw_arxiv_to_jsonl, reformat_csv_nan_value,
                              reformat_jsonl_nan_value, serialize_meta)


class PreprocessWorkerErrorTest(DataJuicerTestCaseBase):
    """A file that fails in a worker process must fail the script, not leave
    its output missing or truncated behind a successful exit."""

    def setUp(self):
        super().setUp()
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.src_dir = os.path.join(self.tmp_dir.name, 'src')
        self.target_dir = os.path.join(self.tmp_dir.name, 'target')
        os.makedirs(self.src_dir)

    def tearDown(self):
        self.tmp_dir.cleanup()
        super().tearDown()

    def _write(self, name, content):
        path = os.path.join(self.src_dir, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            f.write(content)
        return path

    def _write_jsonl(self, name, rows):
        return self._write(name,
                           ''.join(json.dumps(row) + '\n' for row in rows))

    def test_serialize_meta(self):
        self._write_jsonl('a.jsonl', [{'text': 'a', 'x': 1}, {'x': 2}])
        with self.assertRaises(KeyError):
            serialize_meta.main(self.src_dir, self.target_dir)

    def test_deserialize_meta(self):
        self._write_jsonl('a.jsonl', [{
            'text': 'a',
            'source_info': '{"x": 1}'
        }, {
            'text': 'b'
        }])
        with self.assertRaises(KeyError):
            deserialize_meta.main(self.src_dir, self.target_dir)

    def test_reformat_jsonl_nan_value(self):
        self._write_jsonl('a.jsonl', [{'a': 1}, {'a': 'not a number'}])
        with self.assertRaises(Exception):
            reformat_jsonl_nan_value.main(self.src_dir, self.target_dir)

    def test_reformat_csv_nan_value(self):
        self._write('a.csv', 'a,b\n1,2\n3,4,5,6\n')
        with self.assertRaises(Exception):
            reformat_csv_nan_value.main(self.src_dir, self.target_dir)

    def test_raw_arxiv_to_jsonl(self):
        self._write('arXiv_src_0001_001.tar', 'not a tar archive')
        with self.assertRaises(Exception):
            raw_arxiv_to_jsonl.main(self.src_dir,
                                    self.target_dir,
                                    work_dir=os.path.join(
                                        self.tmp_dir.name, 'work'))

    def test_raw_alpaca_cot_merge_add_meta(self):
        self._write('Alpaca-CoT/Chain-of-Thought/a.json', '{"instruction": ')
        with self.assertRaises(Exception):
            raw_alpaca_cot_merge_add_meta.main(self.src_dir,
                                               self.target_dir,
                                               num_proc=1)


if __name__ == '__main__':
    unittest.main()
