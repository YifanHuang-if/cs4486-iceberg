"""Targeted tests for handoff alignment and the large-test-file streaming parser."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from src.finalize import stream_records, aligned_oof, individual_loss, metric_row


class FinalizationTests(unittest.TestCase):
    def test_stream_matches_json_with_tiny_chunks(self):
        records = [{'id': 'x"y', 'inc_angle': 'na', 'band_1': [1.2, -3e-4]},
                   {'id': 'second', 'nested': {'a': [1, 2, 3]}}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'sample.json'
            path.write_text(' \n'+json.dumps(records, indent=2)+' \n', encoding='utf-8-sig')
            self.assertEqual(list(stream_records(path, chunk_size=7)), records)

    def test_stream_rejects_truncated_input(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'sample.json'
            path.write_text('[{"id": "broken"', encoding='utf-8')
            with self.assertRaises(ValueError): list(stream_records(path, chunk_size=5))

    def test_oof_alignment_uses_ids_and_rejects_duplicates(self):
        mapping = pd.DataFrame({'id':['b','a'], 'fold':[0,1], 'label':[1,0], 'angle_missing':[0,1]})
        frame = mapping.copy();frame['probability'] = [.9,.2]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'oof.csv';frame.iloc[::-1].to_csv(path,index=False)
            actual=aligned_oof(path,mapping)
            np.testing.assert_array_equal(actual.probability,[.9,.2])
            pd.concat([frame,frame.iloc[:1]]).to_csv(path,index=False)
            with self.assertRaises(ValueError):aligned_oof(path,mapping)

    def test_log_loss_and_confident_errors(self):
        np.testing.assert_allclose(individual_loss([0,1],[.5,.5]),[-np.log(.5)]*2)
        frame=pd.DataFrame({'label':[0,1,0,1], 'probability':[.99,.01,.2,.8]})
        result=metric_row(frame)
        self.assertEqual(result['errors'],2)
        self.assertEqual(result['high_confidence_errors'],2)
        self.assertEqual(result['false_icebergs'],1)
        self.assertEqual(result['missed_icebergs'],1)


if __name__=='__main__':unittest.main()
