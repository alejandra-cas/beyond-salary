import asyncio
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import validate_firm_size_cutoffs as validation


class FirmValidationReviewTests(unittest.TestCase):
    def test_posting_weights_do_not_change_2018_classification_and_zero_class_f1(self):
        firms = pd.DataFrame({'FIRM_POSTING_COUNT': [1, 5],
                              'ALL_POSTINGS': [100, 10], 'llm_label': ['sme', 'large']})
        scores = validation.score_cutoffs(firms, [1, 2], weight_column='ALL_POSTINGS').set_index('cutoff')
        self.assertEqual(scores.loc[1, 'sme_f1_post_weighted'], 0)
        self.assertAlmostEqual(scores.loc[1, 'combined_f1_post_weighted'], (20 / 120) / 2)
        self.assertEqual(scores.loc[2, 'accuracy_post_weighted'], 1)
        self.assertEqual(scores.loc[2, 'combined_f1_post_weighted'], 1)

    def test_web_review_requires_search_and_records_retrieved_sources(self):
        response = SimpleNamespace(
            output_text='{"label":"large","confidence":0.9,"reason":"National chain","source_urls":["https://example.org/about"],"evidence_period":"current"}',
            output=[SimpleNamespace(type='web_search_call', status='completed',
                                    action=SimpleNamespace(sources=[SimpleNamespace(url='https://example.org/about')]))],
        )
        client = MagicMock()
        client.responses.create = AsyncMock(return_value=response)
        result = asyncio.run(validation.classify_one_firm_async(
            pd.Series({'COMPANY': 1, 'COMPANY_NAME': 'A'}), 'gpt-6-luna', client, 0, web_search=True,
        ))
        request = client.responses.create.await_args.kwargs
        self.assertEqual(request['tools'], [{'type': 'web_search'}])
        self.assertEqual(request['tool_choice'], 'required')
        self.assertEqual(result['source_urls'], ['https://example.org/about'])
        self.assertEqual(result['search_calls'], 1)
        response.output = []
        with self.assertRaisesRegex(RuntimeError, 'without a completed web search'):
            asyncio.run(validation.classify_one_firm_async(
                pd.Series({'COMPANY': 1}), 'gpt-6-luna', client, 0, web_search=True,
            ))

    def test_web_mode_replaces_reviews_without_search(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / 'labels.csv'
            pd.DataFrame({'COMPANY': ['1'], 'llm_model': ['gpt-6-luna'],
                          'llm_label': ['large']}).to_csv(cache_path, index=False)
            review = AsyncMock(return_value={'label': 'sme', 'confidence': 0.8,
                                             'web_search': True, 'review_version': 'web_v1',
                                             'search_calls': 1, 'source_urls': ['https://example.org']})
            client = MagicMock()
            client.close = AsyncMock()
            with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}), \
                    patch.object(validation, 'AsyncOpenAI', return_value=client), \
                    patch.object(validation, 'classify_one_firm_async', review):
                result = asyncio.run(validation.classify_samples_with_llm_async(
                    pd.DataFrame({'COMPANY': [1]}), cache_path, 'gpt-6-luna',
                    'https://api.openai.com/v1', 0, 0, 1, web_search=True,
                ))
            review.assert_awaited_once()
            self.assertTrue(review.await_args.kwargs['web_search'])
            self.assertTrue(result.llm_web_search.all())
            self.assertEqual(result.llm_label.tolist(), ['sme'])

    def test_model_change_and_failed_reviews_are_not_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / 'labels.csv'
            pd.DataFrame({
                'COMPANY': ['1', '2', '3'],
                'llm_model': ['gpt-5.4-mini', 'gpt-6-luna', 'gpt-6-luna'],
                'llm_label': ['large', 'sme', 'N/A'],
            }).to_csv(cache_path, index=False)
            samples = pd.DataFrame({'COMPANY': [1, 2, 3], 'COMPANY_NAME': ['A', 'B', 'C']})
            client = MagicMock()
            client.close = AsyncMock()
            review = AsyncMock(return_value={'label': 'sme', 'confidence': 0.8})
            with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}), \
                    patch.object(validation, 'AsyncOpenAI', return_value=client), \
                    patch.object(validation, 'classify_one_firm_async', review):
                results = asyncio.run(validation.classify_samples_with_llm_async(
                    samples, cache_path, 'gpt-6-luna', 'https://api.openai.com/v1', 0, 0, 1,
                ))
            self.assertEqual(review.await_count, 2)
            self.assertEqual([call.args[0]['COMPANY'] for call in review.await_args_list], [1, 3])
            self.assertTrue(all(call.args[1] == 'gpt-6-luna' for call in review.await_args_list))
            self.assertTrue(results.llm_model.eq('gpt-6-luna').all())
            self.assertTrue(results.llm_label.eq('sme').all())


if __name__ == '__main__':
    unittest.main()
