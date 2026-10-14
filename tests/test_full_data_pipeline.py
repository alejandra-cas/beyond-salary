import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from scripts import label_benefits
from scripts import full_data_pipeline as pipeline
from scripts.build_full_labeled_v2 import assemble_dataset, build_dataset


class FullDataPipelineTests(unittest.TestCase):
    def write_body(self, root, bodies=None):
        bodies = bodies or [
            "Tuition assistance and paid time off in a collaborative workplace.",
            "Mental health support and parental leave.",
            "This job is remote available to eligible applicants.",
        ]
        path = root / "body.csv"
        pd.DataFrame({"ID": [f"id-{index}" for index in range(len(bodies))], "BODY": bodies}).to_csv(path, index=False)
        return path

    def test_streamed_benefits_match_existing_keyword_functions_and_positions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_body(root)
            output = root / "benefits.parquet"
            pipeline.stream_labels(source, output, "benefits", batch_rows=1, workers=2)
            result = pd.read_parquet(output)
            source_frame = pd.read_csv(source)

            specifications = {name: (position, keywords, exclusions) for name, position, keywords, exclusions in pipeline.keyword_specs()}
            for name in pipeline.BENEFIT_NAMES:
                position, keywords, exclusions = specifications[name]
                expected = label_benefits.check_benefits(source_frame["BODY"], keywords, exclusions)
                np.testing.assert_array_equal(result[name].to_numpy(), expected)
                if position:
                    expected_position = label_benefits.check_benefit_position(source_frame["BODY"], keywords, exclusions)
                    np.testing.assert_allclose(
                        result[position].astype(float), expected_position.astype(float), equal_nan=True,
                    )
            metadata = json.loads(pipeline.metadata_path(output).read_text())
            self.assertTrue(metadata["complete"])
            self.assertEqual(metadata["rows"], 3)

    def test_remote_keyword_correction_is_explicit_and_changes_only_opted_in_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_body(root, ["This job is remote available to eligible applicants."])
            legacy = root / "remote-legacy.parquet"
            corrected = root / "remote-corrected.parquet"
            pipeline.stream_labels(source, legacy, "remote")
            pipeline.stream_labels(source, corrected, "remote", correct_remote_keywords=True)
            self.assertFalse(pd.read_parquet(legacy).loc[0, "REMOTE_KW"])
            self.assertTrue(pd.read_parquet(corrected).loc[0, "REMOTE_KW"])
            self.assertFalse(json.loads(pipeline.metadata_path(legacy).read_text())["correct_remote_keywords"])
            self.assertTrue(json.loads(pipeline.metadata_path(corrected).read_text())["correct_remote_keywords"])

    def test_checkpoint_resume_rejects_setting_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_body(root)
            output = root / "labels.parquet"
            original = pipeline.write_label_part
            calls = 0

            def fail_second(batch, path, kind, positions, correction):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("interrupted")
                return original(batch, path, kind, positions, correction)

            with patch.object(pipeline, "write_label_part", side_effect=fail_second):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    pipeline.stream_labels(source, output, "benefits", batch_rows=1)
            parts = output.with_name(output.name + ".parts")
            self.assertTrue((parts / "part-000000.parquet").exists())
            with self.assertRaisesRegex(ValueError, "settings changed"):
                pipeline.stream_labels(source, output, "benefits", batch_rows=2)
            pipeline.stream_labels(source, output, "benefits", batch_rows=1)
            self.assertEqual(len(pd.read_parquet(output)), 3)
            self.assertFalse(parts.exists())

    def build_inputs(self, root):
        root.mkdir(parents=True, exist_ok=True)
        companies = [1] * 6 + [2] * 6 + [4] * 6 + [3]
        years = [2018] * 5 + [2025] + [2018] * 6 + [2018] * 6 + [2025]
        ids = [f"post-{index:02d}" for index in range(len(companies))]
        structured = [json.dumps(["Mental Health"]), json.dumps(["Mental Health Extra"])] + ["[]"] * (len(ids) - 2)
        posts = pd.DataFrame({
            "ID": ids,
            "POSTED": [f"{year}-01-01" for year in years],
            "COMPANY": companies,
            "COMPANY_NAME": [f"Firm {company}" for company in companies],
            "MIN_EDULEVELS_NAME": ["Bachelor's degree"] * len(ids),
            "MIN_YEARS_EXPERIENCE": [2.0] * len(ids),
            "SALARY": [100_000.0] * len(ids),
            "REMOTE_TYPE_NAME": ["Remote"] + ["On-site"] * (len(ids) - 1),
            "STATE_NAME": ["California"] * len(ids),
            "NAICS_2022_6": [541511] * len(ids),
            "NAICS_2022_6_NAME": ["Custom Computer Programming Services"] * len(ids),
            "ONET": ["15-1252.00"] * len(ids),
            "BENEFIT_SUBCATEGORY_NAME": structured,
            "QUARTER": [1] * len(ids),
        })
        posts_path = root / "posts.csv"
        posts.to_csv(posts_path, index=False)

        bodies_path = root / "body.csv"
        pd.DataFrame({
            "ID": ids,
            "BODY": ["Tuition assistance, paid time off, and remote work."] * len(ids),
        }).to_csv(bodies_path, index=False)
        benefits = root / "benefits.parquet"
        remote = root / "remote.parquet"
        pipeline.stream_labels(bodies_path, benefits, "benefits", batch_rows=4)
        pipeline.stream_labels(bodies_path, remote, "remote", batch_rows=4)

        skills_path = root / "skills.csv"
        pd.DataFrame({
            "ID": [ids[0], ids[0], ids[10], ids[1]],
            "SKILL_SUBCATEGORY_NAME": [
                "Artificial Intelligence and Machine Learning (AI/ML)",
                "Artificial Intelligence and Machine Learning (AI/ML)",
                "Artificial Intelligence and Machine Learning (AI/ML)",
                "Other",
            ],
        }).to_csv(skills_path, index=False)
        ai_counts = root / "ai.parquet"
        pipeline.count_ai_skills(skills_path, ai_counts, batch_rows=1)

        sp500 = root / "sp500.csv"
        pd.DataFrame({"COMPANY": [2], "COMPANY_NAME": ["Firm 2"]}).to_csv(sp500, index=False)
        wham = root / "wham.csv"
        pd.DataFrame({
            "id": ids + ["other-country"],
            "country2": ["US"] * len(ids) + ["CA"],
            "wfh_wham_prob": [0.9] * len(ids) + [0.1],
            "wfh_wham": [True] * len(ids) + [False],
        }).to_csv(wham, index=False)
        return posts_path, benefits, remote, ai_counts, sp500, wham

    def test_assembly_uses_mbb_firm_rules_and_keeps_analysis_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs = self.build_inputs(Path(directory))
            frame, audit = assemble_dataset(
                *inputs[:4], sp500=inputs[4], wham=inputs[5], batch_rows=3,
            )
            by_id = frame.set_index("ID")
            self.assertEqual(by_id.loc["post-00", "FIRM_POSTING_COUNT"], 5)
            self.assertEqual(by_id.loc["post-00", "FIRM_SIZE_BUCKET"], "SMEs (<6)")
            self.assertEqual(by_id.loc["post-06", "FIRM_SIZE_BUCKET"], "S&P 500 firms")
            self.assertEqual(by_id.loc["post-12", "FIRM_SIZE_BUCKET"], "Large firms (>=6)")
            self.assertTrue(pd.isna(by_id.loc["post-18", "FIRM_SIZE_BUCKET"]))
            self.assertTrue(by_id.loc["post-00", "S_HEALTH_WELLNESS"])
            self.assertFalse(by_id.loc["post-01", "S_HEALTH_WELLNESS"])
            self.assertTrue(by_id.loc["post-00", "S_REMOTE"])
            self.assertEqual(by_id.loc["post-00", "AI_SKILL_COUNT"], 2)
            self.assertTrue(by_id.loc["post-10", "AI ROLE"])
            self.assertEqual(by_id.loc["post-00", "wfh_wham_prob"], 0.9)
            self.assertIn("REMOTE_KW_POSITION", frame)
            self.assertIn("CULTURE_POSITION", frame)
            self.assertEqual(audit["firm_size_base_year"], 2018)
            self.assertEqual(audit["firm_size_cutoff"], 6)

    def test_build_preserves_post_order_and_requires_explicit_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = self.build_inputs(root)
            posts = pd.read_csv(inputs[0])
            posts = posts.iloc[[2, 0, 1] + list(range(3, len(posts)))].reset_index(drop=True)
            posts.to_csv(inputs[0], index=False)
            output = root / "full" / "labeled_v2.parquet"
            build_dataset(
                output, posts=inputs[0], benefits=inputs[1], remote=inputs[2],
                ai_counts=inputs[3], sp500=inputs[4], wham=inputs[5], batch_rows=3,
            )
            self.assertEqual(pd.read_parquet(output)["ID"].tolist(), posts["ID"].tolist())
            metadata = json.loads(pipeline.metadata_path(output).read_text())
            self.assertTrue(metadata["complete"])
            self.assertEqual(metadata["rows"], len(posts))
            with self.assertRaisesRegex(FileExistsError, "use --overwrite"):
                build_dataset(
                    output, posts=inputs[0], benefits=inputs[1], remote=inputs[2],
                    ai_counts=inputs[3], sp500=inputs[4], wham=inputs[5],
                )

    def test_assembly_rejects_duplicate_posts_and_missing_body_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = self.build_inputs(root)
            posts = pd.read_csv(inputs[0])
            posts.loc[1, "ID"] = posts.loc[0, "ID"]
            posts.to_csv(inputs[0], index=False)
            with self.assertRaisesRegex(ValueError, "duplicate IDs"):
                assemble_dataset(*inputs[:4], sp500=inputs[4], wham=inputs[5])

            inputs = self.build_inputs(root / "second")
            benefits = pd.read_parquet(inputs[1]).iloc[:-1]
            remote = pd.read_parquet(inputs[2]).iloc[:-1]
            partial_benefits = root / "partial-benefits.parquet"
            partial_remote = root / "partial-remote.parquet"
            benefits.to_parquet(partial_benefits, index=False)
            remote.to_parquet(partial_remote, index=False)
            with self.assertRaisesRegex(ValueError, "no body labels"):
                assemble_dataset(inputs[0], partial_benefits, partial_remote, inputs[3],
                                 sp500=inputs[4], wham=inputs[5])
            frame, audit = assemble_dataset(
                inputs[0], partial_benefits, partial_remote, inputs[3],
                sp500=inputs[4], wham=inputs[5], allow_missing_body=True,
            )
            missing = frame.set_index("ID").loc["post-18"]
            self.assertFalse(missing["EDU_ASSISTANCE"])
            self.assertTrue(pd.isna(missing["EDU_ASSISTANCE_POSITION"]))
            self.assertEqual(audit["posts_without_body"], 1)


if __name__ == "__main__":
    unittest.main()
