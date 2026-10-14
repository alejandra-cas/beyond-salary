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

    def test_nan_and_blank_bodies_get_false_labels_and_no_positions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "body.csv"
            pd.DataFrame({"ID": ["a", "b", "c"], "BODY": [None, "   ", "Fully remote; tuition assistance."]}).to_csv(
                source, index=False)
            for kind, flags, positions in [
                ("benefits", pipeline.BENEFIT_NAMES, pipeline.POSITION_NAMES),
                ("remote", ["REMOTE_KW"], ["REMOTE_KW_POSITION"]),
            ]:
                output = root / f"{kind}.parquet"
                pipeline.stream_labels(source, output, kind, batch_rows=2)
                result = pd.read_parquet(output).set_index("ID")
                self.assertFalse(result.loc[["a", "b"], flags].to_numpy().any())
                self.assertTrue(result.loc[["a", "b"], positions].isna().to_numpy().all())
            self.assertTrue(pd.read_parquet(root / "remote.parquet").set_index("ID").loc["c", "REMOTE_KW"])
            self.assertTrue(pd.read_parquet(root / "benefits.parquet").set_index("ID").loc["c", "EDU_ASSISTANCE"])

    def test_assembly_rejects_duplicate_body_label_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            posts, benefits, remote, ai_counts, sp500, wham = self.build_inputs(root)
            for name, path in [("benefits", benefits), ("remote", remote)]:
                labels = pd.read_parquet(path)
                duplicated = root / f"duplicate-{name}.parquet"
                pd.concat([labels, labels.iloc[[0]]]).to_parquet(duplicated, index=False)
                arguments = {"benefits": benefits, "remote": remote, name: duplicated}
                with self.assertRaisesRegex(ValueError, "duplicate IDs"):
                    assemble_dataset(posts, arguments["benefits"], arguments["remote"], ai_counts,
                                     sp500=sp500, wham=wham)

    def test_checkpoint_resume_rejects_changed_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_body(root)
            output = root / "labels.parquet"
            original = pipeline.write_label_part

            def fail_after_first(batch, path, kind, positions, correction):
                if path.name != "part-000000.parquet":
                    raise RuntimeError("interrupted")
                return original(batch, path, kind, positions, correction)

            with patch.object(pipeline, "write_label_part", side_effect=fail_after_first):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    pipeline.stream_labels(source, output, "remote", batch_rows=1)
            self.write_body(root, ["Changed body.", "Another.", "Third.", "Fourth."])
            with self.assertRaisesRegex(ValueError, "Checkpoint input"):
                pipeline.stream_labels(source, output, "remote", batch_rows=1)
            self.assertFalse(output.exists())

    def test_cli_without_arguments_runs_legacy_pipeline(self):
        from scripts import label_benefits_remote

        for module, kind in [(label_benefits, "benefits"), (label_benefits_remote, "remote")]:
            with self.subTest(kind=kind):
                with patch.object(module, "legacy_main", return_value="legacy") as legacy, \
                        patch.object(pipeline, "stream_labels") as streamed, \
                        patch("sys.argv", [f"{module.__name__}.py"]):
                    self.assertEqual(module.main(), "legacy")
                legacy.assert_called_once_with()
                streamed.assert_not_called()

                with patch.object(module, "legacy_main") as legacy, \
                        patch("sys.argv", ["script.py", "--workers", "4"]), \
                        patch("sys.stderr"):
                    with self.assertRaises(SystemExit):
                        module.main()
                legacy.assert_not_called()

        with patch.object(label_benefits_remote, "legacy_main") as legacy, \
                patch.object(pipeline, "stream_labels") as streamed, \
                patch("sys.argv", ["script.py", "--stream", "--input", "in.csv", "--output", "out.parquet"]):
            label_benefits_remote.main()
        legacy.assert_not_called()
        self.assertFalse(streamed.call_args.kwargs["correct_remote_keywords"])

    def test_assembly_accepts_float_formatted_integer_codes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = self.build_inputs(root)
            posts = pd.read_csv(inputs[0])
            posts["NAICS_2022_6"] = posts["NAICS_2022_6"].astype(float)
            posts["COMPANY"] = posts["COMPANY"].astype(float)
            posts.loc[0, "QUARTER"] = np.nan
            posts.to_csv(inputs[0], index=False)
            self.assertIn("541511.0", inputs[0].read_text())
            frame, _ = assemble_dataset(*inputs[:4], sp500=inputs[4], wham=inputs[5])
            self.assertEqual(frame["COMPANY"].dtype, np.int64)
            self.assertEqual(frame["NAICS_2022_6"].dtype, np.int64)
            self.assertEqual(frame["QUARTER"].dtype, np.float64)
            self.assertEqual(frame.loc[0, "NAICS_2022_3_DIGIT"], "541")
            self.assertEqual(frame.set_index("ID").loc["post-06", "FIRM_SIZE_BUCKET"], "S&P 500 firms")


def write_legacy_comparison_inputs(root, rows=240):
    """Synthetic raw inputs covering the cases where the two paths could diverge."""
    rng = np.random.default_rng(7)
    ids = [f"{index:040x}" for index in range(rows)]  # SHA-1-like string IDs
    companies = rng.choice([0, 1, 2, 3, 4, 5, 99], size=rows)
    years = rng.choice([2018, 2019, 2023, 2025], size=rows)
    years[companies == 99] = 2025  # firm absent from 2018 keeps unknown size
    snippets = [
        "Tuition Assistance offered.", "We offer paid time off and PTO.", "Mental   health support",
        "PARENTAL LEAVE available", "inclusive culture", "This role is Fully Remote.",
        "remote available, remote", "Remote Available", "WORK FROM\nHOME option", "",
        "Health insurance and wellness program", "tuition reimbursement not offered",
    ]
    bodies = [" ".join(rng.choice(snippets, size=rng.integers(1, 5))) for _ in range(rows)]
    for index in range(0, rows, 29):
        bodies[index] = None
    structured = [json.dumps(["Mental Health"]), json.dumps(["mental health"]),
                  json.dumps(["Mental Health Extra"]), json.dumps(["Flexible Work Schedules", "Mentorships"]),
                  "[]", None, "not json"]
    posts = pd.DataFrame({
        "ID": ids,
        "POSTED": [f"{year}-0{rng.integers(1, 9)}-15" for year in years],
        "TITLE_RAW": "title", "COMPANY": companies, "COMPANY_NAME": [f"Firm {c}" for c in companies],
        "MIN_EDULEVELS_NAME": "Bachelor's degree", "EMPLOYMENT_TYPE_NAME": "Full-time",
        "MIN_YEARS_EXPERIENCE": rng.choice([np.nan, 0, 1, 3, 7, 15, 25], size=rows),
        "SALARY": rng.choice([np.nan, 50_000.0, 120_000.0], size=rows),
        "REMOTE_TYPE_NAME": rng.choice(["Remote", "Hybrid Remote", "On-site", None], size=rows),
        "CITY_NAME": "city", "STATE_NAME": "California",
        "NAICS_2022_6": rng.choice([541511, 622110, 921110], size=rows),
        "NAICS_2022_6_NAME": "industry", "ONET": rng.choice(["15-1252.00", "29-1141.00", None], size=rows),
        "ONET_NAME": "occupation", "BENEFIT_NAME": "benefit",
        "BENEFIT_SUBCATEGORY_NAME": rng.choice(structured, size=rows),
        "BENEFIT_CATEGORIES_NAME": "category", "YEAR": years, "QUARTER": rng.integers(1, 5, size=rows),
    })
    paths = {name: root / f"{name}.csv" for name in ["posts", "body", "skills", "sp500", "wham"]}
    posts.to_csv(paths["posts"], index=False)
    pd.DataFrame({"ID": ids, "BODY": bodies}).to_csv(paths["body"], index=False)
    ai = "Artificial Intelligence and Machine Learning (AI/ML)"
    pd.DataFrame({"ID": rng.choice(ids, size=360), "SKILL_SUBCATEGORY_NAME": rng.choice([ai, "Other"], size=360)}
                 ).to_csv(paths["skills"], index=False)
    pd.DataFrame({"COMPANY": [2, 5], "COMPANY_NAME": ["Firm 2", "Firm 5"]}).to_csv(paths["sp500"], index=False)
    pd.DataFrame({"id": ids[::2] + ["non-us"], "country2": ["US"] * len(ids[::2]) + ["CA"],
                  "wfh_wham_prob": 0.5, "wfh_wham": True}).to_csv(paths["wham"], index=False)
    return paths


class LegacyEquivalenceTests(unittest.TestCase):
    # Columns the full-data builder intentionally omits; only exploratory notebooks use them.
    OMITTED = {"BODY", "TITLE_RAW", "EMPLOYMENT_TYPE_NAME", "CITY_NAME", "ONET_NAME",
               "BENEFIT_NAME", "BENEFIT_CATEGORIES_NAME"}

    def test_streaming_path_matches_legacy_prepare_and_label_scripts(self):
        from scripts import label_benefits_remote, prepare_data

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = write_legacy_comparison_inputs(root)
            processed = root / "processed"
            processed.mkdir()
            legacy_paths = {
                "posts_csv": raw["posts"], "skills_csv": raw["skills"], "body_csv": raw["body"],
                "wham_data": raw["wham"], "sp500_csv": raw["sp500"],
                "output_parquet": processed / "data_v1.parquet", "sp500_audit_csv": processed / "audit.csv",
            }
            with patch.object(prepare_data, "load_data_paths", return_value=legacy_paths), \
                    patch.object(label_benefits, "input_path", processed / "data_v1.parquet"), \
                    patch.object(label_benefits, "output_path", processed / "labeled_v1.parquet"), \
                    patch.object(label_benefits_remote, "input_path", processed / "labeled_v1.parquet"), \
                    patch.object(label_benefits_remote, "output_path", processed / "labeled_v2.parquet"), \
                    patch("sys.stdout"):
                prepare_data.main()
                label_benefits.legacy_main()
                label_benefits_remote.legacy_main()
                pipeline.stream_labels(raw["body"], root / "benefits.parquet", "benefits", batch_rows=17, workers=2)
                pipeline.stream_labels(raw["body"], root / "remote.parquet", "remote", batch_rows=23, workers=2)
                pipeline.count_ai_skills(raw["skills"], root / "ai.parquet", batch_rows=50)
                streamed, audit = assemble_dataset(
                    raw["posts"], root / "benefits.parquet", root / "remote.parquet", root / "ai.parquet",
                    sp500=raw["sp500"], wham=raw["wham"], batch_rows=31,
                )
            legacy = pd.read_parquet(processed / "labeled_v2.parquet")

            self.assertEqual(streamed["ID"].tolist(), legacy["ID"].tolist())
            self.assertEqual(set(legacy.columns) - set(streamed.columns), self.OMITTED)
            self.assertEqual(set(streamed.columns) - set(legacy.columns), set())
            for column in legacy.columns.drop(list(self.OMITTED)):
                with self.subTest(column=column):
                    expected, actual = legacy[column], streamed[column]
                    if pd.api.types.is_float_dtype(expected) or pd.api.types.is_float_dtype(actual):
                        np.testing.assert_allclose(actual.astype(float), expected.astype(float), equal_nan=True)
                    else:
                        self.assertEqual(actual.astype(object).where(actual.notna(), None).tolist(),
                                         expected.astype(object).where(expected.notna(), None).tolist())

            # Make sure the synthetic data actually exercised the risky cases.
            self.assertGreater(audit["ai_postings"], 0)
            self.assertTrue(legacy["BODY"].isna().any())
            self.assertTrue(legacy["REMOTE_KW"].any() and not legacy["REMOTE_KW"].all())
            self.assertTrue(legacy.loc[legacy["COMPANY"].eq(99), "FIRM_SIZE_BUCKET"].isna().all())
            self.assertIn("S&P 500 firms", set(legacy["FIRM_SIZE_BUCKET"].dropna()))
            self.assertTrue(legacy["wfh_wham_prob"].notna().any() and legacy["wfh_wham_prob"].isna().any())


if __name__ == "__main__":
    unittest.main()
