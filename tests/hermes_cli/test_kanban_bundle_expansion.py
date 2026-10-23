"""Focused regression test for Kanban skill-bundle worker dispatch."""

from unittest import TestCase, main
from unittest.mock import patch

from hermes_cli.kanban_db_dispatch import _worker_skill_names


class WorkerSkillBundleExpansionTests(TestCase):
    def test_registered_bundle_expands_to_existing_member_skills(self):
        bundles = {
            "/polaris-tech-lead": {
                "slug": "polaris-tech-lead",
                "skills": ["codebase-inspection", "systematic-debugging"],
            }
        }
        with patch("agent.skill_bundles.get_skill_bundles", return_value=bundles):
            self.assertEqual(
                _worker_skill_names(["polaris-tech-lead", "codebase-inspection"]),
                ["codebase-inspection", "systematic-debugging"],
            )

    def test_unregistered_name_remains_visible_to_normal_missing_skill_path(self):
        with patch("agent.skill_bundles.get_skill_bundles", return_value={}):
            self.assertEqual(_worker_skill_names(["missing-role"]), ["missing-role"])


if __name__ == "__main__":
    main()
