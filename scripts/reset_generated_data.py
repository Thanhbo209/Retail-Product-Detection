"""Reset generated dataset artifacts while keeping raw data and source files.

Expected usage:
    python scripts/reset_generated_data.py --yes
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path


DELETE_PATHS = [
    Path("data/processed"),
    Path("data/yolo"),
    Path("outputs"),
    Path("runs"),
]
RECREATE_DIRS = [
    Path("data/processed/images"),
    Path("data/processed/labels"),
    Path("data/yolo"),
    Path("outputs/reports"),
    Path("outputs/visualizations"),
    Path("outputs/inference"),
]
LABEL_FIX_LOG_TEMPLATE_PATH = Path("outputs/reports/label_fix_log_template.csv")
LABEL_FIX_LOG_TEMPLATE_TEXT = (
    "image_filename,issue_type,source,action_taken,before_label_count,"
    "after_label_count,review_status,reviewed_by,review_date,notes\n"
    "046.jpg,false_positive_on_empty_shelf,model_prediction,"
    "deleted_wrong_boxes_and_added_missing_products,134,,needs_review,Thanh,,"
    "Bottom shelf gap should not be labeled as product\n"
)
NEVER_DELETE = {
    Path("data/raw"),
    Path("scripts"),
    Path("configs"),
    Path("docs"),
    Path("app"),
    Path("requirements.txt"),
}


@dataclass
class DeleteFailure:
    path: Path
    error: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Delete generated data folders and recreate a clean workspace."
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt.",
    )
    return parser.parse_args()


def is_inside_workspace(path: Path, workspace: Path) -> bool:
    try:
        path.resolve().relative_to(workspace.resolve())
    except ValueError:
        return False
    return True


def collect_delete_targets(workspace: Path) -> list[Path]:
    targets: list[Path] = []

    for relative_path in DELETE_PATHS:
        target = workspace / relative_path
        if target.exists():
            targets.append(target)

    for target in workspace.glob("cvat_import_*"):
        if target.is_dir():
            targets.append(target)

    for target in workspace.glob("cvat_import_*.zip"):
        if target.is_file():
            targets.append(target)

    return sorted(set(targets), key=lambda path: str(path))


def validate_delete_targets(targets: list[Path], workspace: Path) -> None:
    protected_paths = {(workspace / path).resolve() for path in NEVER_DELETE}
    for target in targets:
        resolved_target = target.resolve()
        if not is_inside_workspace(target, workspace):
            raise ValueError(f"Refusing to delete path outside workspace: {target}")
        if resolved_target in protected_paths:
            raise ValueError(f"Refusing to delete protected path: {target}")
        if resolved_target == workspace.resolve():
            raise ValueError("Refusing to delete the workspace root.")


def confirm_reset(targets: list[Path]) -> bool:
    print("The following generated paths will be deleted:")
    if targets:
        for target in targets:
            print(f"- {target}")
    else:
        print("- Nothing to delete")

    answer = input("Continue? Type 'yes' to confirm: ").strip().lower()
    return answer == "yes"


def delete_targets(targets: list[Path]) -> tuple[list[Path], list[DeleteFailure]]:
    deleted: list[Path] = []
    failures: list[DeleteFailure] = []
    for target in targets:
        try:
            if target.is_dir():
                shutil.rmtree(target)
                deleted.append(target)
            elif target.is_file():
                target.unlink()
                deleted.append(target)
        except OSError as exc:
            failures.append(DeleteFailure(path=target, error=str(exc)))
    return deleted, failures


def recreate_dirs(workspace: Path) -> list[Path]:
    recreated: list[Path] = []
    for relative_path in RECREATE_DIRS:
        directory = workspace / relative_path
        directory.mkdir(parents=True, exist_ok=True)
        recreated.append(directory)
    return recreated


def recreate_template_files(workspace: Path) -> list[Path]:
    template_path = workspace / LABEL_FIX_LOG_TEMPLATE_PATH
    template_path.write_text(LABEL_FIX_LOG_TEMPLATE_TEXT, encoding="utf-8")
    return [template_path]


def print_summary(
    deleted: list[Path],
    failures: list[DeleteFailure],
    recreated: list[Path],
    recreated_files: list[Path],
) -> None:
    print("\nDeleted paths")
    print("-" * 32)
    if deleted:
        for path in deleted:
            print(f"- {path}")
    else:
        print("- None")

    print("\nDelete failures")
    print("-" * 32)
    if failures:
        for failure in failures:
            print(f"- {failure.path}: {failure.error}")
        print("\nClose any app using these files, then rerun the reset command.")
    else:
        print("- None")

    print("\nRecreated folders")
    print("-" * 32)
    for path in recreated:
        print(f"- {path}")

    print("\nRecreated files")
    print("-" * 32)
    for path in recreated_files:
        print(f"- {path}")


def main() -> int:
    args = parse_args()
    workspace = Path.cwd()
    targets = collect_delete_targets(workspace)

    try:
        validate_delete_targets(targets, workspace)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1

    if not args.yes and not confirm_reset(targets):
        print("Reset cancelled.")
        return 0

    deleted, failures = delete_targets(targets)
    recreated = recreate_dirs(workspace)
    recreated_files = recreate_template_files(workspace)
    print_summary(deleted, failures, recreated, recreated_files)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
